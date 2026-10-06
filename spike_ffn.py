"""Shared utilities for reproducing Figures 1-3 of the Single-Channel Spike paper
on LLaMA-3.2-1B.

Everything here is specific to LLaMA-3.2-1B:
    d = 2048, m = 8192, spike FFN = block 1, MAGC i* = 894, dominant col j* = 1417.

Contents
--------
- model loading + language-model locator
- random token sampling (Section 3 recipe, seed 42)
- capture of the spike-FFN per-token input h and output y
- FFN weight-matrix accessors and the alpha coefficient vector (Theorem 4.1)
- 2D plotter  : plot_embedding_vector  (Figures 2, 3)
- 3D plotter  : plot_vectors_3d        (Figure 1)
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ----------------------------- constants -----------------------------
MODEL_ID = "meta-llama/Llama-3.2-1B"
SPIKE_LAYER = 1       # the spike FFN is the FFN in the 2nd transformer block (index 1)
MAGC = 894            # massive activation gating channel i*
JSTAR = 1417          # dominant down-projection column j*
D_MODEL = 2048
M_INTER = 8192


def get_cache_dir():
    return os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE") \
        or "/data/mjmao/ood/hf_models"


# ----------------------------- CJK font -----------------------------
def setup_cjk_font():
    """Register a CJK-capable font so multilingual random tokens render instead
    of tofu boxes (random token sequences can be multilingual)."""
    import matplotlib.font_manager as fm
    for path in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                 "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"):
        if os.path.exists(path):
            try:
                fm.fontManager.addfont(path)
                name = fm.FontProperties(fname=path).get_name()
                plt.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                return name
            except Exception:
                continue
    return None


# ----------------------------- model -----------------------------
def load_model(attn_eager=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cache = get_cache_dir()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache, use_fast=True)
    kw = dict(cache_dir=cache, torch_dtype=torch.float32, device_map='cuda')
    if attn_eager:
        kw['attn_implementation'] = 'eager'
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw).eval()
    return tok, model


def get_language_model(model):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model
    raise RuntimeError("cannot locate decoder layers")


# ----------------------------- random tokens -----------------------------
def random_token_rows(tokenizer, n_tokens=5, n_samples=5, seed=42):
    """Section 3 recipe: sample `n_samples` sequences of `n_tokens` random ids.
    Returns an (n_samples, n_tokens) int array (row-major, reproducible)."""
    rng = np.random.default_rng(seed)
    vocab = len(tokenizer.get_vocab())
    return rng.integers(low=0, high=vocab, size=(n_samples, n_tokens), dtype=np.int64)


def token_labels(tokenizer, ids):
    labels = []
    for tid in ids:
        dec = tokenizer.decode([int(tid)])
        tok = tokenizer.convert_ids_to_tokens(int(tid))
        label = dec if dec.strip() != '' else tok
        if tokenizer.bos_token_id is not None and int(tid) == tokenizer.bos_token_id:
            label = '<bos>'
        if '�' in label:                     # incomplete-byte token
            label = 'U+FFFD'
        labels.append(label.replace('\n', '\\n'))
    return labels


# ----------------------------- capture spike FFN -----------------------------
def capture_spike_ffn(model, ids):
    """Run one token sequence and return the spike-FFN per-token input h and
    output y, both shape (seq_len, d). `ids` is a 1D list/array of token ids.

    h = post_attention_layernorm(x + attn_out) at SPIKE_LAYER (the normalized
    input actually fed to the MLP); y = MLP output at SPIKE_LAYER.
    """
    lm = get_language_model(model)
    layer = lm.layers[SPIKE_LAYER]
    store = {}
    h1 = layer.self_attn.register_forward_hook(
        lambda m, i, o: store.__setitem__('attn', o[0] if isinstance(o, tuple) else o))
    h2 = layer.mlp.register_forward_hook(
        lambda m, i, o: store.__setitem__('mlp', o))
    try:
        input_ids = torch.tensor([list(map(int, ids))], dtype=torch.long,
                                 device=next(model.parameters()).device)
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
        x = out.hidden_states[SPIKE_LAYER]                 # residual input (1,seq,d)
        u = store['attn']
        ln = layer.post_attention_layernorm
        v = ln((x + u).to(ln.weight.device))
        h = v[0].detach().cpu().float().numpy()
        y = store['mlp'][0].detach().cpu().float().numpy()
    finally:
        h1.remove(); h2.remove()
    return h, y


# ----------------------------- FFN weights -----------------------------
def get_ffn_matrices(model):
    mlp = get_language_model(model).layers[SPIKE_LAYER].mlp
    Wg = mlp.gate_proj.weight.detach().cpu().float().numpy()   # (m, d)
    Wu = mlp.up_proj.weight.detach().cpu().float().numpy()     # (m, d)
    Wd = mlp.down_proj.weight.detach().cpu().float().numpy()   # (d, m)
    return Wg, Wu, Wd


def alpha_vector(Wg, Wu, Wd_norm, channel=MAGC, positive=True):
    """Coefficient vector alpha_j for input channel `channel` (Theorem 4.1):
        alpha_j = relu(+/- W_gate[j, i]) * W_up[j, i] * ||W_down[:, j]||.
    LLaMA-3.2-1B has a positive gating direction."""
    g = Wg[:, channel]
    up = Wu[:, channel]
    gated = np.where(g > 0, g, 0.0) if positive else np.where(g < 0, g, 0.0)
    return gated * up * Wd_norm


# ----------------------------- 2D plotter -----------------------------
def plot_embedding_vector(vec, out_path, title=None, color='blue', ylabel='Value',
                          highlight_idx=None, report_norm=False, ylim=None,
                          title_loc='center', legend_loc='upper right', legend_bbox=None):
    """Plot a 1D vector as a thin stem/scatter, optionally highlighting indices."""
    vec = np.asarray(vec, dtype=np.float64)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(vec))
    ax.plot(x, vec, color=color, linewidth=0.2)
    ax.scatter(x, vec, color=color, s=0.2)
    if ylim is not None:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title, loc=title_loc)
    ax.set_xlabel('Dimension')
    ax.set_ylabel(ylabel)

    if highlight_idx is not None:
        idxs = highlight_idx if isinstance(highlight_idx, (list, tuple)) else [highlight_idx]
        for idx in idxs:
            ax.scatter([idx], [vec[idx]], color='red', s=50, zorder=5,
                       label=f'Idx {idx}: {vec[idx]:.4f}')

    handles, labels = ax.get_legend_handles_labels()
    if report_norm:
        handles.append(Line2D([], [], linestyle='None', marker='',
                              label=f'Norm: {np.linalg.norm(vec):.4f}'))
        labels.append(f'Norm: {np.linalg.norm(vec):.4f}')
    if handles:
        anchor = {'upper right': (0.98, 0.98), 'upper left': (0.02, 0.98),
                  'lower right': (0.98, 0.02), 'lower left': (0.02, 0.02)}.get(legend_loc, (0.98, 0.98))
        if legend_bbox is not None:
            anchor = legend_bbox
        ax.legend(handles, labels, loc=legend_loc, bbox_to_anchor=anchor, fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', format='png')
    plt.close(fig)
    print(f"  saved {out_path}")


# ----------------------------- 3D plotter -----------------------------
def plot_vectors_3d(mat, out_path, token_labels, title=None, highlight_idx=None,
                    zlabel='Value', base_color='#1f6fe0', highlight_color='red',
                    elev=22, azim=-62, base_lw=1.4, hl_lw=2.8, zticks=None,
                    z_ends_only=False, mark_dim_max=False, title_y=0.98,
                    draw_baseline=False, baseline_lw=1.2, box_aspect=(0.7, 0.85, 0.6)):
    """Stack of 1D vectors (one per token) as a 3D stem plot: tokens on x (front),
    channel index on y (depth), value on z."""
    mat = np.asarray(mat, dtype=np.float64)
    T, D = mat.shape
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    hi = set()
    if highlight_idx is not None:
        hi = {int(i) for i in (highlight_idx if isinstance(highlight_idx, (list, tuple, np.ndarray))
                               else [highlight_idx])}
    hi = {c for c in hi if 0 <= c < D}

    dim = np.arange(D)
    fig = plt.figure(figsize=(6.5, 6))
    ax = fig.add_subplot(111, projection='3d')

    if draw_baseline:
        for t in range(T):
            ax.plot([t, t], [0, D - 1], [0, 0], color=base_color, linewidth=baseline_lw, zorder=1)
    for t in range(T):
        z = mat[t]
        segs = np.empty((D, 2, 3))
        segs[:, 0, 0] = t; segs[:, 0, 1] = dim; segs[:, 0, 2] = 0.0
        segs[:, 1, 0] = t; segs[:, 1, 1] = dim; segs[:, 1, 2] = z
        colors = [base_color] * D
        lws = np.full(D, base_lw)
        for c in hi:
            colors[c] = highlight_color
            lws[c] = hl_lw
        ax.add_collection3d(Line3DCollection(segs, colors=colors, linewidths=lws.tolist()))

    zmin = min(0.0, float(mat.min())); zmax = max(0.0, float(mat.max()))
    if zticks is not None:
        zmin = min(zmin, float(min(zticks))); zmax = max(zmax, float(max(zticks)))
    pad = 0.05 * (zmax - zmin + 1e-9)
    ax.set_xlim(-0.5, T - 0.5); ax.set_ylim(0, D); ax.set_zlim(zmin - pad, zmax + pad)
    if box_aspect is not None:
        ax.set_box_aspect(box_aspect)

    ax.set_xticks(np.arange(T))
    ax.set_xticklabels(token_labels, fontsize=9, rotation=30, ha='right', va='top',
                       rotation_mode='anchor')
    dim_ticks = [0] + sorted(hi) + ([D - 1] if mark_dim_max else [])
    ax.set_yticks(sorted(set(dim_ticks)))
    hi_str = {str(int(c)) for c in hi}
    for lbl in ax.get_yticklabels():
        if lbl.get_text() in hi_str:
            lbl.set_color(highlight_color)
    ax.set_ylabel('Dimension')

    if zticks is not None:
        ax.set_zticks(list(zticks))
    elif z_ends_only:
        from matplotlib.ticker import MaxNLocator
        tv = MaxNLocator(nbins='auto', steps=[1, 2, 2.5, 5, 10]).tick_values(
            float(mat.min()), float(mat.max()))
        tv = [t for t in tv if (zmin - pad) <= t <= (zmax + pad)]
        if len(tv) >= 2:
            ends = [tv[0], 0, tv[-1]] if tv[0] < 0 < tv[-1] else [tv[0], tv[-1]]
            ax.set_zticks(ends)

    ax.set_zlabel('')
    ax.text2D(0.90, 0.55, zlabel, transform=ax.transAxes, rotation=90,
              rotation_mode='anchor', ha='center', va='center', fontsize=12)
    if title:
        ax.set_title(title, y=title_y)
    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.15, format='png')
    plt.close(fig)
    print(f"  saved {out_path}")
