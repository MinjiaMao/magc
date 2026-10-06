import os
import argparse
import numpy as np
import torch
import spike_ffn as S

CHANNELS = [("894", 894), ("0", 0), ("random_1742", 1742)]


def make_base(a, hidden, seed):
    """One Pareto(a) base. Peak = largest RAW value (positive) fixed before signs."""
    np.random.seed(seed)
    pareto = np.random.pareto(a, size=(hidden,)).astype(np.float32)   # >= 0
    max_idx = int(np.argmax(pareto))
    max_val = float(pareto[max_idx])
    signs = np.random.choice([-1.0, 1.0], size=pareto.shape).astype(np.float32)
    return pareto * signs, max_idx, max_val


def place_and_norm(base, max_idx, max_val, channel):
    s = base.copy()
    s[max_idx] = s[channel]          # displaced value -> old argmax slot
    s[channel] = max_val             # fixed positive peak -> target channel
    s = s / np.sqrt(np.mean(s ** 2) + 1e-6)     # RMSNorm
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=float, default=1.0, help="Pareto shape (main text uses 1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="figures/figure2")
    args = ap.parse_args()

    tok, model = S.load_model()
    mlp = S.get_language_model(model).layers[S.SPIKE_LAYER].mlp
    dev = next(mlp.parameters()).device

    base, max_idx, max_val = make_base(args.a, S.D_MODEL, args.seed)

    # pass 1: compute all outputs, then fix the shared y-axis from the MAGC output
    results = []
    for name, ch in CHANNELS:
        s = place_and_norm(base, max_idx, max_val, ch)
        with torch.no_grad():
            out = mlp(torch.from_numpy(s).to(dev).unsqueeze(0).unsqueeze(0))
        results.append((name, ch, s, out.detach().cpu().numpy().reshape(-1)))
    out894 = next(o for n, c, s, o in results if c == S.MAGC)
    yrange = (float(out894.min()), float(out894.max()))

    os.makedirs(args.out, exist_ok=True)
    for name, ch, s, ovec in results:
        legend_loc = 'upper left' if name.startswith('random') else 'upper right'
        S.plot_embedding_vector(
            s, os.path.join(args.out, f"fig2_{name}_input.png"),
            title=f"Input Pareto(a={args.a:g}), peak@ch {ch}",
            highlight_idx=ch, legend_loc=legend_loc)
        S.plot_embedding_vector(
            ovec, os.path.join(args.out, f"fig2_{name}_output.png"),
            title=f"Spike FFN output (peak@ch {ch})", ylim=yrange)
        mx = int(np.argmax(np.abs(ovec)))
        print(f"  peak@ch {ch:>5} ({name:>11}) -> out max |{ovec[mx]:+.3f}| @ch {mx}")
    print("Figure 2 done.")


if __name__ == "__main__":
    main()
