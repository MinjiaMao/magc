import argparse
import numpy as np
import torch
import spike_ffn as S


def identify(model, tokenizer, n_samples=100, n_tokens=5, seed=42, verbose=True):
    lm = S.get_language_model(model)
    layers = lm.layers
    attn_out = {}
    hooks = []
    for i, layer in enumerate(layers):
        hooks.append(layer.self_attn.register_forward_hook(
            lambda m, inp, o, i=i: attn_out.__setitem__(
                i, o[0] if isinstance(o, tuple) else o)))

    rows = S.random_token_rows(tokenizer, n_tokens, n_samples, seed)
    input_ids = torch.tensor(rows, dtype=torch.long,
                             device=next(model.parameters()).device)
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
        hidden_states = out.hidden_states

        # (1) locate the massive-activation block in the residual stream
        ma_block = None
        for l in range(len(layers)):
            mx = float(hidden_states[l][:, 0, :].abs().max())
            if verbose and mx > 50:
                print(f"  block {l:2d}: max|residual_token0| = {mx:8.2f}")
            if mx > 100:
                ma_block = l
                break
        if ma_block is None:
            raise RuntimeError("no massive activation (>100) found in the residual stream")
        spike = ma_block - 1

        # (2) MAGC from the spike-FFN inputs
        x = hidden_states[spike]                              # (N, seq, d)
        u = attn_out[spike]
        ln = layers[spike].post_attention_layernorm
        v = ln((x + u).to(ln.weight.device))
        h = v.mean(dim=0).detach().cpu().float().numpy()      # (seq, d)
        h0, h1 = h[0], h[1]
        diff = np.abs(h0 - h1)
        istar = int(np.argmax(diff))
        positive = bool((h0[istar] - h1[istar]) > 0)
    finally:
        for hk in hooks:
            hk.remove()

    return dict(spike_block=spike, ma_block=ma_block, magc=istar,
                positive=positive, h0=h0, h1=h1, diff=diff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--n_tokens", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--topk", type=int, default=5, help="show the top-k |h0-h1| channels")
    args = ap.parse_args()

    tok, model = S.load_model()
    print("Scanning blocks for the spike FFN (outlier channel > 100):")
    res = identify(model, tok, args.n_samples, args.n_tokens, args.seed)

    i = res['magc']
    print(f"\nSpike FFN block : {res['spike_block']}")
    print(f"MAGC i*         : {i}  (gating direction: {'+' if res['positive'] else '-'})")
    print(f"  h0(i*)={res['h0'][i]:.4f}  h1(i*)={res['h1'][i]:.4f}  "
          f"|h0-h1|={res['diff'][i]:.4f}")
    order = np.argsort(res['diff'])[::-1][:args.topk]
    print(f"\nTop-{args.topk} channels by |h0 - h1|:")
    print(f"  {'rank':>4} {'chan':>6} {'h0':>9} {'h1':>9} {'|h0-h1|':>9}")
    for r, j in enumerate(order):
        mark = '  <- MAGC' if j == i else ''
        print(f"  {r+1:>4} {int(j):>6} {res['h0'][j]:>9.4f} {res['h1'][j]:>9.4f} "
              f"{res['diff'][j]:>9.4f}{mark}")


if __name__ == "__main__":
    main()
