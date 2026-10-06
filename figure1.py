"""Figure 1: Inputs and Outputs of the Spike FFN (LLaMA-3.2-1B).

(a) Value of input h to the spike FFN, MAGC (894) highlighted in red.
(b) Magnitude of output |y| of the spike FFN; the first token exhibits
    massive activations.

Rendered as 3D stem plots (one row per token) for the random 5-token example
used in the paper (seed 42, sample #1, whose first token is " stejně" and whose
894th-channel values are 8.807, 0.814, -0.763, 0.335, 1.029).
"""
import os
import argparse
import numpy as np
import spike_ffn as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_index", type=int, default=1,
                    help="which random sample row to show (paper uses #1)")
    ap.add_argument("--out", default="figures/figure1")
    args = ap.parse_args()

    S.setup_cjk_font()
    tok, model = S.load_model()
    rows = S.random_token_rows(tok, n_tokens=5, n_samples=5, seed=42)
    ids = rows[args.sample_index].tolist()
    labels = S.token_labels(tok, ids)
    h, y = S.capture_spike_ffn(model, ids)

    print(f"tokens: {labels}")
    print(f"h[MAGC={S.MAGC}] per token: " +
          ", ".join(f"{v:.3f}" for v in h[:, S.MAGC]))

    os.makedirs(args.out, exist_ok=True)
    # (a) input h, highlight MAGC 894
    S.plot_vectors_3d(h, os.path.join(args.out, "fig1a_input_h.png"), labels,
                      title="FFN input $h$", highlight_idx=S.MAGC, zlabel='h',
                      z_ends_only=True, mark_dim_max=True)
    # (b) output magnitude |y|
    S.plot_vectors_3d(np.abs(y), os.path.join(args.out, "fig1b_output_y.png"), labels,
                      title="FFN output $|y|$", highlight_idx=None, zlabel='magnitude',
                      base_lw=4.5, zticks=[0, 200, 400], mark_dim_max=True,
                      draw_baseline=True, baseline_lw=1.2)
    print("Figure 1 done.")


if __name__ == "__main__":
    main()
