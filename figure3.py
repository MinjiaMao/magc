"""Figure 3: Visualization results for LLaMA-3.2-1B.

(a) alpha vector when i* = 894 (Theorem 4.1), with the largest-magnitude entry
    highlighted -- it is highly concentrated at j* = 1417.
(b) omega_{1417}: the dominant down-projection column (normalized). It is highly
    co-linear with the FFN output y (compare to Figure 1b) and carries the
    massive-activation pattern.
(c) omega_0 and (d) omega_1: other columns, with lower peak magnitude and no
    massive-activation pattern.
"""
import os
import argparse
import numpy as np
import spike_ffn as S


def omega(Wd, Wd_norm, j):
    """Unit-normalized down-projection column omega_j = W_down[:, j] / ||.||."""
    return Wd[:, j] / (Wd_norm[j] + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/figure3")
    args = ap.parse_args()

    tok, model = S.load_model()
    Wg, Wu, Wd = S.get_ffn_matrices(model)
    Wd_norm = np.linalg.norm(Wd, axis=0)

    alpha = S.alpha_vector(Wg, Wu, Wd_norm, channel=S.MAGC, positive=True)
    jstar = int(np.argmax(np.abs(alpha)))
    print(f"j* (argmax |alpha|) = {jstar}  (paper: {S.JSTAR}),  alpha[j*] = {alpha[jstar]:.4f}")

    os.makedirs(args.out, exist_ok=True)
    # (a) alpha vector, highlight only the dominant column j*
    S.plot_embedding_vector(
        alpha, os.path.join(args.out, "fig3a_alpha.png"),
        title="Positive Gate alpha vector (i=MAGC)",
        highlight_idx=[jstar], legend_loc='upper right', legend_bbox=(0.98, 0.80))
    # (b,c,d) down-projection columns omega_j
    for tag, j in [("fig3b_omega1417", jstar), ("fig3c_omega0", 0), ("fig3d_omega1", 1)]:
        w = omega(Wd, Wd_norm, j)
        S.plot_embedding_vector(
            w, os.path.join(args.out, f"{tag}.png"),
            title=f"$\\omega_{{{j}}}$",
            highlight_idx=[int(np.argmax(np.abs(w)))])
    print("Figure 3 done.")


if __name__ == "__main__":
    main()
