# MAGC

Source code for Massive Activation Gating Channel in Large Language Models

Reproduce figures on
**LLaMA-3.2-1B** (`d = 2048`, `m = 8192`, spike FFN = block 1,
MAGC `i* = 894`, dominant column `j* = 1417`).

## Files
| File | What it generates |
|------|-------------------|
| `identify_magc.py` | derive the spike block and MAGC `i*` from activations (instead of assuming them); recovers block 1, `i* = 894`, direction `+` |、
| `figure1.py` | **Figure 1** — spike-FFN input `h` (MAGC 894 in red) and output magnitude `y`, 3D per token |
| `figure2.py` | **Figure 2** — Pareto(a=1) input with the max placed in channel 894 / 0 / 1742, and the resulting spike-FFN outputs |
| `figure3.py` | **Figure 3** — `alpha` vector (i\*=894, highlighting j\*=1417) and the down-projection columns `omega_{1417}`, `omega_0`, `omega_1` |

## Run
```bash
# environment: transformers (5.x used here) + torch, a CUDA GPU
python figure1.py      # -> figures/figure1/{fig1a_input_h.png, fig1b_output_y.png}
python figure2.py      # -> figures/figure2/fig2_{894,0,random_1742}_{input,output}.png
python figure3.py      # -> figures/figure3/{fig3a_alpha, fig3b_omega1417, fig3c_omega0, fig3d_omega1}.png
```
Outputs are written under `figures/`. 
