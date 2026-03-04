# Worst-case low-rank approximation (wcPCA)

This repository contains all code to reproduce the simulations, method comparisons, and real-data applications of the paper *Worst-case low rank approximation*.

---

## Repository Structure

```
worst-case-low-rank-approx/
├── simulations/           # Synthetic experiments validating theory and empirical behavior
├── comparison/            # Benchmarks against Fair PCA and StablePCA
├── application/           # Real-data applications (FLUXNET, ecosystem data)
├── jmlr.mplstyle          # Matplotlib style for publication figures
└── README.md
```

All scripts cache results in a `results/` subdirectory and save figures to `figures/`. Pass `--rerun` to force recomputation from scratch.

### Simulations

| Section/Figure in the paper | Script | Figure file |
|---|---|---|
| Figure 2 | `illustrative_example.py` | `figures/eg-minpca.png` |
| Section 5.1.1. (Figure 3) | `sim1_theoretical.py` | `figures/sim1_theoretical.png` |
| Section 5.1.2. (Figure 4) | `sim2_avg_vs_wc.py` | `figures/sim2_avg_vs_wc_relative.png` |
| Section 5.1.3. (Figure 5) | `sim3_finite_sample.py` | `figures/sim3_finite_sample.png` |
| Section 5.1.4. (Figure 6) | `sim4_het_noise.py` | `figures/sim4_het_noise_*.pdf` |
| Section 5.1.5. (Figure 7 and B.iii) | `sim5_maxmc.py` | `figures/sim5_*.png` |
<!-- | Section 5.1.2. (Figure 4) | `sim_maxrcs_two_solutions.py` | `figures/` -->

Usage:
```bash
cd simulations

python illustrative_example.py
python sim1_theoretical.py          # or --rerun to recompute
python sim2_avg_vs_wc.py
python sim3_finite_sample.py
python sim4_het_noise.py
python sim5_maxmc.py
```

### Comparison (Appendix B.2.1.)

```diff
! TODO: document adding the Fair PCA repo
```

Benchmarks maxRCS against published baselines on the objectives MM\_Var and MM\_Loss, varying the dimension and number of environments.

| Script | Description |
|---|---|
| `comparison.py` | Orchestrates all methods and produces combined comparison figures |
| `minpca_sim.py` | Runs minPCA (projected gradient descent) |
| `fairpca.py` | Runs FairPCA via SDP and multiplicative weights (Tantipongpipat et al., NeurIPS 2019) |
| `stablepca.py` | Runs StablePCA |

```diff
! TODO: StablePCA ref + add links.
```

Usage: 
```bash
cd comparison
python comparison.py                # runs all methods and produces combined figures
```

Individual methods can also be run separately:

```bash
python minpca_sim.py  [--rerun] [--objective {MM_Var,MM_Loss}] [--p {10,50}] [--n_envs {5,50}]
python fairpca.py     [--rerun]
python stablepca.py   [--rerun]
```

### Applications

```diff
! TODO: document data access/decide what to do here
```

Section 6.1. correspond to the directory `fluxnet/`. Running the following produces Figure 8, Figure C.i, and Figure C.ii.

```bash
cd application/fluxnet
python fluxnet_analysis.py          # or --rerun
```

Section 6.2. correspond to the directory `ecosystem/`. Running the following produces Figure 9 and Figure C.iii.

```bash
cd ../ecosystem
python ecosystem_analysis.py        # or --rerun
```

---

## Setup

1. Create a virtual environment

```bash
python -m venv venv_wcPCA
source venv_wcPCA/bin/activate
```

2. Install the minPCA package

```bash
# Navigate to where you want to clone minPCA
git clone https://github.com/anyafries/minPCA.git
cd minPCA
pip install . --extra-index-url https://download.pytorch.org/whl/cpu
cd ..
```

3. Install remaining dependencies

```diff
! TODO: remove fancyimpute
```

```bash
pip install -r requirements.txt
```
---

## Citation

If you use this code, please cite:

```diff
! TODO: add
```

<!--```bibtex
```-->

---

## License

See [LICENSE](LICENSE).
