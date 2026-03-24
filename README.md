# Worst-case low-rank approximations (wcPCA)

This repository contains the code for the research paper [Worst-case low-rank approximations](https://arxiv.org/abs/2603.11304).

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
```

3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

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

### Comparison with other optimization schemes (Appendix B.2.1.)

See [`comparison/`](https://github.com/anyafries/wcPCA/tree/main/comparison) for details. Compares our projected gradient descent against [FairPCA](https://github.com/SDPforAll/multiCriteriaDimReduction) (Tantipongpipat et al., NeurIPS 2019) and [StablePCA](https://github.com/zywang0701/StablePCA) (Wang et al., 2026).

### Applications

Section 6.1. corresponds to the directory `fluxnet/`. 
Data are available upon reasonable request.
Running the following produces Figure 8, Figure C.i, and Figure C.ii.

```bash
cd application/fluxnet
python fluxnet_analysis.py          # or --rerun
```

Section 6.2. corresponds to the directory `ecosystem/`. 
Data is available from the original paper of [Migliavacca et al. (2021)](https://www.nature.com/articles/s41586-021-03939-9): navigate to https://zenodo.org/records/5153538, where their code and data is stored. Download `data/InputDataMigliavacca2021.csv` and copy this to `applications/ecosystem/data`. Then, run `prepare_data.R` to add continent information.
Thereafter, running the following produces Figure 9 and Figure C.iii.

```bash
cd ../ecosystem
python ecosystem_analysis.py        # or --rerun
```

## Citation

If you use this code, please cite:

```bibtex
@misc{Fries2026,
      title={Worst-case low-rank approximations}, 
      author={Anya Fries and Markus Reichstein and David Blei and Jonas Peters},
      year={2026},
      eprint={2603.11304},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2603.11304}, 
}
```

## License

See [LICENSE](LICENSE).
