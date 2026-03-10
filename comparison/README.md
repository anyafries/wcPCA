# Compare different optimization schemes for wcPCA

Compares projected gradient descent (PGD) against published baselines on the objectives minPCA (MM\_Var) and maxRegret (MM\_Loss), varying the dimension and number of environments.

## Optimization schemes

We compare against 
1. *Multi-Criteria Dimensionality Reduction with Applications to Fairness* ([NeurIPS paper](https://papers.nips.cc/paper/) and [arXiv version](https://arxiv.org/abs/1902.11281v2)), and use adapted code (with permission) from [their GitHub](https://github.com/SDPforAll/multiCriteriaDimReduction), stored in `FairPCA/`. 
2. *StablePCA: Distributionally Robust Learning of Shared Representations from Multi-Source Data* ([arXiv version](https://arxiv.org/abs/2505.00940)), and use the code from [their GitHub](https://github.com/zywang0701/StablePCA) (MIT license). The relevant files are in `StablePCA/`.

## File structure

| Script | Description |
|---|---|
| `comparison.py` | Orchestrates all methods and produces combined comparison figures |
| `minpca_sim.py` | Runs minPCA (projected gradient descent) |
| `fairpca.py` | Runs FairPCA via SDP and multiplicative weights (Tantipongpipat et al., NeurIPS 2019) |
| `stablepca.py` | Runs StablePCA |

Usage: 
```bash
cd comparison
python comparison.py                # runs all methods and produces combined figures
```

Individual methods can also be run separately:

```bash
python minpca_sim.py  [--rerun] [--objective {MM_Var,MM_Loss}] [--p {10,50}] [--n_envs {5,50}]
python fairpca.py     [--rerun] [--objective {MM_Var,MM_Loss}] [--p {10,50}] [--n_envs {5,50}]
python stablepca.py   [--rerun] [--objective {MM_Var,MM_Loss}] [--p {10,50}] [--n_envs {5,50}]
```