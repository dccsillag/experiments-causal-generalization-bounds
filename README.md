# Experiments for 'Generalization Bounds for Causal Regression: Insights, Guarantees and Sensitivity Analysis'

Published at ICML 2024  
[arXiv](https://arxiv.org/abs/2405.09516) ⋅ [Paper PDF](http://dccsillag.xyz/projects/causal-generalization-bounds/icml2024/paper.pdf) ⋅ [Poster PDF](http://dccsillag.xyz/projects/causal-generalization-bounds/icml2024/poster.pdf) ⋅ [ICML page](https://icml.cc/virtual/2024/poster/33968)

Abstract:
> Many algorithms have been recently proposed for causal machine learning.
> Yet, there is little to no theory on their quality, especially considering finite samples.
> In this work, we propose a theory based on generalization bounds that provides such guarantees.
> By introducing a novel change-of-measure inequality, we are able to tightly bound the model loss in terms of the deviation of the treatment propensities over the population, which we show can be empirically limited.
> Our theory is fully rigorous and holds even in the face of hidden confounding and violations of positivity.
> We demonstrate our bounds on semi-synthetic and real data, showcasing their remarkable tightness and practical utility.

## Code structure

This repository contains the code to reproduce all experiments and figures in the paper:

- The code for Figure 1 is in `figure1.py`;
- The code for Figure 2 is in `figure2.py`; and
- The code for Figure 3 is in `figure3.py`.

The other files are: `data.py`, which contains data loading utilities, and `metalearners.py`, which implements T-, S- and X-learners.

The figures from the main paper are reproduced by using the default arguments to the scripts. Figures in the appendices can be reproduced by using the appropriate arguments.

Dependencies are managed via [Poetry](https://python-poetry.org/); for information on using Poetry, see its [Basic usage](https://python-poetry.org/docs/basic-usage/) page.

## Citing

```bibtex
@InProceedings{csillag24a,
  title = {Generalization Bounds for Causal Regression: Insights, Guarantees and Sensitivity Analysis},
  author = {Csillag, Daniel and Struchiner, Claudio Jose and Goedert, Guilherme Tegoni},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  pages = {9576--9602},
  year = {2024},
  editor = {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = {235},
  series = {Proceedings of Machine Learning Research},
  month = {21--27 Jul},
  publisher = {PMLR},
  pdf = {https://raw.githubusercontent.com/mlresearch/v235/main/assets/csillag24a/csillag24a.pdf},
  url = {https://proceedings.mlr.press/v235/csillag24a.html},
  abstract = {Many algorithms have been recently proposed for causal machine learning. Yet, there is little to no theory on their quality, especially considering finite samples. In this work, we propose a theory based on generalization bounds that provides such guarantees. By introducing a novel change-of-measure inequality, we are able to tightly bound the model loss in terms of the deviation of the treatment propensities over the population, which we show can be empirically limited. Our theory is fully rigorous and holds even in the face of hidden confounding and violations of positivity. We demonstrate our bounds on semi-synthetic and real data, showcasing their remarkable tightness and practical utility.}
}
```
