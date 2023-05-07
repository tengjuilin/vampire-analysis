# `vampire-analysis`

![GitHub](https://img.shields.io/github/license/tengjuilin/vampire-analysis)
[![Documentation Status](https://readthedocs.org/projects/vampire/badge/?version=latest)](https://vampire.readthedocs.io/en/latest/?badge=latest)
![PyPI](https://img.shields.io/pypi/v/vampire-analysis)

VAMPIRE (Visually Aided Morpho-Phenotyping Image Recognition) analysis quantifies and visualizes heterogeneity of cell and nucleus morphology [1]. It is used widely in analyzing microglial shape change in response to oxygen-glucose deprivation [2] and morphological changes in cancer metastasis [3].

[`vampire-analysis`](https://pypi.org/project/vampire-analysis/) provides a reproducible, fully-documented, and easy-to-use Python package that is based on the method and software used the in [`vampireanalysis`](https://pypi.org/project/vampireanalysis/) GUI [1]. Main advantages include:

- operating-system-independent package API
- [full documentation](https://vampire.readthedocs.io/en/latest/) with easy-to-read code
- flexible input and filtering options
- flexible plotting options

## Installation

See documentation for [detailed installation guide](https://vampire.readthedocs.io/en/latest/user/installation.html). If Python is installed on your machine, type the following line into your command prompt to install via [PyPI](https://pypi.org/project/vampire-analysis/):

```bash
pip install vampire-analysis
```

## Getting started

See documentation for detailed guide for basics of [building](https://vampire.readthedocs.io/en/latest/user/build_basics.html) and [applying](https://vampire.readthedocs.io/en/latest/user/apply_basics.html) models. If you have `build.xlsx` under `C:\vampire` containing the build image set information, you can build the model with

```python
>>> import pandas as pd  # used to read excel files
>>> import vampire as vp  # recommended import signature

>>> build_df = pd.read_excel(r'C:\vampire\build.xlsx')
>>> vp.model.build_models(build_df, random_state=1)
```

If you have `apply.xlsx` under `C:\vampire` containing the apply image set information, you can apply the model with

```python
>> > apply_df = pd.read_excel(r'C:\vampire\apply.xlsx')
>> > vp.model.transform_datasets(apply_df)
```

Flexible options are provided for [building](https://vampire.readthedocs.io/en/latest/user/build_advanced.html) and [applying](https://vampire.readthedocs.io/en/latest/user/apply_advanced.html) models in the advanced section in the documentation.

## References

[1] Phillip, J.M., Han, KS., Chen, WC. et al. A robust unsupervised machine-learning method to quantify the morphological heterogeneity of cells and nuclei. *Nat Protoc* **16**, 754–774 (2021). <https://doi.org/10.1038/s41596-020-00432-x>

[2]  Joseph, A, Liao, R, Zhang, M, et al. Nanoparticle-microglial interaction in the ischemic brain is modulated by injury duration and treatment. *Bioeng Transl Med.* 2020; 5:e10175. <https://doi.org/10.1002/btm2.10175>

[3] Wu, PH., Phillip, J., Khatau, S. et al. Evolution of cellular morpho-phenotypes in cancer metastasis. *Sci Rep* **5**, 18437 (2016). <https://doi.org/10.1038/srep18437>
