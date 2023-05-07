# Overview

![GitHub](https://img.shields.io/github/license/tengjuilin/vampire-analysis)
[![Documentation Status](https://readthedocs.org/projects/vampire/badge/?version=latest)](https://vampire.readthedocs.io/en/latest/?badge=latest)
![PyPI](https://img.shields.io/pypi/v/vampire-analysis)

VAMPIRE (Visually Aided Morpho-Phenotyping Image Recognition) quantifies and visualizes shape modes of cell and nucleus morphology [1]. VAMPIRE has been used to analyze morphological changes of

1. *in vitro* cancer cells in cancer metastasis [2],
2. *ex vivo* rat microglia in response to oxygen-glucose deprivation [3],
3. *ex vivo* ferret microglia in response to oxygen-glucose deprivation [4],
4. *ex vivo* rat microglia in response to brain-derived extravellular vescicle treatment [5],
5. *in vivo* MGluR5 rat model's microglia at different ages and sexes [6].

[vampire-analysis](https://pypi.org/project/vampire-analysis/) provides a reproducible, fully-documented, and easy-to-use Python package that is based on the method and software used the in [vampireanalysis GUI](https://pypi.org/project/vampireanalysis/) [(GitHub source)](https://github.com/kukionfr/VAMPIRE_open) [1]. Main advantages include:

- Operating-system-independent package API
- Full documentation with easy-to-read code
- Flexible input and filtering options
- Flexible plotting options

## Installation

See documentation for [detailed installation guide](https://vampire.readthedocs.io/en/latest/user/installation.html). If Python is installed on your machine, type the following line into your command prompt to install via [PyPI](https://pypi.org/project/vampire-analysis/):

```bash
pip install vampire-analysis
```

## Getting started

See documentation for detailed guide for basics of [fitting](https://vampire.readthedocs.io/en/latest/user/build_basics.html) a VAMPIRE model and [transforming](https://vampire.readthedocs.io/en/latest/user/apply_basics.html) a dataset using a VAMPIRE model. If you have ``build.xlsx`` under ``C:\vampire`` containing the build image set information, you can build the model with

```python
import pandas as pd  # used to read excel files
import vampire as vp  # recommended import signature
```

```python
build_df = pd.read_excel(r'C:\vampire\build.xlsx')
vp.quickstart.fit_models(build_df, random_state=1)
```

If you have ``apply.xlsx`` under ``C:\vampire`` containing the apply
image set information, you can apply the model with

```python
apply_df = pd.read_excel(r'C:\vampire\apply.xlsx')
vp.quickstart.transform_datasets(apply_df)
```

Flexible options are provided for [building](https://vampire.readthedocs.io/en/latest/user/build_advanced.html) and [applying](https://vampire.readthedocs.io/en/latest/user/apply_advanced.html) models in the advanced section in the documentation.

## References

1. Phillip, J. M.; Han, K.-S.; Chen, W.-C.; Wirtz, D.; Wu, P.-H. A Robust Unsupervised Machine-Learning Method to Quantify the Morphological Heterogeneity of Cells and Nuclei. *Nat Protoc* **2021**, *16* (2), 754–774. https://doi.org/10.1038/s41596-020-00432-x.
2. Wu, P.-H.; Gilkes, D. M.; Phillip, J. M.; Narkar, A.; Cheng, T. W.-T.; Marchand, J.; Lee, M.-H.; Li, R.; Wirtz, D. Single-Cell Morphology Encodes Metastatic Potential. *Science Advances* **2020**, *6* (4), eaaw6938.
3. Joseph, A.; Liao, R.; Zhang, M.; Helmbrecht, H.; McKenna, M.; Filteau, J. R.; Nance, E. Nanoparticle-Microglial Interaction in the Ischemic Brain Is Modulated by Injury Duration and Treatment. *Bioengineering & Translational Medicine* **2020**, *5* (3), e10175. https://doi.org/10.1002/btm2.10175.
4. Wood, T. R.; Hildahl, K.; Helmbrecht, H.; Corry, K. A.; Moralejo, D. H.; Kolnik, S. E.; Prater, K. E.; Juul, S. E.; Nance, E. A Ferret Brain Slice Model of Oxygen–Glucose Deprivation Captures Regional Responses to Perinatal Injury and Treatment Associated with Specific Microglial Phenotypes. *Bioengineering & Translational Medicine* **2022**, *7* (2), e10265. https://doi.org/10.1002/btm2.10265.
5. Nguyen, N. P.; Helmbrecht, H.; Ye, Z.; Adebayo, T.; Hashi, N.; Doan, M.-A.; Nance, E. Brain Tissue-Derived Extracellular Vesicle Mediated Therapy in the Neonatal Ischemic Brain. *International Journal of Molecular Sciences* **2022**, *23* (2), 620. https://doi.org/10.3390/ijms23020620.
6. Dahl, V.; Helmbrecht, H.; Rios Sigler, A.; Hildahl, K.; Sullivan, H.; Janakiraman, S.; Jasti, S.; Nance, E. Characterization of a MGluR5 Knockout Rat Model with Hallmarks of Fragile X Syndrome. *Life* **2022**, *12* (9), 1308. https://doi.org/10.3390/life12091308.
