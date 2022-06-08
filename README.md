# lestpy

## Overview
### Why
Lestpy is aimed to model regression problems with a bunch of logical interactions. This approach is an alternative way to describe the relationship between the features and the targets.


## Getting Started

### Installation

You can install LestPy from [PyPI]():

```sh
    python -m pip install lestpy
```

### Last release
Lespy 0.0.11
```sh
    python -m pip install lestpy --upgrade
```

### Main classes, methods and attributes
* Interaction
Methods:
    1. compute
    2. get_interaction_list
    3. get_interaction_dict
    3. add_interaction_dict
    4. remove_interactions
* InteractionBuilder
* Transformer
Methods:
    1. fit
    2. transform
    3. fit_transform
    4. inverse_transform
* LBM_Regression
Methods:
    1. transform
    2. fit
    3. fit_transform
    4. predict
    5. optimize
    6. features_analysis
    7. print_model
    7. fitting_score
    8. extract_features
* Display
Methods:
    1. ternary_diagram
    2. response_surface
    3. pareto_frontier
    4. sensibility_analysis
    5. display_interaction
    6. residues
    7. fit
    8. metrics_curve
    9. describe
    10. corr_graph (in development)
* Outliers_Inspection
Methods:
    1. cooks_distance
    2. mahalanobis_distance
    3. z_score

### How to

## Use
Lestpy is designed to be used similarly to sklearn modelization classes and their methods (fit(), transform(), predict(), ...)

## Contributing
There are many ways to support the development of lestpy:

* **File an issue** on Github, if you encounter problems, have a proposal, etc.
* **Send an email with ideas** to the author.
* **Submit a pull request** on Github if you improved the code and know how to use git.


## Links
The source code and issue tracker of this package is to be found on **Github**: [pyonysos/lestpy].


[pyonysos/lestpy]: https://github.com/pyonysos/lestpy