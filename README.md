This is a **boilerplate** repo for a machine-learning project involving **Big Dataset Images**.

In particular

- It provides a structure template for loading big dataset in **tensorflow** to ensure model can be trained with all data.
- It is agnostic of the type of tensorflow model involved
- It is well suited for short research projects, typical of few-weeks coding bootcamps such as Le Wagon DataScience

# Detailed package workflow

## Architecture
- `ts_boilerplate` package
  - `main.py` comprises the main routes to be called from the CLI (`train` and `cross-validate`)
  - `params.py` contains project-level global variable to be set manually
<br>

- `data` folder contains
  - `raw` and `clean` folder should contain 2D arrays `data` with (axis 0) representing timesteps, and (axis 1) columns containing tagets and covariates, as per [picture](https://github.com/lewagon/data-images/blob/master/DL/time-series-covariates.png?raw=true)
    ```python
    data.shape = (length, n_targets+n_covariates)
    ```
  - `Xy` may persist your tuple (X,y) of 3D arrays to be fed to your models
    ```python
    X.shape = (n_samples, input_length, n_covariates)
    y.shape = (n_samples, output_length, n_targets)
    ```
- `notebooks`
  - `tutorial_ts_forecasting.ipynb` is a recommended read before diving into this project. It contains visuals that will help you fill global project params and understand naming conventions
  - `create_dummy_tests.ipynb` will help you understand how tests have been built

<br>

- `tests` folder detailed below

## How to test your code?
Run this in your terminal from the root project folder to check your code
- `pytest`
- `pytest -m "not optional"`  to only check mandatory tests
- `pytest -m "not optional" -m "not slow"` to also avoid tests that may be slow (involving fitting your model)

These tests require `ts_boilerplate/params.py` to be filled corresponding to your true project speficities


# TODO
# V1
- [x] Basic training & cross-val routes, with test
- [x] Make model fit well for univariate, multivariate (n_tagets >1) & sequences (output_sequence_lenght>1)
- [x] Make tests pass for stride > 1
- [x] Add tests about the model (shape of prediction, etc)

# V2
- [ ] Add backtesting as main route. Very important concept to teach to students
- [ ] Refacto `model.py`
  - [ ] Rename `pipeline.py` because it may comprises the pre-processing such as scaling etc...
  - [ ] Turn into a class `TsPipeline()` instead of pure functions

# V3
- [ ] Add tests for future-covariates
- [ ] create Makefile
  - [ ] Include DAG of the project
- [ ] publish to lewagon community
