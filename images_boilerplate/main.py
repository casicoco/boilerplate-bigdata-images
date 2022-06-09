'''
Top level orchestrator of the project. To be called from the CLI.
It comprises all the "routes" you may want to call
'''
import numpy as np
from ts_boilerplate.data import get_X_y, get_folds, train_test_split
from ts_boilerplate.model import get_model, fit_model, predict_output
from ts_boilerplate.metrics import mape
from ts_boilerplate.params import CROSS_VAL, TRAIN, DATA
from typing import Tuple, List


def train(data: np.ndarray, print_metrics: bool = False, save_metrics: bool = False):
    """
    Train the model in this package on one fold `data` containing the 2D-array of time-series for your problem
    Returns `metrics_test` associated with the training
    """
    # $CHALLENGIFY_BEGIN
    data_train, data_test = train_test_split(data, **TRAIN)
    X_train, y_train = get_X_y(data_train, **TRAIN)
    X_test, y_test = get_X_y(data_test, **TRAIN)
    model = get_model(X_train, y_train)
    history = fit_model(model, X_train, y_train)
    y_pred = predict_output(model, X_test)
    metrics_test = mape(y_test, y_pred)
    if print_metrics:
        print("### Test Metric: ", metrics_test)
    return metrics_test
    # $CHALLENGIFY_END


def cross_validate(data: np.ndarray, print_metrics: bool = False, save_metrics: bool = False):
    """
    Cross-Validate the model in this package on`data`
    Returns `metrics_cv`: the list of test metrics at each fold
    """
    # $CHALLENGIFY_BEGIN
    folds = get_folds(data, **CROSS_VAL)
    metrics_cv = []
    for fold in folds:
        metrics_fold = train(fold, print_metrics=print_metrics)
        metrics_cv.append(metrics_fold)

    if print_metrics:
        print(f"### CV metrics after {len(folds)} folds ### ")
        print(metrics_cv)
    return metrics_cv
    # $CHALLENGIFY_END
