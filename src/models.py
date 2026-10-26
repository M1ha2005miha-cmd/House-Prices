from lightgbm import LGBMRegressor
from pandas.core.common import random_state
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.preprocessing import build_preprocessor

REGRESSORS = {
    'ridge': Ridge,
    'xgboost': XGBRegressor,
    'lightgbm': LGBMRegressor,
}


def get_regressor(name: str, params: dict, random_state: int) -> BaseEstimator:
    return REGRESSORS[name](**params, random_state=random_state)


def build_model(name: str, params: dict, num_cols: list[str], cat_cols: list[str], random_state: int) -> Pipeline:
    return Pipeline([
        ('preprocessor', build_preprocessor(num_cols, cat_cols)),
        ('regressor', get_regressor(name, params, random_state=random_state)),
    ])