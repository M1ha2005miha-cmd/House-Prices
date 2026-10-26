from lightgbm import LGBMRegressor
from pandas.core.common import random_state
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.preprocessing import build_preprocessor

RANDOM_STATE = 42


def get_regressor(name: str) -> BaseEstimator:
    regressor = {
        'ridge': Ridge(),
        'xgboost': XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
        ),
        'lightgbm': LGBMRegressor(
             n_estimators=1000,
            learning_rate=0.05,
            num_leaves=15,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
    }
    return regressor[name]


def build_model(name: str, num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    return Pipeline([
        ('preprocessor', build_preprocessor(num_cols, cat_cols)),
        ('regressor', get_regressor(name)),
    ])