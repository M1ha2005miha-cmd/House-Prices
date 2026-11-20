from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.linear_model import  ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.preprocessing import build_preprocessor
from src.neural_net import TorchMLRegressor

REGRESSORS = {
    'linear': LinearRegression,
    'ridge': Ridge,
    'lasso': Lasso,
    'elasticnet': ElasticNet,
    'xgboost': XGBRegressor,
    'lightgbm': LGBMRegressor,
    'mlp': TorchMLRegressor
}
NO_RANDOM_STATE = {'linear'}


def get_regressor(name: str, params: dict, random_state: int) -> BaseEstimator:
    params = dict(params)
    model_type = params.pop('type', name)
    if model_type not in NO_RANDOM_STATE:
        params['random_state'] = random_state
    return REGRESSORS[model_type](**params)


def build_model(name: str, params: dict, num_cols: list[str], cat_cols: list[str], random_state: int, cat_encoder: str) -> Pipeline:
    return Pipeline([
        ('preprocessor', build_preprocessor(num_cols, cat_cols, cat_encoder)),
        ('regressor', get_regressor(name, params, random_state=random_state)),
    ])
