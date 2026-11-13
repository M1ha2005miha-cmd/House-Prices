import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold

def evaluate_model(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    stratified: bool = False
) -> tuple[float, float]:
    if stratified:
        price_bins = pd.qcut(y, q=10, labels=False)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv = list(splitter.split(X, price_bins))
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
    return scores.mean(), scores.std()
    
