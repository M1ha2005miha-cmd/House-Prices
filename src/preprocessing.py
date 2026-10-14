import pandas as pd

NONE_FILL_COLS = [
    'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'MasVnrType',
]

ZERO_FILL_COLS = [
    'MasVnrArea',
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
    'GarageCars', 'GarageArea', 'BsmtFullBath', 'BsmtHalfBath',
]


def fill_absent_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[NONE_FILL_COLS] = df[NONE_FILL_COLS].fillna('None')
    df[ZERO_FILL_COLS] = df[ZERO_FILL_COLS].fillna(0)
    return df