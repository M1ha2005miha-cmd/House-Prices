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

QUALITY_COLS = [
    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
    'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC',
]

QUALITY_MAP = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}


def fill_absent_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[NONE_FILL_COLS] = df[NONE_FILL_COLS].fillna('None')
    df[ZERO_FILL_COLS] = df[ZERO_FILL_COLS].fillna(0)
    return df

def encode_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in QUALITY_COLS:
        df[col] = df[col].map(QUALITY_MAP)
    return df


def convert_mssubclass_to_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['MSSubClass'] = df['MSSubClass'].astype(str)
    return df