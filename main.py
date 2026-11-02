import numpy as np
import pandas as pd 
import yaml

from src.preprocessing import apply_static_transform, remove_outliers
from src.models import build_model
from src.evaluation import evaluate_model


def load_config(path: str) -> dict:
    with open(path) as file: 
        return yaml.safe_load(file)


def load_data(config: dict) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    train = pd.read_csv(config['data']['train_path'])
    test = pd.read_csv(config['data']['test_path'])

    train = apply_static_transform(remove_outliers(train))
    test = apply_static_transform(test)

    features = train.drop(columns=['Id', 'SalePrice'])
    target = np.log1p(train['SalePrice'])
    return features, target, test


def split_feature_types(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    num_cols = features.select_dtypes(include='number').columns.tolist()
    cat_cols = features.select_dtypes(exclude='number').columns.tolist()
    return num_cols, cat_cols


def run_cross_validation(
    config: dict,
    features: pd.DataFrame,
    target: pd.Series,
    num_cols: list[str],
    cat_cols: list[str],
) -> pd.DataFrame:
    rows = []
    for name, params in config['models'].items():
        model = build_model(name, params, num_cols, cat_cols, config['cv']['random_state'])
        rmse_mean, rmse_std = evaluate_model(model, features, target, **config['cv'])
        rows.append({'model': name, 'rmse_mean': rmse_mean, 'rmse_std': rmse_std})
    return pd.DataFrame(rows).sort_values('rmse_mean')


def make_submission(
    config: dict,
    features: pd.DataFrame,
    target: pd.Series,
    test: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
) -> pd.DataFrame:
    predictions = []
    for name in config['ensemble']['models']:
        params = config['models'][name]
        model = build_model(name, params, num_cols, cat_cols, config['cv']['random_state'])
        model.fit(features, target)
        predictions.append(model.predict(test))

    log_price = np.mean(predictions, axis=0)
    return pd.DataFrame({'Id': test['Id'], 'SalePrice': np.expm1(log_price)})


def main():
    config = load_config('config.yaml')
    features, target, test = load_data(config)
    num_cols, cat_cols = split_feature_types(features)
    
    cv_results = run_cross_validation(config, features, target, num_cols, cat_cols)
    print(cv_results.to_string(index=False, float_format='%.4f'))

    submission = make_submission(config, features, target, test, num_cols, cat_cols)
    submission.to_csv(config['data']['submission_path'], index=False)
    print(f'Submission saved: {config['data']['submission_path']}')
    
    
    

if __name__ == '__main__':
    main()