# House Prices - Advanced Regression Techniques

Предсказание цены дома по 79 признакам ([соревнование Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)).
Метрика - RMSE между логарифмами предсказанной и реальной цены

## Результаты 

5-fold кросс-валидации на `train` (1458 домов после удаления выбросов):

| Модель | RMSE |
|---|---|
| **Ансамбль: Ridge + XGBoost + MLP** | **0.1107** |
| XGBoost | 0.1150 ± 0.0073 |
| MLP(PyTorch) | 0.1183 ± 0.0073 |
| Ridge (бейзлайн) | 0.1198 ± 0.0084 |
| LightGBM | 0.1236 ± 0.0084 |

Kaggle Public Score: **0.12977**

# Запуск

1. Скачайте данные со страницы соревнования и положите `train.csv` и `test.csv` в папку `data/`.
2. Установите окуржение и запустите:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Скрипт выводит таблицу кросс-валидации и сохраняет `submission.csv`.
Модели, гиперпараметры и состав ансамбля задаются в `config.yaml`.

## Структура

```
├── config.yaml            # пути, параметры CV, модели, ансамбль
├── main.py                # точка входа
├── notebooks/eda.ipynb    # разведочный анализ с выводами
└── src/
    ├── preprocessing.py   # заполнение пропусков, кодирование, препроцессор
    ├── models.py          # реестр моделей
    ├── neural_net.py      # MLP на PyTorch в интерфейсе sklearn
    └── evaluation.py      # кросс-валидация
```
## Ключевые решения

- **Целевая переменная - `log1p(SalePrice)`. **Убирает скошенность распределения(1.88 -> 0.12) и совпадает с метрикой соревнования.
