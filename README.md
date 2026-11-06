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
- **Пропуски по смыслу признака.** Для большинства признаков `NA` означает отсутствие объекта (нет гаража, подвала), такие пропуски заполняются `"None"` или `0`, а не медианой.
- **Удалены 2 выброса** - недостроенные дома (`SaleConfition = Partial`) с огромной площадью и низкой ценой.
- **Вся предобработка с обучением внутри `Pipeline`**, чтобы на каждом фолде медианы, категории и масштаб считались только по обучающей части - без утечки данных.
-**Ансамбль выбран по корреляции ошибок  out-of-fold предсказаний.** XGBoost и LightGBM ошибаются похоже (корреляция 0.92), поэтому их пара хуже одного XGBoost. Модели разных семейств дополняют друг друга.
**Разрыв CV (0.111) и Kaggle (0.130)** во многом объясняется домом `Id 2550` в `test`: площадь 5095 кв. футов, `ScaleCondition = Patrial`. Это тот же тип выброса, что был удален из `train`, поэтому модель оценивает его как обычный особняк (~1.28 млн $). В ручную предсказание не исправлялось, чтобы не подгонять результат под тестовую выборку.

## Известная проблема на macOS

PyTorch, scikit-learn и Homebrew-версия `libomp` (для XGBoost/LightGBM) приносят разные копии OpenMP. В одном процессе это может приводит к зависанию или падению Python. Решение - заменить копии PyTorch и scikit-learn ссылками на Homebrew-версию:

```bash
cd .venv/lib/python3.*/site-pachages
mv torch/lib/libomp.dylib torch/lib/libomp.dylib.bak
ln -s /opt/homebrew/opt/libomp/lib/libomp.dylib torch/lib/libomp.dylib
mv sklearn/.dylibs/libomp.dylib sklearn/.dylibs/libomp.dylib.bak
ln -s /opt/homebrew/opt/libomp/lib/libomp.dylib sklearn/.dylibs/libomp.dylib
```

