# Customer Churn Prediction (Telecom)

## Описание проекта

Цель проекта — предсказать отток клиентов телеком-сервиса на горизонте одного месяца для последующих сценариев удержания и таргетированных коммуникаций.

## Данные

- 7043 клиентов
- 20 признаков
- целевая переменная: Churn
- доля churn: ~26.5% (класс несбалансирован)

Источник датасета: публичный Telco Customer Churn (Kaggle).

## Этапы работы

1. Постановка задачи и формулирование бизнес-проблемы.
2. EDA:
   - анализ распределений признаков;
   - определение ключевых факторов оттока;
   - выделение поведенческих сегментов.
3. Построение baseline-модели (Logistic Regression).
4. Обучение бустинга (CatBoostClassifier).
5. Подбор порога классификации под F1/Recall.
6. Интерпретация результатов и итоговые выводы.

## Основные результаты

### EDA-инсайты
Выявлено, что churn чаще встречается у клиентов:
- с низким tenure (новички),
- на месячном контракте (`Month-to-month`),
- с более высокой стоимостью ежемесячного платежа,
- без услуг OnlineSecurity и TechSupport,
- использующих Electronic check,
- использующих Fiber optic.

### Моделирование

| Модель | F1 | ROC-AUC | Recall (churn) |
|---|---|---|---|
| Logistic Regression | 0.61 | 0.842 | 0.56 |
| CatBoost (opt thr=0.35) | 0.64 | 0.846 | 0.73 |

Оптимизация порога позволила снизить FN и повысить recall churn-клиентов, что важно для бизнес-сценариев удержания.

### Интерпретируемость

Топ-факторы оттока (feature importance CatBoost):
- Contract
- InternetService
- tenure
- TotalCharges
- MonthlyCharges
- PaymentMethod
- OnlineSecurity

## Используемый стек

- Python
- pandas / numpy
- sklearn
- catboost
- matplotlib / seaborn

## Структура проекта
data/
   telco_churn.csv
notebooks/
   cat_boost_info/
   01_eda_baseline_catboost.ipynb
requirements.txt
README.md

## Вывод

Модель может использоваться для выделения клиентов с высоким риском churn и применения персонализированных retention-кампаний.