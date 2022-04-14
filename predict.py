
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

AGG_COLS = ["material_code", "company_code", "country", "region", "manager_code"]

FOLDS = 5

class FeatureEngineering:
  """
  Создает новые признаки:
  1. Lag признаки;
  2. Скользящие средние;
  """

  def __init__(self, df):
    self.df = df

  # генерация случайного шума случайный шум
  def random_noise(self):
      return np.abs(np.random.normal(scale=0.1, size=(len(self.df),))) 

  # lag признаки
  def lag_features(self):
    lags = [3, 6, 9, 12, 18] 
    for lag in lags:
      self.df['sales_lag_' + str(lag)] = self.df.groupby(AGG_COLS)['volume'].transform(
          lambda x: x.shift(lag)) + self.random_noise() # Adding random noise to each value.
      
    return self.df

  # скользящие аггрегированные признаки
  def roll_agg_features(self):
    windows = [3, 6, 9, 12, 18] 
    for window in windows:
        self.df['sales_roll_mean_' + str(window)] = self.df.groupby(AGG_COLS)['volume']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window,
                                         min_periods=3,
                                         win_type="triang").mean()) + self.random_noise()
        
        self.df['sales_roll_max_' + str(window)] = self.df.groupby(AGG_COLS)['volume']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window,
                                         min_periods=3).max()) + self.random_noise()
        
        self.df['sales_roll_min_' + str(window)] = self.df.groupby(AGG_COLS)['volume']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window,
                                         min_periods=3).min()) + self.random_noise()

    return self.df


  # применение обоих методов
  def transform(self):
    self.df = self.lag_features()
    self.df = self.roll_agg_features()
    
    return self.df



def predict(df: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:
    """
    Вычисление предсказаний.

    Параметры:
        df:
          датафрейм, содержащий все сделки с начала тренировочного периода до `month`; типы
          колонок совпадают с типами в ноутбуке `[SC2021] Baseline`,
        month:
          месяц, для которого вычисляются предсказания.

    Результат:
        Датафрейм предсказаний для каждой группы, содержащий колонки:
            - `material_code`, `company_code`, `country`, `region`, `manager_code`,
            - `prediction`.
        Предсказанные значения находятся в колонке `prediction`.
    """

    group_ts = df.groupby(AGG_COLS + ["month"])["volume"].sum().unstack(fill_value=0)
    group_ts[month] = 0
    new_df = pd.DataFrame(group_ts.stack()).reset_index()
    new_df = new_df.rename(columns={0: 'volume'})
    new_df['volume'] = np.log1p(new_df['volume'])
    fe_data = FeatureEngineering(new_df).transform()
    fe_data['month'] = fe_data['month'].astype(str)

    predicting_data = fe_data[fe_data['month'] == str(month)[:10]].reset_index()

    model = CatBoostRegressor()
    predictions = pd.DataFrame()
    for i in range(FOLDS):
      model_path = f'cv_model_{i}.cbm'
      model.load_model(model_path)
      prediction = model.predict(predicting_data[model.feature_names_])
      predictions[f'prediction_{i}'] = prediction

    preds_df = predicting_data[AGG_COLS].copy()
    preds_df["prediction"] = np.expm1(np.mean(predictions, axis=1))

    return preds_df
