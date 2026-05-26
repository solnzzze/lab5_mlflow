import sys
import os
sys.path.append(os.getcwd())

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dvclive import Live
from src.loggers import get_logger
from prepare_dataset import load_config

TARGET = 'price'


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def test_model(config):
    """
        Оценка качества работы модели на тестовой выборке (этап 4).
    """
    logger = get_logger('TEST_MODEL')
    logger.info('Evaluate model')

    # Загружаем обученную модель и преобразователь целевой переменной
    with open(config['test']['model_path'], "rb") as f:
        model = pickle.load(f)
    with open(config['test']['power_path'], "rb") as f:
        power_trans = pickle.load(f)

    # Тестовая выборка
    df_test = pd.read_csv(config['test']['testset_path'])
    X_test = df_test.drop(columns=[TARGET]).values
    y_test = df_test[TARGET].values

    # Предсказание и обратное преобразование к реальным ценам
    y_pred_scaled = model.predict(X_test)
    y_pred = power_trans.inverse_transform(y_pred_scaled.reshape(-1, 1))

    rmse, mae, r2 = eval_metrics(y_test, y_pred)
    logger.info(f'RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.4f}')

    # Логируем метрики через dvclive (для dvc metrics / plots)
    with Live() as live:
        live.log_metric("rmse", rmse)
        live.log_metric("mae", mae)
        live.log_metric("r2", r2)


if __name__ == "__main__":
    config = load_config("./src/config.yaml")
    test_model(config)
