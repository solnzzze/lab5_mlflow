from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, PowerTransformer
from sklearn.model_selection import train_test_split
import pandas as pd
import yaml
import sys
import os
sys.path.append(os.getcwd())

from src.loggers import get_logger

TARGET = 'price'


def load_config(config_path):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    return config


def clear_data(path2data):
    """
        Очистка табличного набора данных (этап 1).
    """
    logger = get_logger('CLEAR_DATA')
    logger.info('Clean dataset')

    df = pd.read_csv(path2data)

    # Заполняем пропуски на случай неполных данных:
    # числовые признаки — медианой
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # категориальные — самым частым значением
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Удаляем выбросы по целевой переменной (анализ распределения price)
    df = df[df[TARGET] < 13000000]
    df = df[df[TARGET] > 1750000]

    df = df.reset_index(drop=True)

    # Кодируем категориальные колонки числами
    # (mainroad, guestroom, basement, hotwaterheating,
    #  airconditioning, prefarea, furnishingstatus)
    ordinal = OrdinalEncoder()
    df[cat_cols] = ordinal.fit_transform(df[cat_cols])

    return df


def scale_frame(frame):
    df = frame.copy()
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scale = scaler.fit_transform(X.values)
    Y_scale = power_trans.fit_transform(y.values.reshape(-1, 1))
    return X_scale, Y_scale, power_trans


def featurize(dframe, config) -> None:
    """
        Генерация новых признаков (этап 2).
    """
    logger = get_logger('FEATURIZE')
    logger.info('Create features')

    # Площадь на одну спальню
    dframe['AreaPerBedroom'] = dframe['area'] / dframe['bedrooms'].replace(0, 1)

    # Общее количество комнат (спальни + санузлы)
    dframe['TotalRooms'] = dframe['bedrooms'] + dframe['bathrooms']

    # Площадь на этаж
    dframe['AreaPerStory'] = dframe['area'] / dframe['stories'].replace(0, 1)

    # Признак наличия парковки
    dframe['HasParking'] = (dframe['parking'] > 0).astype(int)

    features_path = config['featurize']['features_path']
    dframe.to_csv(features_path, index=False)
    logger.info(f'Features saved to {features_path}')


if __name__ == "__main__":
    config = load_config("./src/config.yaml")
    df_prep = clear_data(config['data_load']['dataset_csv'])
    featurize(df_prep, config)
