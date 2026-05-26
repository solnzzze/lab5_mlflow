import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
from sklearn.model_selection import train_test_split

from src.loggers import get_logger
from prepare_dataset import load_config


def data_split(config):
    """
        Разбиение данных на обучающую и тестовую выборки.
    """
    logger = get_logger('DATA_SPLIT')
    logger.info('Split dataset into train/test')

    df = pd.read_csv(config['featurize']['features_path'])

    train_df, test_df = train_test_split(
        df,
        test_size=config['data_split']['test_size'],
        random_state=42,
    )

    train_df.to_csv(config['data_split']['trainset_path'], index=False)
    test_df.to_csv(config['data_split']['testset_path'], index=False)

    logger.info(f'Train shape: {train_df.shape}, Test shape: {test_df.shape}')


if __name__ == "__main__":
    config = load_config("./src/config.yaml")
    data_split(config)
