
from pandas._config import config
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, PowerTransformer
from sklearn.model_selection import train_test_split
import pandas as pd
import yaml
import sys
import os
sys.path.append(os.getcwd())

from src.loggers import get_logger

def load_config(config_path):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    return config

def clear_data(path2data):
    df = pd.read_csv(path2data)
    cat_columns = ['Make', 'Model', 'Style', 'Fuel_type', 'Transmission']
    num_columns = ['Year', 'Distance', 'Engine_capacity(cm3)', 'Price(euro)']
    
    question_dist = df[(df.Year <2021) & (df.Distance < 1100)]
    df = df.drop(question_dist.index)
    # Анализ и очистка данных
    # анализ гистограмм
    question_dist = df[(df.Distance > 1e6)]
    df = df.drop(question_dist.index)
    
    # здравый смысл
    question_engine = df[df["Engine_capacity(cm3)"] < 200]
    df = df.drop(question_engine.index)
    
    # здравый смысл
    question_engine = df[df["Engine_capacity(cm3)"] > 5000]
    df = df.drop(question_engine.index)
    
    # здравый смысл
    question_price = df[(df["Price(euro)"] < 101)]
    df = df.drop(question_price.index)
    
    # анализ гистограмм
    question_price = df[df["Price(euro)"] > 1e5]
    df = df.drop(question_price.index)
    
    #анализ гистограмм
    question_year = df[df.Year < 1971]
    df = df.drop(question_year.index)
    
    df = df.reset_index(drop=True)  
    ordinal = OrdinalEncoder()
    ordinal.fit(df[cat_columns])
    Ordinal_encoded = ordinal.transform(df[cat_columns])
    df_ordinal = pd.DataFrame(Ordinal_encoded, columns=cat_columns)
    df[cat_columns] = df_ordinal[cat_columns]
    return df

def scale_frame(frame):
    df = frame.copy()
    X,y = df.drop(columns = ['Price(euro)']), df['Price(euro)']
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scale = scaler.fit_transform(X.values)
    Y_scale = power_trans.fit_transform(y.values.reshape(-1,1))
    return X_scale, Y_scale, power_trans

def featurize(dframe, config) -> None:
    """
        Генерация новых признаков
    """
    logger = get_logger('FEATURIZE')
    logger.info('Create features')
    dframe['Distance_by_year'] = dframe['Distance']/(2022 - dframe['Year'])
    dframe['age'] = 2024 - dframe['Year']
    mean_engine_cap = dframe.groupby('Style')['Engine_capacity(cm3)'].mean()
    dframe['eng_cap_diff'] = dframe.apply(lambda row: abs(row['Engine_capacity(cm3)'] - mean_engine_cap[row['Style']]), axis=1)

    max_engine_cap = dframe.groupby('Style')['Engine_capacity(cm3)'].max()
    dframe['eng_cap_diff_max'] = dframe.apply(lambda row: abs(row['Engine_capacity(cm3)'] - max_engine_cap[row['Style']]), axis=1)

    features_path = config['featurize']['features_path']
    dframe.to_csv(features_path, index=False)


if __name__ == "__main__":
    config = load_config("./src/config.yaml")
    df_prep = clear_data(config['data_load']['dataset_csv'])
    df_new_featur = featurize(df_prep, config)
    