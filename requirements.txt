from sklearn.preprocessing import StandardScaler, PowerTransformer
import pandas as pd
import mlflow
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline
import pickle
from sklearn.ensemble import ExtraTreesRegressor

TARGET = 'price'


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train(config):
    df_train = pd.read_csv(config['data_split']['trainset_path'])
    df_test = pd.read_csv(config['data_split']['testset_path'])
    print(df_train.shape)

    X_train, y_train = df_train.drop(columns=[TARGET]).values, df_train[TARGET].values
    X_val, y_val = df_test.drop(columns=[TARGET]).values, df_test[TARGET].values

    power_trans = PowerTransformer()
    y_train = power_trans.fit_transform(y_train.reshape(-1, 1))
    y_val = power_trans.transform(y_val.reshape(-1, 1))

    mlflow.set_experiment("house prices")
    with mlflow.start_run():
        if config['train']['model_type'] == "tree":
            lr_pipe = Pipeline(steps=[('scaler', StandardScaler()),
                                      ('model', ExtraTreesRegressor(random_state=42))])
            params = {'model__n_estimators': config['train']['n_estimators']}
        else:
            lr_pipe = Pipeline(steps=[('scaler', StandardScaler()),
                                      ('model', SGDRegressor(random_state=42))])
            params = {'model__alpha': config['train']['alpha'],
                      "model__fit_intercept": [False, True]}

        clf = GridSearchCV(lr_pipe, params, cv=config['train']['cv'], n_jobs=4)
        clf.fit(X_train, y_train.reshape(-1))
        best = clf.best_estimator_
        y_pred = best.predict(X_val)
        y_price_pred = power_trans.inverse_transform(y_pred.reshape(-1, 1))
        print(y_price_pred[:5])
        print(power_trans.inverse_transform(y_val.reshape(-1, 1))[:5])
        (rmse, mae, r2) = eval_metrics(
            power_trans.inverse_transform(y_val.reshape(-1, 1)), y_price_pred)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        print("R2=", r2)

        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(best, "model", signature=signature)

        with open(config['train']['model_path'], "wb") as file:
            pickle.dump(best, file)

        with open(config['train']['power_path'], "wb") as file:
            pickle.dump(power_trans, file)
