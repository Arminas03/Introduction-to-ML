from utilities import load_data
from xgboost import XGBRegressor, plot_importance
import optuna
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, mean_squared_error, make_scorer
from optuna.trial import Trial
from optuna_dashboard import run_server

def objective(trial: Trial, x_train, y_train):
    xgb_model = XGBRegressor(
            n_estimators = trial.suggest_int('n_estimators', 200, 1000),
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1),
            max_depth = trial.suggest_int('max_depth', 3, 10),
            min_child_weight = trial.suggest_int('min_child_weight', 1, 15),
            subsample = trial.suggest_float('subsample', 0.6, 1),
            gamma = trial.suggest_float('gamma', 0, 5),
            reg_lambda = trial.suggest_float('lambda', 0, 1),
            reg_alpha = trial.suggest_float('alpha', 0, 1),
            objective = 'reg:squarederror',
    )
    i_split = int(0.9 * len(x_train))
    x_train, x_validation = x_train[:i_split], x_train[i_split:]
    y_train, y_validation = y_train[:i_split], y_train[i_split:]
    
    xgb_model.fit(x_train, y_train)
    y_prediction = xgb_model.predict(x_validation)
    return mean_squared_error(y_prediction, y_validation)


def prediction_performance(x_train, y_train, x_test, y_test, params):
    best_xgb_model = XGBRegressor(**params, objective = 'reg:squarederror')
    best_xgb_model.fit(x_train, y_train)
    plot_importance(best_xgb_model, importance_type='gain', max_num_features=10, height=0.5)
    plt.show()

    x_test, _, _, _, = load_data('stock_data_test.csv', 1)

    y_pred = best_xgb_model.predict(x_test)[:-1]
    data = pd.read_csv('stock_data_test.csv')
    data = data[data['ticker'].str.strip().str.startswith("BA")]
    y_test = data['rv_lag_1'][1:]

    print(mean_squared_error(y_test, y_pred))

    indexes = np.arange(len(y_pred))

    plt.plot(indexes, y_test, label='y_test', color='blue', marker='o')
    plt.plot(indexes, y_pred, label='y_pred', color='red', marker='x')

    plt.xlabel('Index (or Time)')
    plt.ylabel('Value')
    plt.title('Plotting Sequential Data on the Same Graph')

    plt.show()


def main():
    x_train, y_train, x_test, y_test = load_data("stock_data_train.csv", 1)
    n_study_trials = 500

    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(storage=storage, study_name='xgboost_tuning', direction='minimize')
    study.optimize(lambda trial: objective(trial, x_train, y_train), n_trials=n_study_trials)

    print(f"Best value: {study.best_value} (params: {study.best_params})")

    prediction_performance(x_train, y_train, x_test.values, y_test, study.best_params)

    

if __name__ == "__main__":
    main()