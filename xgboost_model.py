from utilities import load_data, test_model
from xgboost import XGBRegressor, plot_importance
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from optuna.trial import Trial


def _train_val_split(split_percentage, x_train, y_train):
    i_split = int(split_percentage * len(x_train))

    return (
        x_train[:i_split], y_train[:i_split],
        x_train[i_split:], y_train[i_split:]
    )


def _xgb_model_setup(trial: Trial) -> XGBRegressor:
    return XGBRegressor(
        n_estimators = trial.suggest_int('n_estimators', 200, 1000),
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1),
        max_depth = trial.suggest_int('max_depth', 3, 10),
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10),
        subsample = trial.suggest_float('subsample', 0.6, 1),
        gamma = trial.suggest_float('gamma', 0, 5),
        reg_lambda = trial.suggest_float('lambda', 0.2, 1),
        reg_alpha = trial.suggest_float('alpha', 0.2, 1),
        objective = 'reg:squarederror'
    )


def _objective(trial: Trial, x_train, y_train):
    x_train, y_train, x_validation, y_validation = _train_val_split(0.8, x_train, y_train)
    
    model = _xgb_model_setup(trial)
    model.fit(x_train, y_train)

    return mean_squared_error(model.predict(x_validation), y_validation)


def _model_feature_selection(model):
    plot_importance(model, importance_type='gain', max_num_features=10, height=0.5)
    plt.show()


def get_predictions():
    x_train, y_train, x_test, y_test = load_data("stock_data_train.csv", 1)
    n_study_trials = 100

    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: _objective(trial, x_train, y_train),
        n_trials=n_study_trials
    )

    tuned_model = XGBRegressor(**study.best_params, objective = 'reg:squarederror')
    tuned_model.fit(x_train, y_train)

    _model_feature_selection(tuned_model)

    # For accuracy testing
    # return test_model(tuned_model, x_test, y_test)

    
if __name__ == "__main__":
    get_predictions()