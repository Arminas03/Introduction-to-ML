import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
from utilities import plot_preds


def _extract_and_transform_data(data: pd.DataFrame, selected_features: list):
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    return (
        x_scaler.fit_transform(data[selected_features].values),
        y_scaler.fit_transform(data[['rv_lead_1']].values),
        y_scaler
    )


def _sequential_data(x, y, seq_length):
    return (
        np.array([x[i:i + seq_length] for i in range(len(x) - seq_length + 1)]),
        np.array([y[i + seq_length - 1] for i in range(len(x) - seq_length + 1)])
    )
    


def _load_data_lstm(path, split_percentage, seq_length, stock_name=""):
    data = pd.read_csv(path)

    data = data[data['ticker'].str.strip().str.startswith(stock_name)]
    i_split = int(split_percentage * len(data))

    selected_features = ['medrv_lag', 'vix_lag', 'rv_minus_lag']
    x, y, y_scaler = _extract_and_transform_data(data, selected_features)
    x_seq, y_seq = _sequential_data(x, y, seq_length)

    return (
        x_seq[:i_split], y_seq[:i_split],
        x_seq[i_split:], y_seq[i_split:],
        y_scaler
    )


def _model_setup():
    return Sequential([
        LSTM(2),
        Dense(1)
    ])


def _train_model(x, y, model: Sequential):
    model.compile(optimizer='adam', loss='mse')
    model.fit(x, y, epochs = 100)

    model.summary()


def _test_model(model: Sequential, x_test, y_test, y_scaler: StandardScaler):
    y_pred = model.predict(x_test)

    y_test = y_scaler.inverse_transform(y_test)
    y_pred = y_scaler.inverse_transform(y_pred).ravel()

    print(mean_squared_error(y_test, y_pred))
    plot_preds(y_test, y_pred)

    return y_pred


def get_predictions():
    x_train, y_train, x_test, y_test, y_scaler = _load_data_lstm(
        'stock_data_train.csv', 0.8, seq_length = 1
    )

    model = _model_setup()
    _train_model(x_train, y_train, model)

    return _test_model(model, x_test, y_test, y_scaler)


if __name__ == '__main__':
    get_predictions()