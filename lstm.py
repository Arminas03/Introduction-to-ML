import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
from utilities import plot_preds


def _extract_and_transform_data(x: pd.DataFrame, y: pd.DataFrame, selected_features: list):
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    return (
        x_scaler.fit_transform(x[selected_features].values),
        y_scaler.fit_transform(y.values),
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

    x = data.drop(columns = ['rv_lead_1'])
    y = data[['rv_lead_1']]

    selected_features = ['medrv_lag', 'vix_lag', 'rv_minus_lag']
    x_train, y_train, x_test, y_test = (
        x[:i_split], y[:i_split], x[i_split:], y[i_split:]
    )
    
    x_train, y_train, _ = _extract_and_transform_data(x_train, y_train, selected_features)
    x_test, y_test, y_scaler_test = _extract_and_transform_data(x_test, y_test, selected_features)
    
    x_seq_train, y_seq_train = _sequential_data(x_train, y_train, seq_length)
    x_seq_test, y_seq_test = _sequential_data(x_test, y_test, seq_length)

    return (
        x_seq_train, y_seq_train,
        x_seq_test, y_seq_test,
        y_scaler_test
    )


def _model_setup(initial_units=16, dr=0):
    return Sequential([
        LSTM(initial_units, return_sequences=True, activation="relu", use_bias=True, dropout=dr),
        LSTM(int(initial_units/2), return_sequences=True, activation="relu", use_bias=True, dropout=dr),
        LSTM(int(initial_units/4), activation="relu", use_bias=True, dropout=dr),
        Dense(int(initial_units/8), activation="relu", use_bias=True),
        Dense(1)
    ])

    # return Sequential([
    #     LSTM(2),
    #     Dense(1)
    # ])


def _train_model(x, y, model: Sequential):
    model.compile(optimizer='adam', loss='mse')
    model.fit(x, y, epochs = 100)

    model.summary()


def _test_model(model: Sequential, x_test, y_test, y_scaler: StandardScaler, return_mse=False):
    y_pred = model.predict(x_test)

    y_test = y_scaler.inverse_transform(y_test)
    y_pred = y_scaler.inverse_transform(y_pred).ravel()
    mse = mean_squared_error(y_test, y_pred)

    print(mse)
    plot_preds(y_test, y_pred)

    return mse if return_mse else y_pred


def test_grid_parameters(
        grid_search_params, row_label, column_label, num_cells
):
    x_train, y_train, x_test, y_test, y_scaler = _load_data_lstm(
        'stock_data_train.csv', 0.8, seq_length = row_label
    )

    model = _model_setup(num_cells, column_label)
    _train_model(x_train, y_train, model)

    grid_search_params.loc[row_label, column_label] = _test_model(
        model, x_test, y_test, y_scaler, True
    )


def grid_search():
    row_labels = [1, 3, 5, 7, 9]
    column_labels = [0.2, 0.35, 0.5]
    num_cells_first_layer = [32, 64]

    grid_search_params32 = pd.DataFrame(index=row_labels, columns=column_labels)
    grid_search_params64 = pd.DataFrame(index=row_labels, columns=column_labels)

    for row_label in row_labels:
        for column_label in column_labels:
            for num_cells in num_cells_first_layer:
                test_grid_parameters(
                    grid_search_params32 if num_cells == 32 else grid_search_params64,
                    row_label, column_label, num_cells
                )

    print(grid_search_params32)
    print(grid_search_params64)


def get_predictions():
    x_train, y_train, x_test, y_test, y_scaler_test = _load_data_lstm(
        'stock_data_train.csv', 0.8, seq_length = 1
    )

    model = _model_setup()
    _train_model(x_train, y_train, model)

    return _test_model(model, x_test, y_test, y_scaler_test)


if __name__ == '__main__':
    get_predictions()