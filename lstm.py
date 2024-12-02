import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
# from keras.api.callbacks import TensorBoard - could be useful
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Dropout
from keras import regularizers
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from utilities import load_data
from sklearn.preprocessing import OneHotEncoder


def load_data_lstm(path, split_percentage):
    data = pd.read_csv(path)
    seq_length = 1

    data = data[data['ticker'].str.strip().str.startswith("BA")]
    i_split = int(split_percentage * len(data))

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x = x_scaler.fit_transform(data[['medrv_lag', 'rv_lag_1', 'vix_lag', 'rv_lag_5']].values)
    # x = data.drop(columns=['date', 'rv_lead_1', 'ticker'])
    y = y_scaler.fit_transform(data[['rv_lead_1']].values)

    x_seq = np.array([x[i:i + seq_length] for i in range(len(x) - seq_length + 1)])
    y_seq = np.array([y[i + seq_length - 1] for i in range(len(x) - seq_length + 1)])

    return x_seq[:i_split], y_seq[:i_split], x_seq[i_split:], y_seq[i_split:], x_scaler, y_scaler


def model_setup():
    return Sequential([
        LSTM(8, return_sequences=True, activation="relu", use_bias=False),
        LSTM(4, activation="relu", use_bias=False),
        Dense(1)
    ])


def train_model(x, y, model: Sequential):
    model.compile(optimizer='adam', loss='mse')
    model.fit(x, y, epochs = 300)

    model.summary()


def main():
    x_train, y_train, _, _, _, y_scaler = load_data_lstm('stock_data_train.csv', 1)

    model = model_setup()
    train_model(x_train, y_train, model)
    # y_pred = model.predict(x_test)

    x_scaler = StandardScaler()
    x_test, _, _, _, _, _ = load_data_lstm('stock_data_test.csv', 1)

    y_pred = model.predict(x_test)[:-1]
    data = pd.read_csv('stock_data_test.csv')
    data = data[data['ticker'].str.strip().str.startswith("BA")]
    y_test = data['rv_lag_1'][1:]

    # y_test = y_scaler.inverse_transform(y_test)
    y_pred = y_scaler.inverse_transform(y_pred)

    print(mean_squared_error(y_test, y_pred))

    indexes = np.arange(len(y_pred))

    plt.plot(indexes, y_test, label='y_test', color='blue', marker='o')
    plt.plot(indexes, y_pred, label='y_pred', color='red', marker='x')

    plt.xlabel('Index (or Time)')
    plt.ylabel('Value')
    plt.title('Plotting Sequential Data on the Same Graph')

    plt.show()


if __name__ == '__main__':
    main()