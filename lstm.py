import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
from sklearn.metrics import mean_squared_error
# from keras.api.callbacks import TensorBoard - could be useful
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense


def load_data(path, split_percentage):
    data = pd.read_csv(path)
    i_split = int(split_percentage * len(data))

    x = data.drop(columns=['date', 'rv_lead_1', 'ticker']).values
    x = x.reshape((x.shape[0], 1, x.shape[1]))
    y = data['rv_lead_1'].values

    return x[:i_split], y[:i_split], x[i_split:], y[i_split:]


def model_setup():
    return Sequential([
        LSTM(8, return_sequences=True),
        LSTM(4, return_sequences=True),
        LSTM(1),
        Dense(1)
    ])


def train_model(x, y, model: Sequential):
    model.compile(optimizer='adam', loss='mse')
    model.fit(x, y, epochs=50)

    model.summary()


def main():
    x_train, y_train, x_test, y_test = load_data('stock_data_train.csv', 0.8)

    model = model_setup()
    train_model(x_train, y_train, model)
    
    print(mean_squared_error(y_test, model.predict(x_test)))


if __name__ == '__main__':
    main()