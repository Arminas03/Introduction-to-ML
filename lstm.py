import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
import pandas as pd


def load_data(path):
    data = pd.read_csv(path)

    x = data[['rv_lag_1', 'rv_lag_5']].values
    y = data['rv_lead_1']

    return x.reshape((x.shape[0], 1, x.shape[1])), y

def main():
    x_train, y_train = load_data('stock_data_train.csv')

    model = keras.Sequential()

    model.add(keras.layers.LSTM(8, input_shape=(1, 2)))
    model.add(keras.layers.LSTM(4, input_shape=(1, 2)))

    model.compile(optimizer='sgd', loss='mse')
    model.fit(x_train, y_train, epochs=10)

    model.summary()


if __name__ == '__main__':
    main()