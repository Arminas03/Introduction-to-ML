import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error


def load_data(path, split_percentage, stock_name=""):
    data = pd.read_csv(path)
    data = data[data['ticker'].str.strip().str.startswith(stock_name)]
    data['ticker'] = OneHotEncoder(sparse_output=False).fit_transform(data[['ticker']])

    i_split = int(split_percentage * len(data))
    selected_features = ['medrv_lag', 'rv_lag_1', 'vix_lag', 'rv_lag_5']

    x = data[selected_features]
    y = data['rv_lead_1'].values

    return x[:i_split], y[:i_split], x[i_split:], y[i_split:]


def plot_preds(y_test, y_pred):
    indexes = np.arange(len(y_pred))
    plt.plot(indexes, y_test, label='y_test', color='blue', marker='o')
    plt.plot(indexes, y_pred, label='y_pred', color='red', marker='x')

    plt.xlabel('Day')
    plt.ylabel('Value')
    plt.title('Predictions vs True Values')

    plt.show()


def test_model(model, x_test, y_test):
    y_pred = model.predict(x_test)

    print(f"MSE = {mean_squared_error(y_test, y_pred)}")
    plot_preds(y_test, y_pred)