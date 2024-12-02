from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error
from utilities import load_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    model = LinearRegression()

    x_train, y_train, _, _ = load_data("stock_data_train.csv", 1)

    model.fit(x_train, y_train)

    x_test, _, _, _, = load_data('stock_data_test.csv', 1)
    data = pd.read_csv('stock_data_test.csv')
    data = data[data['ticker'].str.strip().str.startswith("BA")]
    y_test = data['rv_lag_1'][1:]

    y_pred = model.predict(x_test)[:-1]

    print(mean_squared_error(y_pred, y_test))
    print(model.coef_)

    indexes = np.arange(len(y_pred))

    plt.plot(indexes, y_test, label='y_test', color='blue', marker='o')
    plt.plot(indexes, y_pred, label='y_pred', color='red', marker='x')

    plt.xlabel('Index (or Time)')
    plt.ylabel('Value')
    plt.title('Plotting Sequential Data on the Same Graph')

    plt.show()


if __name__ == "__main__":
    main()