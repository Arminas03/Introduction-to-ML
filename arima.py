from statsmodels.tsa.arima.model import ARIMA
from utilities import load_data
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


def main():
    _, y_train, _, y_test = load_data("stock_data_train.csv", 0.8)

    model = ARIMA(y_train, order=(2, 1, 2))
    fitted_model = model.fit()
    y_pred = fitted_model.forecast(steps=len(y_test))

    print(mean_squared_error(y_pred, y_test))

    indexes = np.arange(len(y_pred))

    plt.plot(indexes[500:600], y_test[500:600], label='y_test', color='blue', marker='o')
    plt.plot(indexes[500:600], y_pred[500:600], label='y_pred', color='red', marker='x')

    plt.xlabel('Index (or Time)')
    plt.ylabel('Value')
    plt.title('Plotting Sequential Data on the Same Graph')

    plt.show()

if __name__ == "__main__":
    main()