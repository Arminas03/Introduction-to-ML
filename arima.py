from statsmodels.tsa.arima.model import ARIMA
from utilities import load_data, plot_preds
from sklearn.metrics import mean_squared_error


def main():
    _, y_train, _, y_test = load_data("stock_data_train.csv", 0.8)

    model = ARIMA(y_train, order=(2, 1, 2)).fit()
    y_pred = model.forecast(steps=len(y_test))

    print(mean_squared_error(y_pred, y_test))
    plot_preds(y_test, y_pred)


if __name__ == "__main__":
    main()