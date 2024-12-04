from sklearn.linear_model import LinearRegression
from utilities import load_data, test_model


def main():
    x_train, y_train, x_test, y_test = load_data("stock_data_train.csv", 0.8)
    model = LinearRegression()

    model.fit(x_train, y_train)
    print(f"Coefficients: {model.coef_}")

    test_model(model, x_test, y_test)


if __name__ == "__main__":
    main()