from utilities import load_data
from sklearn.linear_model import LinearRegression
import pandas as pd


def get_final_predictions(stock_name):
    x_train, y_train, _, _ = load_data("stock_data_train.csv", 1, stock_name)
    x_test, _, _, _, = load_data('stock_data_test.csv', 1, stock_name)
    
    model = LinearRegression()
    model.fit(x_train, y_train)
    print(model.coef_)

    y_pred = model.predict(x_test)

    pd.DataFrame(y_pred, columns=[f"Predictions_{stock_name}"]).to_csv(f"final_predictions_{stock_name}.csv", index=False)


if __name__ == "__main__":
    get_final_predictions("MSFT")