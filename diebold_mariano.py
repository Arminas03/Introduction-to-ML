import regression, lstm
from utilities import load_data
import numpy as np
from math import sqrt
from scipy.stats import norm


def calculate_test_statistic(y_true, y_pred_1, y_pred_2):
    err_1 = np.array([(y - y_p) ** 2 for y, y_p in zip(y_true, y_pred_1)])
    err_2 = np.array([(y - y_p) ** 2 for y, y_p in zip(y_true, y_pred_2)])

    d = err_1 - err_2

    d_bar = np.mean(d)
    n = len(y_true)
    var_d = sum([(d_t - d_bar) ** 2 for d_t in d]) / n

    return d_bar / sqrt(var_d / n)


def main():
    _, _, _, y_true = load_data("stock_data_train.csv", 0.8)

    y_pred_1 = regression.get_predictions()
    y_pred_2 = lstm.get_predictions()

    dm_statistic = calculate_test_statistic(y_true, y_pred_1, y_pred_2)

    print(f"DM-statistic = {dm_statistic}")
    print(f"Resulting p-value under 0.05 significance level: {
            2 * (1 - norm.cdf(abs(dm_statistic)))
        }"
    )


if __name__ == "__main__":
    main()