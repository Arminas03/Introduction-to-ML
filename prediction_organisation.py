import pandas as pd


def main():
    predictions = {
        0: pd.read_csv('final_predictions_MSFT.csv').values,
        1: pd.read_csv('final_predictions_GE.csv').values,
        2: pd.read_csv('final_predictions_AAPL.csv').values,
        3: pd.read_csv('final_predictions_BA.csv').values,
        4: pd.read_csv('final_predictions_JNJ.csv').values
    }

    template = pd.read_csv('stock_data_test.csv')

    for i in range(len(template['rv_lead_1'].values)):
        template.loc[i, "rv_lead_1"] = predictions[i % 5][i // 5]

    template.to_csv('predictions_group_11.csv', index=False, header=True)


if __name__ == "__main__":
    main()