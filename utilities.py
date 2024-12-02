import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def load_data(path, split_percentage):
    data = pd.read_csv(path)
    data = data[data['ticker'].str.strip().str.startswith("BA")]
    i_split = int(split_percentage * len(data))

    data['ticker'] = OneHotEncoder(sparse_output=False).fit_transform(data[['ticker']])

    # x = data.drop(columns=['date', 'rv_lead_1'])
    x = data[['medrv_lag', 'rv_lag_1', 'vix_lag', 'rv_lag_5']]
    y = data['rv_lead_1'].values

    return x[:i_split], y[:i_split], x[i_split:], y[i_split:]