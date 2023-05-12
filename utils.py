import pandas as pd
from imblearn.over_sampling import RandomOverSampler

# Preprocess of data
def preprocess_data(data, categorical_cols):
    '''
    One-hot encode of all categorical columns
    '''
    encoded = pd.get_dummies(data[categorical_cols], prefix=categorical_cols)
    
    # update data with new columns
    data = pd.concat([encoded, data], axis=1)
    data.drop(categorical_cols, axis=1, inplace=True)

    return data

# Oversample data (for data imbalance)
def oversample(X, y):
    over = RandomOverSampler(sampling_strategy='minority')
    # Convert to numpy
    x_np = X.to_numpy()
    y_np = y.to_numpy()
    # Oversample
    x_np, y_np = over.fit_resample(x_np, y_np)
    # Convert to pandas
    x_pd = pd.DataFrame(x_np, columns = X.columns)
    y_pd = pd.Series(y_np, name = y.name)
    return x_pd, y_pd