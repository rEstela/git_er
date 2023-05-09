import pandas as pd

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




    