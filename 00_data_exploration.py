import pandas as pd
import matplotlib.pyplot as plt
from utils import preprocess_data

# %% Load and show data

# set path
filepath = 'C:/Users/estela.ribeiro/GIT_er/healthcare-dataset-stroke-data.csv'
# read data
data = pd.read_csv(filepath)

# show some info
print('-------------------------')
print('Data shape: ', data.shape)
print('Data Info:  ', data.info)
print('Count_values: ', data.count)

# Show histogram for all columns
columns = data.columns

for col in columns:
    print('col: ', col)
    data[col].hist()
    plt.title(col)
    plt.show()

# %% Preprocessed data

categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
preprocessed_data = preprocess_data(data, categorical_cols)

print('-------------------------')
print('Preprocessed data Shape: ', preprocessed_data.shape)
print('Preprocessed data Info:  ', preprocessed_data.info)
print('Count_values: ', preprocessed_data.count)

# Removing NaN values
# Fill missing values of BMI with 0
preprocessed_data.bmi = preprocessed_data.bmi.fillna(0)

# Drop ID column
preprocessed_data.drop(['id'], axis=1, inplace=True)