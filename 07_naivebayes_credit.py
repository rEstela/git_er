import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_curve, auc


# Setting the seed to allow reproducibility
np.random.seed(42)

# Read dataset
df = pd.read_csv('credit_risk_dataset.csv')

# Data exploration
print('-> Dataset shape: ', df.shape)
print(df.info())
print(df.describe())

ccol = df.select_dtypes(include=['object']).columns
ncol = df.select_dtypes(include=['int', 'float']).columns

print('\nThe number of categorical columns are:', len(ccol))
print('The number of numerical columns are: ', len(ncol))

print('\nThe NUMERICAL columns are:\n')
for i in ncol:
    print('->', i, '-', df[i].nunique())
print('\n--------------------------\n')
print('The CATEGORICAL columns are:\n')
for i in ccol:
    print('->', i, '-', df[i].nunique())

# 'loan_int_rate' describes the Interest Rate offered on Loans by Banks 
# or any financial institution. There is no fixed value as it varies from bank to bank. 
# Hence I am removing this column for our analysis.
df.drop(['loan_int_rate'], axis=1, inplace=True)

# Analysing the Target variable 'loan_status'
print('\nTarget Variable: \n', df['loan_status'].value_counts(normalize=True))

# Checking for missing values:
print('--> Missing values: \n', df.isnull().any())

# NaN values:
print('--> NaN values: \n', df.isna().sum())

# Remove nan
df.dropna(inplace=True)

# Remove person above age 80
df = df.loc[df['person_age'] < 80, :]
# Remove person employed over 60 years
df = df.loc[df['person_emp_length'] < 60, :]

# Concert categorical data into one-hot-encoding
df = pd.get_dummies(df)

x = df.drop('loan_status', axis=1)
y = df['loan_status']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size = 0.2, stratify=df['loan_status'], shuffle=True)

nb_model = GaussianNB()
nb_model.fit(x_train, y_train)

y_pred = nb_model.predict(x_test)

print(f"F1 Score {f1_score(y_test, y_pred, average='macro')}")
print(f'Accuracy {accuracy_score(y_test, y_pred)}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Generate ROC curve
fpr, tpr, _ = roc_curve(y_test, nb_model.predict_proba(x_test)[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()


# Calculate conditional probabilities
feature_importance = {}
for i, feature_name in enumerate(x.columns):
    feature_importance[feature_name] = nb_model.theta_[1, i] - nb_model.theta_[0, i]

# Sort the feature importance dictionary by values
sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

# Print the feature importance
for feature, importance in sorted_importance:
    print(f"{feature}: {importance}")

# Extract feature names and importance values for plotting
features, importance = zip(*sorted_importance)

# Create a barplot of feature importance
plt.figure(figsize=(10, 6))
plt.bar(features, importance)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()
