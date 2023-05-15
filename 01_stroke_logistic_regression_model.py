import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

from utils import preprocess_data, oversample

# set path
filepath = 'C:/Users/estela.ribeiro/GIT_er/healthcare-dataset-stroke-data.csv'
# read data
data = pd.read_csv(filepath)

# preprocess data
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
preprocessed_data = preprocess_data(data, categorical_cols)

# Removing NaN values
# Fill missing values of BMI with 0
preprocessed_data.bmi = preprocessed_data.bmi.fillna(0)

# Drop ID column
preprocessed_data.drop(['id'], axis=1, inplace=True)

print('Data Shape:', preprocessed_data.shape)

X = preprocessed_data.iloc[:,:-1]
y = preprocessed_data.iloc[:,-1]

# Split the data for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Oversample the train data
X_train, y_train = oversample(X_train, y_train)

print('Data train after oversampling: ', X_train.shape)

print('Training...')
logreg_model = LogisticRegression(random_state=42)
logreg_model.fit(X_train, y_train)
print('Training finished.')

# Evaluate model predictions
y_pred = logreg_model.predict(X_test)

print(f"F1 Score {f1_score(y_test, y_pred, average='macro')}")
print(f"Accuracy {accuracy_score(y_test, y_pred)}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
#print('Confusion Matrix: ')
#print(conf_matrix)

plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Generate ROC
fpr, tpr, thresholds = roc_curve(y_test, logreg_model.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)

# Plot AUROC
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Feature importance
importance = logreg_model.coef_[0]

# Sort feature importance
sorted_indices = np.argsort(importance)[::-1]
sorted_importance = importance[sorted_indices]
sorted_features = X.columns[sorted_indices]

for i,v in enumerate(sorted_importance):
    print('Feature: %s, Score: %.5f' % (X.columns[i],v))

# Define colors for positive and negative values
colors = ['red' if imp < 0 else 'blue' for imp in sorted_importance]

# Plot the feature importance with colored bars
plt.figure(figsize=(10, 6))
plt.bar(range(len(importance)), sorted_importance, color=colors, tick_label=sorted_features)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

# Select ans individual instance for feature importance analysis
instance_index = 0
individual_instance = X_test.iloc[instance_index]

# Get feature importance for the individual instance
individual_importance = logreg_model.predict_proba([individual_instance])[0]

# Display feature importance for the individual instance
print('Feature Importance for Individual Instance')
for ind_feature, ind_importance in zip(X.columns, individual_importance):
    print(f'{ind_feature}: {ind_importance}')

# Sort individual importance
individual_sorted_indices = np.argsort(individual_importance)[::-1]
individual_sorted_importance = importance[individual_sorted_indices]
individual_sorted_features = X.columns[individual_sorted_indices]

# Plot the feature importance with colored bars
plt.figure(figsize=(10, 6))
plt.bar(range(len(individual_importance)), individual_sorted_importance, color=colors, tick_label=individual_sorted_features)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Individual Feature Importance')
plt.tight_layout()
plt.show()