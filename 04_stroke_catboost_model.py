import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_curve, auc
from utils import preprocess_data, oversample

#Set path
filepath = 'C:/Users/estela.ribeiro/GIT_er/healthcare-dataset-stroke-data.csv'

# Read data
data = pd.read_csv(filepath)

# Preprocess data
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
preprocessed_data = preprocess_data(data, categorical_cols)

# Removing NaN values
## Fill missing values of BMI with 0
preprocessed_data.bmi = preprocessed_data.bmi.fillna(0)
## Drop ID column
preprocessed_data.drop(['id'], axis=1, inplace=True)

print('Data Shape: ', preprocessed_data.shape)

# Get X and y data
X = preprocessed_data.iloc[:,:-1]
y = preprocessed_data.iloc[:,-1]

# Split the data for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Oversample train data
X_train, y_train = oversample(X_train, y_train)

print('Data train after oversampling: ', X_train.shape)

# Fit model
print('Training...')
cat_model = CatBoostClassifier(random_state=42)
cat_model.fit(X_train, y_train)

# Evaluate model predictions
y_pred = cat_model.predict(X_test)

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
fpr, tpr, _ = roc_curve(y_test, cat_model.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Feature Importanc
importance = cat_model.get_feature_importance()

# Create a sorted list of feature importance values
sorted_importance = sorted(zip(X.columns, importance), key=lambda x: x[1], reverse=True)


# Print feature importance
print("Feature Importance:")
for feature, importance_score in sorted_importance:
    print(f"{feature}: {importance_score}")

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=[score for _, score in sorted_importance], y=[feature for feature, _ in sorted_importance], palette='Blues')
plt.title('Feature Importance (CatBoost)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()
