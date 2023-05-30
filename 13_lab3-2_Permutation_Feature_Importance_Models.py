import numpy as np
from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier

from sklearn.inspection import permutation_importance


# as_frame param requires scikit-learn >- 0.23
data = load_wine(as_frame=True)

print('Dataset shape: ', data.frame.shape)

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=42)

# Instantiate StandardScaler
scaler = StandardScaler()

# Fit it to the train data
scaler.fit(X_train)

# Use it to transform the train and test data
X_train = scaler.transform(X_train)

# Notice that the scaler is trained on the train data to avoid data leakage from the test set
X_test = scaler.transform(X_test)

## -----------------------------------------
# Select 5 new classifiers
clfs = {"Random Forest": RandomForestClassifier(n_estimators=10, random_state=42),
        "Laso": Lasso(alpha=0.05), 
        "Ridge": Ridge(), 
        "Decision Tree": DecisionTreeClassifier(), 
        "Support Vector": SVC()}

# Feature Importance Function
def feature_importance(clf, x, y, top_limit=None):
    # Retrieve the bunch object after 50 repeats
    # n_repeats is the number of times that each feature was permuted to compute the final score
    bunch = permutation_importance(clf, x, y, n_repeats=50, random_state=42)

    # Average feature importance
    imp_means = bunch.importances_mean

    # List that contains the index of each feature in descending order of importance
    ordered_imp_means_args = np.argsort(imp_means)[::-1]

    # If no limit print all features
    if top_limit is None:
        top_limit = len(ordered_imp_means_args)

    # Print relevant information
    for i, _ in zip (ordered_imp_means_args, range(top_limit)):
        name = data.feature_names[i]
        imp_score = imp_means[i]
        imp_std = bunch.importances_std[i]
        print(f"Feature {name} with index {i} has an average importance score of {imp_score:.3f} +/- {imp_std:.3f}\n")

# Compute feature importance on the test set given a classifier
def fit_compute_importance(clf):
  clf.fit(X_train, y_train)
  print(f"📏 Mean accuracy score on the test set: {clf.score(X_test, y_test)*100:.2f}%\n")
  print("🔝 Top 4 features when using the test set:\n")
  feature_importance(clf, X_test, y_test, top_limit=4)

# Print results
for name, clf in clfs.items():
  print("====="*20)
  print(f"➡️ {name} classifier\n")
  fit_compute_importance(clf)