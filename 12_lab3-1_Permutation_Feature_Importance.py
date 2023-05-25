import numpy as np
from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.inspection import permutation_importance


# as_frame param requires scikit-learn >- 0.23
data = load_wine(as_frame=True)

print('Dataset shape: ', data.frame.shape)

x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, random_state=42)

# Instantiate StandardScaler
scaler = StandardScaler()

# Fit it to the train data
scaler.fit(x_train)

# Use it to transform the train and test data
x_train = scaler.transform(x_train)

# Notice that the scaler is trained on the train data to avoid data leakage from the test set
x_test = scaler.transform(x_test)

# Fit the classifier
rf_clf = RandomForestClassifier(n_estimators=10, random_state=42).fit(x_train, y_train)

# Print the mean accuracy achieved by the classifier on the test set
print('Mean accuracy achieved by the classifier on the test set: ', rf_clf.score(x_test, y_test))

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

feature_importance(rf_clf, x_train, y_train)
print('==='*10)
feature_importance(rf_clf, x_test, y_test)

## ------------------------------------------------------------
# Preserve only the top 3 features
X_train_top_features3 = x_train[:,[6, 9, 12]]
X_test_top_features3 = x_test[:,[6, 9, 12]]

# Re-train with only these features
rf_clf_top3 = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_train_top_features3, y_train)

# Compute mean accuracy achieved
print('\nRetraining with only the 3 top features: ', rf_clf_top3.score(X_test_top_features3, y_test))

## ------------------------------------------------------------
# Preserve only the top 4 features
X_train_top_features4 = x_train[:,[0, 6, 9, 12]]
X_test_top_features4 = x_test[:,[0, 6, 9, 12]]

# Re-train with only these features
rf_clf_top4 = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_train_top_features4, y_train)

# Compute mean accuracy achieved
print('\nRetraining with only the 4 top features: ', rf_clf_top4.score(X_test_top_features4, y_test))





## ------------------------------------------------------------
print('\n===> REPORT')
print('RANDOM FOREST MODEL')
print('Accuracy using all features: ', rf_clf.score(x_test, y_test))
print('Accuracy using top 3 features: ', rf_clf_top3.score(X_test_top_features3, y_test))
print('Accuracy using top 4 features: ', rf_clf_top4.score(X_test_top_features4, y_test))