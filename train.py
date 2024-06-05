import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump

# Load the dataset
data = pd.read_csv('card_transdata.csv')

# Define feature columns and target column
feature_columns = [
    'distance_from_home',
    'distance_from_last_transaction',
    'ratio_to_median_purchase_price',
    'repeat_retailer',
    'used_chip',
    'used_pin_number',
    'online_order'
]
target_column = 'fraud'

# Split the data into features and target
X = data[feature_columns]
y = data[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)
dump(dt_clf, 'decision_tree_model.joblib')

dt_y_pred = dt_clf.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_y_pred)
print(f'Decision Tree Classifier Accuracy: {dt_accuracy:.2f}')

# Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)
dump(rf_clf, 'random_forest_model.joblib')

rf_y_pred = rf_clf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print(f'Random Forest Classifier Accuracy: {rf_accuracy:.2f}')

# Logistic Regression Classifier
lr_clf = LogisticRegression(max_iter=1000, random_state=42)
lr_clf.fit(X_train, y_train)
dump(lr_clf, 'logistic_regression_model.joblib')

lr_y_pred = lr_clf.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_y_pred)
print(f'Logistic Regression Classifier Accuracy: {lr_accuracy:.2f}')

print("All models trained, saved, and evaluated successfully.")