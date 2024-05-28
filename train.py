import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load

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

clf = DecisionTreeClassifier(random_state=42)

clf.fit(X_train, y_train)

model_filename = 'ccf.pkl'

#joblib.dump(clf, model_filename)
dump(clf, open('ccf.joblib','wb'))


print("Model trained and saved successfully.")
