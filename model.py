import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle

warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv('indian_liver_patient.csv')
print(df.shape)
print(df.columns)
print(df.head())
print(df.describe())

# Preprocess data
df = df.drop_duplicates()
df = df.dropna()
Gender = {'Male': 1, 'Female': 2}
df.Gender = df.Gender.map(Gender)
y = df.Dataset
X = df.drop('Dataset', axis=1)

# Handle imbalance with SMOTE
sm = SMOTE(random_state=42)
X, y = sm.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=500),
    "KNeighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(),
    "GaussianNB": GaussianNB(),
    "XGBoost": XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=10),
    "LightGBM": lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=10),
    "CatBoost": CatBoostClassifier(iterations=500, learning_rate=0.05, depth=10, verbose=0),
    "SVM": SVC(kernel='rbf', C=1, gamma='scale')
}

# Train and evaluate models
for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100
        print(f"Accuracy of {name}: {acc:.2f}%")
    except Exception as e:
        print(f"Error training {name}: {e}")

# Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=[
        ('rf', models["Random Forest"]),
        ('xgb', models["XGBoost"]),
        ('lgb', models["LightGBM"]),
        ('cat', models["CatBoost"])
    ],
    final_estimator=LogisticRegression()
)
stacking_clf.fit(X_train, y_train)
y_pred_stacking = stacking_clf.predict(X_test)
print("Accuracy of Stacking Classifier: {:.2f}%".format(accuracy_score(y_test, y_pred_stacking) * 100))

# Save the best model
best_model = stacking_clf  # Since stacking classifier might perform the best
pickle.dump(best_model, open("best_model.pkl", "wb"))
