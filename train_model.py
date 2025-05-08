import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load data
data = pd.read_csv("data.csv")

# Features and target
X = data.drop("Attrition", axis=1)
y = data["Attrition"]

# Save feature names
FEATURE_NAMES = X.columns.tolist()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale all features (same as your training code)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model (tuned SVM)
svm = SVC(probability=True, C=1, kernel='rbf', random_state=42)
svm.fit(X_train_scaled, y_train)

# Save everything
with open("svm_model.pkl", "wb") as f:
    pickle.dump(svm, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("features.pkl", "wb") as f:
    pickle.dump(FEATURE_NAMES, f)

