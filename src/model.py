import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load data
data = pd.read_csv("data/employee_data.csv")

# Features & target
X = data.drop(["performance", "performance_score"], axis=1)
y = data["performance"]

# Convert target to numbers
y = y.map({"Low": 0, "Medium": 1, "High": 2})

# Save column order (IMPORTANT)
joblib.dump(X.columns.tolist(), "models/columns.pkl")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model (improved)
model = XGBClassifier(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=200,
    objective='multi:softmax',
    num_class=3
)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/xgb_model.pkl")

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.show()