import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

data = pd.DataFrame({
    "age": np.random.randint(20, 60, n),
    "experience": np.random.randint(1, 20, n),
    "salary": np.random.randint(20000, 100000, n),
    "training_hours": np.random.randint(10, 100, n),
    "department": np.random.choice(["HR", "Tech", "Sales"], n)
})

# Improved performance formula (more spread)
data["performance_score"] = (
    0.5 * data["experience"] +
    0.4 * data["training_hours"] +
    0.0005 * data["salary"] +
    np.random.normal(0, 5, n)   # adds randomness
)

# Balanced bins (VERY IMPORTANT FIX)
data["performance"] = pd.cut(
    data["performance_score"],
    bins=[0, 30, 60, 120],
    labels=["Low", "Medium", "High"]
)

# One-hot encoding
data = pd.get_dummies(data, columns=["department"])

data.to_csv("data/employee_data.csv", index=False)

print("Dataset Created Successfully!")
print(data["performance"].value_counts())  # check balance