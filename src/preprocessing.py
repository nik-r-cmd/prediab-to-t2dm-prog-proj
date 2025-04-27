import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle

DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "youth_diabetes.csv"))

df.drop(columns=["ID", "Diabetes_Type"], inplace=True, errors="ignore")
df = df[df["Parent_Diabetes_Type"] != "Type 1"]

df["BMI_Category"] = pd.cut(df["BMI"], [0, 18.5, 25, 30, np.inf], labels=["Underweight", "Normal", "Overweight", "Obese"])
df["FBS_Risk"] = pd.cut(df["Fasting_Blood_Sugar"], [0, 100, 125, np.inf], labels=["Normal", "Prediabetes", "Diabetes"])
df["HbA1c_Risk"] = pd.cut(df["HbA1c"], [0, 5.7, 6.4, np.inf], labels=["Normal", "Prediabetes", "Diabetes"])
df["Cholesterol_Risk"] = pd.cut(df["Cholesterol_Level"], [0, 200, 240, np.inf], labels=["Normal", "Borderline", "High"])

df["Metabolic_Risk_Score"] = (
    ((df["BMI"] >= 30) * 2) +
    ((df["Fasting_Blood_Sugar"] >= 126) * 2) +
    ((df["HbA1c"] >= 6.5) * 2) +
    ((df["Cholesterol_Level"] >= 240))
).astype(int)

act_map = {"Sedentary": 3, "Moderate": 1, "Active": 0}
diet_map = {"Unhealthy": 3, "Moderate": 1, "Healthy": 0}

df["Lifestyle_Risk_Score"] = (
    df["Physical_Activity_Level"].map(act_map).fillna(0) +
    df["Dietary_Habits"].map(diet_map).fillna(0) +
    (df["Fast_Food_Intake"] / 10 * 3).astype(int) +
    (df["Smoking"] == "Yes") * 2 +
    (df["Alcohol_Consumption"] == "Yes") +
    (df["Screen_Time"] / 12 * 2).astype(int)
)

df["Risk_Progression_Years"] = (
    10 - (
        df["Genetic_Risk_Score"] * 0.5 +
        df["Metabolic_Risk_Score"] * 0.8 +
        df["Lifestyle_Risk_Score"] * 0.6 +
        (df["HbA1c"] - 5.0) * 0.7 +
        (df["Fasting_Blood_Sugar"] - 90) * 0.05 +
        (df["Cholesterol_Level"] - 180) * 0.01 +
        (df["Stress_Level"] * 0.3) -
        (df["Sleep_Hours"] - 7) * 0.2
    )
).clip(1, 10).round(1)

df.to_csv(os.path.join(DATA_DIR, "full_dataset_with_target.csv"), index=False)

leaky = ["Metabolic_Risk_Score", "Lifestyle_Risk_Score", "FBS_Risk", "HbA1c_Risk", "Cholesterol_Risk", "BMI_Category"]
df_model = df.drop(columns=leaky)

X = df_model.drop(columns=["Risk_Progression_Years"])
y = df_model["Risk_Progression_Years"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
])

X_proc = preprocessor.fit_transform(X)

pd.DataFrame(X_proc).to_csv(os.path.join(DATA_DIR, "processed_diabetes_data.csv"), index=False)
np.savez(os.path.join(DATA_DIR, "train_test_split.npz"),
         X_train=X_proc, y_train=y)

joblib.dump(preprocessor, os.path.join(DATA_DIR, "preprocessor.joblib"))
with open(os.path.join(DATA_DIR, "feature_names.pkl"), "wb") as f:
    pickle.dump(preprocessor.get_feature_names_out().tolist(), f)

plt.figure(figsize=(14, 12))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=True, fmt=".2f", square=True)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "correlation_matrix.png"))
plt.close()

print("Preprocessing complete")


