
# ============================================================
# TITANIC SURVIVAL PREDICTION PROJECT
# Logistic Regression Machine Learning Model
# VS Code Python Version
# ============================================================


# 1. IMPORT LIBRARIES

import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

print("All libraries imported successfully!")


# ============================================================
# 2. LOAD ALL THREE DATASETS
# ============================================================

required_files = [
    "train.csv",
    "test.csv",
    "gender_submission.csv"
]

for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(
            f"{file} not found! Please place it in the project folder."
        )

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
gender_submission = pd.read_csv("gender_submission.csv")

print("\nAll three datasets loaded successfully!")


# ============================================================
# 3. DISPLAY DATASETS
# ============================================================

print("\nTRAIN DATASET")
print(train_df.head())

print("\nTEST DATASET")
print(test_df.head())

print("\nGENDER SUBMISSION DATASET")
print(gender_submission.head())

print("\nDATASET SHAPES")
print("Train:", train_df.shape)
print("Test:", test_df.shape)
print("Gender Submission:", gender_submission.shape)


# ============================================================
# 4. DATASET INFORMATION
# ============================================================

print("\nTRAINING DATASET INFORMATION")
train_df.info()

print("\nMISSING VALUES IN TRAINING DATASET")
print(train_df.isnull().sum())

print("\nDUPLICATE RECORDS:", train_df.duplicated().sum())

print("\nSTATISTICAL DESCRIPTION")
print(train_df.describe())


# ============================================================
# 5. DATA VISUALIZATION - SURVIVAL COUNT
# ============================================================

plt.figure(figsize=(6, 4))

sns.countplot(data=train_df, x="Survived")

plt.title("Titanic Survival Count")
plt.xlabel("Survival (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.show()


# ============================================================
# 6. SURVIVAL BASED ON GENDER
# ============================================================

plt.figure(figsize=(6, 4))

sns.countplot(data=train_df, x="Sex", hue="Survived")

plt.title("Survival Based on Gender")

plt.tight_layout()
plt.show()


# ============================================================
# 7. SURVIVAL BASED ON PASSENGER CLASS
# ============================================================

plt.figure(figsize=(6, 4))

sns.countplot(data=train_df, x="Pclass", hue="Survived")

plt.title("Survival Based on Passenger Class")

plt.tight_layout()
plt.show()


# ============================================================
# 8. AGE DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 4))

sns.histplot(data=train_df, x="Age", bins=30, kde=True)

plt.title("Passenger Age Distribution")

plt.tight_layout()
plt.show()


# ============================================================
# 9. SELECT INPUT FEATURES AND TARGET
# ============================================================

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

X = train_df[features]
y = train_df["Survived"]

# Features for Kaggle test dataset
X_kaggle_test = test_df[features]

print("\nINPUT FEATURES")
print(X.head())

print("\nTARGET VARIABLE")
print(y.head())


# ============================================================
# 10. DEFINE NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

categorical_features = [
    "Sex",
    "Embarked"
]


# ============================================================
# 11. DATA PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("\nData preprocessing completed!")


# ============================================================
# 12. SPLIT TRAINING DATASET FOR MODEL EVALUATION
# ============================================================

X_train, X_validation, y_train, y_validation = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Validation Data:", X_validation.shape)


# ============================================================
# 13. CREATE LOGISTIC REGRESSION PIPELINE
# ============================================================

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

print("\nLogistic Regression model created successfully!")


# ============================================================
# 14. TRAIN THE MODEL
# ============================================================

model.fit(X_train, y_train)

print("Model training completed successfully!")


# ============================================================
# 15. PREDICT VALIDATION DATASET
# ============================================================

y_pred = model.predict(X_validation)

print("\nACTUAL VALUES")
print(y_validation.head(10).values)

print("\nPREDICTED VALUES")
print(y_pred[:10])


# ============================================================
# 16. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(y_validation, y_pred)

precision = precision_score(
    y_validation,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_validation,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_validation,
    y_pred,
    zero_division=0
)

print("\nMODEL EVALUATION RESULTS")
print("--------------------------------")

print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1 Score:  {f1 * 100:.2f}%")

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_validation,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_validation, y_pred)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Survived", "Survived"],
    yticklabels=["Not Survived", "Survived"]
)

plt.title("Titanic Survival Prediction - Confusion Matrix")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")

plt.tight_layout()
plt.show()


# ============================================================
# 18. TRAIN FINAL MODEL USING COMPLETE TRAINING DATASET
# ============================================================

final_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

final_model.fit(X, y)

print("\nFinal model trained using complete training dataset!")


# ============================================================
# 19. PREDICT SURVIVAL USING TEST.CSV
# ============================================================

test_predictions = final_model.predict(X_kaggle_test)

print("\nKAGGLE TEST PREDICTIONS")
print(test_predictions[:10])

print("\nTotal Test Predictions:", len(test_predictions))


# ============================================================
# 20. CREATE KAGGLE SUBMISSION FILE
# ============================================================

submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": test_predictions
})

submission.to_csv("submission.csv", index=False)

print("\nsubmission.csv created successfully!")

print("\nSUBMISSION PREVIEW")
print(submission.head())


# ============================================================
# 21. COMPARE WITH GENDER SUBMISSION BASELINE
# ============================================================

comparison = submission.merge(
    gender_submission[["PassengerId", "Survived"]],
    on="PassengerId",
    suffixes=("_Model", "_Gender")
)

agreement = (
    comparison["Survived_Model"] ==
    comparison["Survived_Gender"]
).mean()

print("\nCOMPARISON WITH GENDER BASELINE")
print("--------------------------------")

print(
    f"Agreement with gender baseline: {agreement * 100:.2f}%"
)

print("\nCOMPARISON PREVIEW")
print(comparison.head())

# NOTE:
# This is agreement, not actual test accuracy.
# gender_submission.csv contains baseline predictions,
# not the actual survival answers.


# ============================================================
# 22. PREDICT A NEW PASSENGER
# ============================================================

new_passenger = pd.DataFrame([{
    "Pclass": 3,
    "Sex": "male",
    "Age": 25,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S"
}])

prediction = final_model.predict(new_passenger)

probability = final_model.predict_proba(new_passenger)

print("\nTITANIC SURVIVAL PREDICTION")

if prediction[0] == 1:
    print("Result: Passenger Survived")
else:
    print("Result: Passenger Did Not Survive")

print(
    "Survival Probability:",
    round(probability[0][1] * 100, 2),
    "%"
)


# ============================================================
# 23. SAVE TRAINED MACHINE LEARNING MODEL
# ============================================================

joblib.dump(final_model, "titanic_model.pkl")

print("\nModel saved successfully as titanic_model.pkl")


# ============================================================
# 24. LOAD SAVED MODEL AND VERIFY
# ============================================================

loaded_model = joblib.load("titanic_model.pkl")

loaded_prediction = loaded_model.predict(new_passenger)

print("\nLoaded Model Prediction:", loaded_prediction[0])


# ============================================================
# 25. FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 50)

print("TITANIC SURVIVAL PREDICTION PROJECT")

print("=" * 50)

print("Machine Learning Algorithm: Logistic Regression")

print("Training Dataset Size:", train_df.shape)

print("Kaggle Test Dataset Size:", test_df.shape)

print("Training Samples:", len(X_train))

print("Validation Samples:", len(X_validation))

print(f"Accuracy:  {accuracy * 100:.2f}%")

print(f"Precision: {precision * 100:.2f}%")

print(f"Recall:    {recall * 100:.2f}%")

print(f"F1 Score:  {f1 * 100:.2f}%")

print("Kaggle Predictions:", len(submission))

print("\nGENERATED FILES")

print("- submission.csv")

print("- titanic_model.pkl")

print("=" * 50)

print("PROJECT COMPLETED SUCCESSFULLY!")

print("=" * 50)
