# Phase 1: Understand the Data
import pandas as pd

# Load the dataset
df = pd.read_csv('Bank Customer Churn Prediction.csv')

# Display the first and last few rows of the dataset
print("First few rows of the dataset:")
print(df.head())
print("\nLast few rows of the dataset:")
print(df.tail())

# Display dataset structure
print("\nDataset Info:")
print(df.info())  

# Descriptive statistics of the dataset
print("\nDescriptive Statistics:")
print(df.describe())

# Class imbalance check (for churn column)
print("\nClass distribution (imbalance checking):")
print(df['churn'].value_counts())

# Correlation between numeric columns
corr_matrix = df.corr(numeric_only=True)
print("\nCorrelation Matrix:")
print(corr_matrix)

# To see the correlation with 'churn' specifically
print("\nCorrelation with Churn:")
print(corr_matrix['churn'].sort_values(ascending=False))


# Phase 2: Preprocess the Data
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# 1. Drop irrelevant features like 'customer_id'
df = df.drop('customer_id', axis=1)

# 2. One-hot encode categorical features (e.g., 'country', 'gender')
df = pd.get_dummies(df, drop_first=True)

# 3. Check for missing values (the dataset doesn't have any missing values based on previous checks)
#    If there were missing values, you would handle them here.

# 4. Feature scaling (for models that require scaling, like Logistic Regression or SVM)
numeric_features = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary']
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# 5. Handle class imbalance using SMOTE
# Separate features (X) and target (y)
X = df.drop('churn', axis=1)  # Features
y = df['churn']  # Target variable (churn)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Verify the new distribution of the target variable after resampling
print("\nClass distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())


# Phase 3: Choose the Right Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialize the Random Forest model
model = RandomForestClassifier(random_state=42)


# Phase 4: Train/Test Well
# 1. Train the model on the training data
model.fit(X_train, y_train)

# 2. Make predictions on the test data
y_pred = model.predict(X_test)

# 3. Evaluate the model's performance using classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import joblib

# Save the trained RandomForest model to a .pkl file
joblib.dump(model, 'model.pkl')
