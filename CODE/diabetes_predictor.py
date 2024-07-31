import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Check current working directory
print("Current Working Directory:", os.getcwd())

# Specify the correct path to the Excel dataset
data_path = "diabetes.csv"  # Update this path if needed

# Load the dataset from Excel
try:
    diabetes_dataset = pd.read_excel(data_path)  # Using read_excel instead of read_csv
except FileNotFoundError:
    print(f"Error: File '{data_path}' not found. Please check the data path.")
    exit()

# Display basic information about the dataset
print("Shape of the dataset:", diabetes_dataset.shape)
print("\nDescription of the dataset:")
print(diabetes_dataset.describe())
print("\nClass distribution (Outcome):")
print(diabetes_dataset['Outcome'].value_counts())

# Separate features (X) and target (Y) variables
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standardize the data
scaler = StandardScaler()
standardized_data = scaler.fit_transform(X)

# Split data into training and testing sets (consider stratify=Y for imbalanced data)
X_train, X_test, Y_train, Y_test = train_test_split(standardized_data, Y, test_size=0.2, random_state=2)

# Explore different SVM kernels (optional)
kernel_options = ['linear', 'rbf', 'poly']
for kernel in kernel_options:
    classifier = SVC(kernel=kernel)
    # Train, evaluate, and compare performance for each kernel

# Initialize SVM classifier (example with linear kernel)
classifier = SVC(kernel='linear', probability=True)

# Train the classifier
classifier.fit(X_train, Y_train)

# Predict on training data
Y_train_pred = classifier.predict(X_train)
training_accuracy = accuracy_score(Y_train, Y_train_pred)
print('Accuracy score on training data:', training_accuracy)

# Predict on test data
Y_test_pred = classifier.predict(X_test)
test_accuracy = accuracy_score(Y_test, Y_test_pred)
print('Accuracy score on test data:', test_accuracy)

# Example input data for prediction (adjust as needed)
input_data = np.array([5, 166, 72, 19, 175, 25.8, 0.587, 51]).reshape(1, -1)
input_data_standardized = scaler.transform(input_data)

# Predict using the trained model
prediction = classifier.predict(input_data_standardized)

# Display prediction result and consider adding probability scores
if prediction[0] == 0:
    print('Prediction: The person is not diabetic')
else:
    print('Prediction: The person is diabetic')

    # (Optional) Get class probabilities
    class_probs = classifier.predict_proba(input_data_standardized)[0]
    print(f"Probabilities: Diabetic = {class_probs[1]}, Not Diabetic = {class_probs[0]}")
