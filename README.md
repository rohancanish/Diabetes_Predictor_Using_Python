# Diabetes Predictor Using Python

## Overview

The **Diabetes Predictor** is a machine learning project that aims to predict whether a patient has diabetes based on various medical parameters. This project uses Python and various libraries such as Pandas, NumPy, Scikit-learn, and more. The dataset used is the PIMA Indians Diabetes Database.

## Features

- Data preprocessing and cleaning
- Exploratory data analysis (EDA)
- Model training and evaluation
- Predictions and results visualization

## Dataset

The dataset used in this project is the diabetes.csv file . It contains the following attributes:

1. Pregnancies
2. Glucose
3. BloodPressure
4. SkinThickness
5. Insulin
6. BMI
7. DiabetesPedigreeFunction
8. Age
9. Outcome

## Requirements

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook (optional, for running the project in a notebook interface)

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/rohancanish/Diabetes_Predictor_Using_Python.git
    cd Diabetes_Predictor_Using_Python
    ```

2. **Create a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate    # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Jupyter Notebook:**

    ```bash
    jupyter notebook
    ```

2. Open `diabetes_predictor.ipynb` and follow the instructions in the notebook to preprocess the data, train the model, and make predictions.

3. Alternatively, you can run the script directly from the command line:

    ```bash
    python diabetes_predictor.py
    ```

## Project Structure

