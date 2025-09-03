# Titanic Survival Prediction

This project analyzes the classic Titanic dataset to predict whether a passenger survived the disaster. It covers the entire data science workflow, including:

- **Exploratory Data Analysis (EDA)** to understand the data and find initial insights.
- **Feature Engineering** to create new, meaningful features from the existing data.
- **Data Preprocessing** to handle missing values and prepare data for modeling.
- **Machine Learning Modeling** to train and evaluate several classification algorithms to predict passenger survival.

## 🎯 Project Goal

The primary objective is to build a machine learning model that accurately predicts if a passenger survived the Titanic shipwreck, based on features like their age, sex, class, and fare.

## 📊 Dataset

The project uses the Titanic dataset provided by Kaggle. It is split into two files:

- `train.csv`: Contains training data along with the ground truth (the `Survived` column).
- `test.csv`: Contains test data for which we need to predict the survival outcome.

### Key Features

- **Survived**: The target variable. `0` for deceased, `1` for survived.
- **Pclass**: Ticket class (`1` = 1st, `2` = 2nd, `3` = 3rd). A proxy for socio-economic status.
- **Sex**: Passenger's sex.
- **Age**: Passenger's age in years.
- **SibSp**: Number of siblings or spouses aboard the Titanic.
- **Parch**: Number of parents or children aboard the Titanic.
- **Ticket**: Ticket number.
- **Fare**: Passenger fare.
- **Cabin**: Cabin number.
- **Embarked**: Port of embarkation (`C` = Cherbourg, `Q` = Queenstown, `S` = Southampton).

## ⚙️ Project Workflow

The analysis is structured in a sequential manner to ensure a clear and logical workflow.

### 1. Exploratory Data Analysis (EDA)

In this step, I dived deep into the dataset to uncover patterns, anomalies, and relationships between variables. Key activities included:
- Analyzing the distribution of individual features (e.g., Age, Fare).
- Visualizing the survival rate based on different features like `Sex`, `Pclass`, and `Embarked`.
- Investigating correlations between features.
- Understanding the extent of missing data in columns like `Age`, `Cabin`, and `Embarked`.

### 2. Feature Engineering & Preprocessing

Based on insights from EDA, the data was cleaned and transformed to be suitable for machine learning models.
- **Handling Missing Values**:
  - `Age`: Filled missing values using the median age, grouped by `Pclass` and `Sex`.
  - `Embarked`: Filled missing values with the most frequent port of embarkation.
  - `Cabin`: Created a new feature `Has_Cabin` to indicate if a passenger had a recorded cabin number, as the specific cabin number had too many missing values.
- **Creating New Features**:
  - `FamilySize`: Combined `SibSp` and `Parch` to get the total family size.
  - `IsAlone`: A binary feature derived from `FamilySize`.
- **Converting Categorical Features**:
  - Converted `Sex` and `Embarked` into numerical format using one-hot encoding.

### 3. Model Building & Evaluation

Several classification algorithms were trained on the preprocessed data to find the best-performing model. The models I experimented with include:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- Gradient Boosting

Models were evaluated based on their **accuracy** on a validation set. The final model's performance was also assessed using a **confusion matrix** to understand the types of errors it makes (false positives vs. false negatives).

## 🏆 Results

After training and evaluation, the **Gradient Boosting** model performed the best, achieving a cross-validated accuracy of approximately **83%**.

The most important features for predicting survival were found to be `Title`, `Sex`, `Pclass`, and `Fare`.

## 🛠️ Technologies & Libraries Used

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook