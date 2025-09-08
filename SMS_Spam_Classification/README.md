# SMS Spam Classification Project

## Group Members Detail:
- Name: Muhammad Amir
- Uni Roll No: 846112
- College Roll No: Cs-1712

## Project Title:
SMS Spam Classification

## Project Domain / Category:
Data Science/Machine Learning

## Abstract / Introduction:
Unwanted SMS messages, commonly known as spam SMS, pose significant problems for mobile users. Identifying such spam messages is a critical challenge in internet and wireless networks. This project utilizes Python text classification techniques to identify and classify spam messages. We evaluate the accuracy, time, and error rates by applying suitable algorithms such as Naïve Bayes (Gaussian and Multinomial) and J48 (Decision Tree) on an SMS Dataset. The project also includes a comparison to determine which algorithm performs best for this text classification task.

## Project Functionalities / Features:

### 1. Pre-processing
Most real-world data is incomplete, noisy, and contains missing values. Therefore, a crucial pre-processing step is applied to clean and prepare the data for analysis. This involves:
- **Lowercasing**: Converting all text to lowercase to ensure uniformity.
- **Punctuation and Number Removal**: Eliminating special characters and numerical digits that do not contribute to the classification.
- **Tokenization**: Breaking down text into individual words or tokens.
- **Stop Word Removal**: Removing common words (e.g., 'the', 'is', 'a') that do not carry significant meaning.
- **Stemming**: Reducing words to their root form (e.g., 'running' to 'run') to reduce dimensionality.

### 2. Feature Selection
After pre-processing, feature selection is performed to identify the most relevant features for classification. In this project, we used **TF-IDF (Term Frequency-Inverse Document Frequency)** for feature extraction. TF-IDF is a numerical statistic that reflects how important a word is to a document in a collection or corpus. While the prompt mentioned 'Best First Feature Selection', TF-IDF is a more common and effective method for text classification feature extraction, which was implemented.

### 3. Apply Spam Filter Algorithms
This project implements and evaluates three classification algorithms:
- **Gaussian Naïve Bayes**: A probabilistic classifier based on Bayes' theorem with the assumption of independence between features. It assumes that the features follow a Gaussian (normal) distribution.
- **Multinomial Naïve Bayes**: Another variant of Naïve Bayes, particularly suitable for classification with discrete features (e.g., word counts for text classification). It is widely used in document classification.
- **Decision Tree (J48)**: J48 is an open-source Java implementation of the C4.5 algorithm, which builds a decision tree from a set of training data. It is used here to represent the decision tree classifier.

The process involves:
- **Handle Data**: Loading the dataset and splitting it into training and test datasets.
- **Summarize Data**: Summarizing the properties in the training dataset to calculate probabilities and make predictions.
- **Make a Prediction**: Using the summaries of the dataset to generate a single prediction.
- **Make Predictions**: Generating predictions given a test dataset and a summarized training dataset.

### 4. Train & Test Data
The dataset is split into 75% for training and 25% for testing. This standard practice ensures that the models are trained on a significant portion of the data and then evaluated on unseen data to assess their generalization capability.

### 5. Confusion Matrix
A confusion matrix is generated for each algorithm to describe the performance of the classification model. It provides a detailed breakdown of correct and incorrect predictions, showing:
- **True Positives (TP)**: Correctly predicted spam messages.
- **True Negatives (TN)**: Correctly predicted ham (non-spam) messages.
- **False Positives (FP)**: Ham messages incorrectly predicted as spam (Type I error).
- **False Negatives (FN)**: Spam messages incorrectly predicted as ham (Type II error).

### 6. Accuracy
The accuracy of all algorithms is calculated and compared. Accuracy is defined as the ratio of correctly predicted observations to the total observations. This metric helps in determining which algorithm performs best in classifying SMS messages as spam or ham.

## Tools / Technologies:
- **Python**: The primary programming language used for implementing the classification models and data processing.
- **Anaconda**: A distribution of Python and R for scientific computing, used for managing packages and environments.
- **Libraries**: `pandas` for data manipulation, `nltk` for natural language processing, and `scikit-learn` for machine learning algorithms and utilities.

## How to Run the Project:

1.  **Prerequisites**:
    - Ensure you have Python 3.x installed.
    - Install Anaconda (recommended) or `pip`.

2.  **Setup Environment**:
    - If using `pip`, install the required libraries:
      ```bash
      pip install pandas nltk scikit-learn kagglehub
      ```
    - Download NLTK data (punkt and stopwords):
      ```bash
      python -m nltk.downloader punkt stopwords punkt_tab
      ```

3.  **Download the Dataset**:
    - The `spam.csv` dataset is automatically downloaded and placed in the project directory when you run `download_dataset.py`.

4.  **Run the Scripts**:
    Execute the Python scripts in the following order:

    a.  **Download Dataset (if not already present)**:
        ```bash
        python download_dataset.py
        ```

    b.  **Pre-processing and Feature Selection**:
        ```bash
        python preprocess_features.py
        ```
        This script will generate `tfidf_vectorizer.pkl` and `processed_data.pkl`.

    c.  **Train Models**:
        ```bash
        python train_models.py
        ```
        This script will train the Gaussian Naive Bayes, Multinomial Naive Bayes, and Decision Tree (J48) models and save them as `.pkl` files.

    d.  **Evaluate Models**:
        ```bash
        python evaluate_models.py
        ```
        This script will load the trained models, make predictions, calculate accuracy, generate confusion matrices, and print the evaluation results. The results will also be saved to `model_evaluation_results.pkl`.

## Results:

### Model Evaluation Results:
```
                     Model  Accuracy          Confusion Matrix
0     Gaussian Naive Bayes  0.858579  [[1033, 169], [28, 163]]
1  Multinomial Naive Bayes  0.967696    [[1202, 0], [45, 146]]
2      Decision Tree (J48)  0.954056   [[1174, 28], [36, 155]]
```

**Interpretation of Results:**

-   **Multinomial Naive Bayes** achieved the highest accuracy (0.9677 or 96.77%), making it the best-performing algorithm among the three for this SMS spam classification task. Its confusion matrix shows a high number of true positives and true negatives, with relatively few false positives and false negatives.
-   **Decision Tree (J48)** performed well with an accuracy of 0.9541 (95.41%), indicating strong performance, though slightly lower than Multinomial Naive Bayes.
-   **Gaussian Naive Bayes** had the lowest accuracy (0.8586 or 85.86%) and a higher number of false positives and false negatives compared to the other two models, suggesting it is less suitable for this specific text classification problem with the current feature representation.

This comparison clearly indicates that **Multinomial Naive Bayes** is the most effective algorithm for SMS spam classification based on the given dataset and feature extraction method.


