# Email Spam Classification

This project focuses on classifying emails as either "Spam" or "Not Spam" using machine learning techniques. The classification is based on textual features extracted from email content, and multiple models are used to evaluate performance.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Machine Learning Models](#machine-learning-models)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Conclusion](#conclusion)

## Overview
The goal of this project is to build a machine learning pipeline that can accurately classify emails as spam or not. The pipeline includes data preprocessing, feature extraction, model training, and evaluation.

## Dataset
The dataset used in this project contains email messages with corresponding labels:
- **Spam**: Emails flagged as unsolicited or junk.
- **Ham**: Legitimate, non-spam emails.

The dataset is expected to be in CSV format with at least two columns:
- `label`: Indicates whether the email is "spam" or "ham".
- `message`: The content of the email.

## Preprocessing
To prepare the data for machine learning models:
1. **Text Cleaning**: 
   - Removed punctuation.
   - Tokenized email text.
   - Removed stopwords.
2. **Feature Extraction**: 
   - Used TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical features with a limit of 3000 features.
3. **Label Encoding**: 
   - Mapped "ham" to `0` and "spam" to `1`.

## Machine Learning Models
The following machine learning models from the `scikit-learn` library were implemented and evaluated:
- **Logistic Regression**
- **Support Vector Machines (SVM)**
- **Decision Trees**
- **Stochastic Gradient Descent (SGD)**
- **Multilayer Perceptron (MLP)**

## Evaluation
Models were evaluated using standard metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

Performance metrics for each model were compared to identify the best approach for spam classification.

## Usage
1. Open the `email-spam-classification.ipynb` notebook in Jupyter or Google Colab.
2. Run each cell sequentially to preprocess data, train models, and evaluate their performance.
3. Modify the code as needed to test additional datasets or configurations.

## Conclusion
This project demonstrates the application of multiple machine learning models for spam email classification. The preprocessing pipeline and model evaluation ensure a robust framework for handling real-world datasets. The approach can be extended to other text classification tasks with minimal modifications.

Feel free to use and adapt this project for your own use cases. Contributions and feedback are welcome!
