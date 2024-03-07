# Mental Health Diagnosis Model

This repository contains code for a machine learning model designed to predict mental health diagnoses based on patient data. 
The model utilizes various techniques for data loading, cleaning, feature engineering, and machine learning modelling.

 The repo shows the utilization external APIs, NLP word embeddings, the Random Forest algorithm, and Streamlit for building the mental health diagnosis prediction system

## DataLoader

The `DataLoader` class is responsible for loading and restructuring mental health patient data from an Excel file. It provides the following functionalities:

- **load_data():** Loads dataset from an Excel file using pandas.
- **split_into_columns():** Splits the dataset from its initial one-cell format into three distinct columns: 'patient_id', 'concept', and 'extraction'.
- **pivot_data():** Pivots the data to create patient profiles with 'patient_id' as index and 'concept' values as columns.

## DataCleaner

The `DataCleaner` class is used to clean and explore the patient data for insights and preparation for modeling. It offers the following functionalities:

- **check_missing_values():** Checks and counts missing values in the dataset.
- **check_data_size():** Retrieves the dimensions of the dataset (rows, columns).
- **check_data_types():** Obtains the data types of columns in the dataset.
- **data_info():** Explores the data to find common and unique values in each 'experience' related column.

## FeatureEngineer

The `FeatureEngineer` class performs feature engineering on the patient data to create features for modeling.
It utilizes external APIs and pre-trained models for enhancing features. It provides the following functionalities:

- **preprocess_text():** Preprocesses text data by tokenizing, lemmatizing, and removing stop words.
- **vectorize_text():** Vectorizes text using pre-trained word embeddings (GloVe).
- **vectorize_medication():** Encodes medications using BioBERT embeddings, a pre-trained model specifically designed for biomedical text.
- **aggregate_experiences():** Aggregates patient experiences by taking the mean of the embeddings of each experience.
- **encode_medications():** Encodes medications and returns a feature set.
- **combine_features():** Combines patient experience and medication features into a single feature set for modeling.

## DataSplitter

The `DataSplitter` class splits the dataset into training and testing sets. It offers the following functionalities:

- **split_data():** Splits the data into training and testing sets using the `train_test_split` function from scikit-learn.

## Modelling

The `Modelling` class handles training, evaluating, tuning hyperparameters, and cross-validating machine learning models. It provides functionalities for:

- **train_model():** Trains a specified model on the training data.
- **evaluate_model():** Evaluates a specified model on the testing data.
- **tune_hyperparameters():** Tunes the hyperparameters of a specified model using GridSearchCV.
- **cross_validate_model():** Performs cross-validation on a specified model.
- **retrain_on_full_data():** Retrains a specified model on the full dataset (training + testing data).
