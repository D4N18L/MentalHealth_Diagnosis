import pandas as pd
import numpy as np
import itertools
from collections import Counter
from typing import List, Dict, Tuple, Any, Union

# Ensure the necessary libraries are installed
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
import joblib
import gensim.downloader as api
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


class DataLoader:
    """
    Load and restructure mental health patient data from an Excel file.
    """

    def __init__(self, dataset: str):
        self.dataset = dataset
        self.data = self.load_data()

    def load_data(self) -> pd.DataFrame:
        """
        Load dataset from an Excel file using pandas.

        Args:
            None

        Returns:
            pd.DataFrame: Loaded data as a pandas DataFrame.
        """
        return pd.read_excel(self.dataset, engine='openpyxl')

    def split_into_columns(self) -> pd.DataFrame:
        """
        Split the dataset from its initial one-cell format into three distinct columns.

        Args:
            None

        Returns:
            pd.DataFrame: Data with 'patient_id', 'concept', and 'extraction' columns.
        """
        self.data = self.data['patient_id\tconcept_1\textraction_1'].str.split('\t', expand=True)
        self.data.columns = ['patient_id', 'concept', 'extraction']
        return self.data

    def pivot_data(self) -> pd.DataFrame:
        """
            Pivot the data to create patient profiles with 'patient_id' as index and 'concept' values as columns.

            Args:
                None

            Returns:
                pd.DataFrame: Pivoted DataFrame representing patient profiles.
            """
        pivoted_data = self.data.pivot_table(index='patient_id', columns='concept',
                                             values='extraction', aggfunc=lambda x: list(x))

        self.data = pivoted_data.reset_index()  # Reset the index to make 'patient_id' a column

        for col in ['medication', 'patient_experience', 'diagnosis']:
            if col in self.data.columns:
                self.data[col] = self.data[col].apply(lambda x: x if isinstance(x, list) else [
                    'No_' + col])  # Handle missing values by replacing them with 'No_' + column name

        return self

    def get_patient_profiles(self) -> pd.DataFrame:
        """
        Get patient profiles by loading, splitting, and pivoting the data.

        Args:
            None

        Returns:
            pd.DataFrame: Patient profiles with 'patient_id' as index and 'concept' values as columns.
        """
        self.split_into_columns()
        self.pivot_data()
        return self.data


class DataCleaner:
    """
    Clean and explore the patient data for insights and preparation for modeling.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def check_missing_values(self) -> pd.Series:
        """
        Check and count missing values in the dataset.

        Args:
            None

        Returns:
            pd.Series: Counts of missing values in each column.
        """
        return self.data.isnull().sum()

    def check_data_size(self) -> Tuple[int, int]:
        """
        Get the size of the dataset.

        Args:
            None

        Returns:
            Tuple[int, int]: The dimensions of the dataset (rows, columns).
        """
        return self.data.shape

    def check_data_types(self) -> pd.Series:
        """
        Get the data types of columns in the dataset.

        Args:
            None

        Returns:
            pd.Series: Data types of each column.
        """
        return self.data.dtypes

    def data_info(self) -> Dict[str, Any]:
        """
        Explore the data to find common and unique values in each 'experience' related column.

        Args:
            None

        Returns:
            Dict[str, Any]: Information about the most and least common values, number of unique values, and all unique values in each column.
        """
        info = {}
        columns_to_analyze = ['patient_experience', 'medication', 'diagnosis']

        for column in columns_to_analyze:
            # Flatten the nested lists in the specified column
            column_items = list(itertools.chain(*self.data[column]))

            # Count the occurrences of each item
            item_counts = Counter(column_items)

            # Find the most common and least common item
            most_common_value = item_counts.most_common(1)[0] if item_counts else ('None', 0)
            least_common_value = item_counts.most_common()[-1] if item_counts else ('None', 0)

            # Get unique values
            unique_values = set(item_counts)

            info[f'most_common_{column}'] = most_common_value
            info[f'least_common_{column}'] = least_common_value
            info[f'Number of unique values for {column} '] = len(unique_values)
            info[f"Unique Values: {column}"] = unique_values

        return info


class FeatureEngineer:
    """
    This class performs separate feature engineering tasks on the patient experiences and medications and then
    combines the features into a single feature set to be put through the machine learning modelling process.
    """

    def __init__(self, dataset):
        self.data = dataset
        self.one_hot_encoder = OneHotEncoder()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        self.word_vectors = api.load(
            'glove-wiki-gigaword-100')  # Load pre-trained word vectors for patient experiences data

        # Load pre-trained BioBERT tokenizer and model for medication data
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.1_pubmed")
        self.model = AutoModel.from_pretrained("monologg/biobert_v1.1_pubmed")

    def preprocess_text(self, text):
        """
        Checks to see if the text is a list or a string and then preprocesses it accordingly.
        """
        if isinstance(text, list):  # Check if the text is a list
            return [self._process_single_text(t) for t in text]  # Preprocess each text in the list
        else:
            return self._process_single_text(text)  # Preprocess a single text

    def _process_single_text(self, single_text):
        """
        Preprocess a single text by tokenizing, lemmatizing, and removing stop words.
        """
        tokens = word_tokenize(single_text.lower())  # Tokenize and lowercase
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if
                  token.isalpha() and token not in self.stop_words]  # Lemmatize and remove stop words if the token is alphabetic and not a stop word
        return tokens

    def vectorize_text(self, tokens):
        """
        Vectorize a list of tokens using pre-trained word vectors.
        """
        vectors = [self.word_vectors[word] for word in tokens if
                   word in self.word_vectors]  # Get the word vectors for each token if the token is in the word vectors
        return np.mean(vectors, axis=0) if vectors else np.zeros(
            self.word_vectors.vector_size)  # Calculate the mean vector of all tokens if there are tokens, otherwise return a vector of zeros

    def vectorize_medication(self, medications):
        """
        Vectorize medications data using BioBERT embeddings.

        BioBert model is trained on  medical data and are more suited for this task than the pre-trained word vectors used for the patient experiences.
        """

        # Handle "No_medication" case
        if medications == "No_medication":  # Check if the medication is "No_medication"
            return np.zeros(self.model.config.hidden_size)  # Return a vector of zeros

        # Check if medications are in a list
        if not isinstance(medications, list):
            medications = [medications]

        # Tokenize and preprocess each medication, then generate BioBERT embeddings
        embeddings = []
        for med in medications:
            # Handle multiple medications
            if isinstance(med, list):
                med_text = ' '.join(self.preprocess_text(' '.join(med)))  # Preprocess and join multiple medications
            else:
                med_text = ' '.join(self.preprocess_text(med))  # Preprocess a single medication

            inputs = self.tokenizer(med_text, return_tensors="pt", padding=True, truncation=True,
                                    max_length=512)  # Tokenize the medication text
            outputs = self.model(**inputs)  # Generate BioBERT embeddings for the medication text
            med_embeddings = outputs.last_hidden_state
            embeddings.append(
                np.mean(med_embeddings.detach().numpy(), axis=1))  # Calculate the mean embedding of each medication

        # Calculate the mean embedding of all medications
        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            return np.zeros(self.model.config.hidden_size)

    def aggregate_experiences(self, experiences):
        """
        Aggregate patient experiences by taking the mean of the embeddings of each experience and returning a single embedding.
        This handles the case where there are multiple experiences for a single patient, a single experience, and no experiences at all.
        """
        # Check if there are multiple experiences
        if len(experiences) > 1:
            all_embeddings = []
            for exp in experiences:
                processed_text = self.preprocess_text(exp)  # Process each experience
                exp_embedding = self.vectorize_text(processed_text)  # Get embedding
                all_embeddings.append(exp_embedding)
            return np.mean(all_embeddings, axis=0) if all_embeddings else np.zeros(self.word_vectors.vector_size)

        # Check if there is a single experience
        elif len(experiences) == 1:
            processed_text = self.preprocess_text(experiences[0])  # Process the single experience
            return self.vectorize_text(processed_text)

        # Handle case with no experiences
        else:
            return np.zeros(self.word_vectors.vector_size)

    def encode_medications(self):
        """
        Encode medications and return a feature set. This also handles the case where there are multiple medications for a single patient, a single medication, and no medications at all.
        """
        medication_embeddings = self.data['medication'].apply(self.vectorize_medication)  # Vectorize medications
        return np.vstack(medication_embeddings)  # Stack the medication embeddings into a feature set

    def combine_features(self):
        """
        Combine the patient experience and medication features into a single feature set for modelling.
        """
        patient_experience_vectors = [self.aggregate_experiences(exp) for exp in self.data['patient_experience']]
        medication_features = self.encode_medications()  # Encode medications
        combined_features = np.hstack(
            (patient_experience_vectors, medication_features))  # Combine patient experience and medication features
        return combined_features  # Return the combined feature set


class DataSplitter:
    """
    Splits the dataset into training and testing sets.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Initializes the DataSplitter with features and labels.

        Args:
            features (np.ndarray): The feature set for splitting.
            labels (np.ndarray): The label set for splitting.
        """
        self.features = features
        self.labels = labels

    def split_data(self, test_size: float = 0.2, random_state: int = None) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits the data into training and testing sets.

        Args:
            test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
            random_state (int, optional): Controls the shuffling applied to the data before applying the split. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The split data (X_train, X_test, y_train, y_test).
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size=test_size, random_state=random_state)  # Split the data
        return X_train, X_test, y_train, y_test  # Return the split data containing features and labels


class Modelling:
    """
    Handles training, evaluating, and hyperparameter tuning of machine learning models.
    """

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        """
        Initializes the Modelling class with training and testing data.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_test : Testing features.
            y_test : Testing labels.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.models = {
            'random_forest': RandomForestClassifier()
        }

    def train_model(self, model_name: str):
        """
        Trains a specified model on the training data.

        Args:
            model_name (str): The name of the model to train.

        Raises:
            ValueError: If the specified model name is not found in the models dictionary.
        """
        if model_name in self.models:  # Check if the model name is in the models dictionary
            self.models[model_name].fit(self.X_train, self.y_train)  # Train the model
        else:
            raise ValueError(f"Model {model_name} not found.")  # Raise an error if the model name is not found

    def evaluate_model(self, model_name: str):
        """
        Evaluates a specified model on the testing data.

        Args:
            model_name (str): The name of the model to evaluate.

        Raises:
            ValueError: If the specified model name is not found in the models dictionary.
        """
        if model_name in self.models:  # Check if the model name is in the models dictionary
            predictions = self.models[model_name].predict(self.X_test)  # Make predictions
            print(f"Evaluation Report for {model_name}:")  # Print the evaluation report
            print(classification_report(self.y_test, predictions, zero_division=1))  # Print the classification report
            print("Accuracy:", accuracy_score(self.y_test, predictions))  # Print the accuracy score
        else:
            raise ValueError(f"Model {model_name} not found.")  # Raise an error if the model name is not found

    def tune_hyperparameters(self, model_name: str, param_grid: dict, cv: int = 5):
        """
        Tunes the hyperparameters of a specified model using GridSearchCV.

        Args:
            model_name (str): The name of the model to tune.
            param_grid (dict): The parameter grid to explore.
            cv (int, optional): Number of cross-validation folds. Defaults to 5.

        Raises:
            ValueError: If the specified model name is not found in the models dictionary.
        """
        if model_name in self.models:  # Check if the model name is in the models dictionary
            grid_search = GridSearchCV(self.models[model_name], param_grid, cv=cv,
                                       scoring='accuracy')  # Initialize the grid search
            grid_search.fit(self.X_train, self.y_train)  # Fit the grid search to the training data
            self.models[
                model_name] = grid_search.best_estimator_  # Set the model to the best estimator found by the grid search
            print(f"Best Parameters for {model_name}: {grid_search.best_params_}")  # Print the best parameters
        else:
            raise ValueError(f"Model {model_name} not found.")  # Raise an error if the model name is not found

    def cross_validate_model(self, model_name: str, cv: int = 5):
        """
        Performs cross-validation on a specified model.

        Args:
            model_name (str): The name of the model to cross-validate.
            cv (int, optional): Number of cross-validation folds. Defaults to 5.

        Raises:
            ValueError: If the specified model name is not found in the models dictionary.
        """
        if model_name in self.models:  # Check if the model name is in the models dictionary
            scores = cross_val_score(self.models[model_name], self.X_train, self.y_train, cv=cv,
                                     scoring='accuracy')  # Perform cross-validation
            print(f"Cross-Validation Scores for {model_name}: {scores}")  # Print the cross-validation scores
            print(f"Average Score: {np.mean(scores)}")  # Print the average score
        else:
            raise ValueError(f"Model {model_name} not found.")  # Raise an error if the model name is not found

    def retrain_on_full_data(self, model_name: str):
        """
        Retrains a specified model on the full dataset (training + testing data).

        Args:
            model_name (str): The name of the model to retrain.

        Raises:
            ValueError: If the specified model name is not found in the models dictionary.
        """
        if model_name in self.models:  # Check if the model name is in the models dictionary
            X_full = np.vstack((self.X_train, self.X_test))  # Stack the training and testing features
            y_full = np.vstack((self.y_train, self.y_test))  # Stack the training and testing labels#
            self.models[model_name].fit(X_full, y_full)  # Retrain the model on the full dataset
        else:
            raise ValueError(f"Model {model_name} not found.")  # Raise an error if the model name is not found
