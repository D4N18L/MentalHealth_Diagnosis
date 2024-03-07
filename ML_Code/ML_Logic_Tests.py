import unittest

import numpy as np
import pandas as pd
from MentalHealth_ML.MentalHealth_ML import DataLoader, DataCleaner, DataSplitter


class TestDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = DataLoader('Patient_Tests_Sample.xlsx')

    def test_load_data(self):
        data = self.loader.load_data()
        self.assertIsInstance(data, pd.DataFrame, "Data should be loaded as a pandas DataFrame")

    def test_split_into_columns(self):
        # Test column splitting, ensuring a fresh DataLoader instance is used for each test
        loader = DataLoader('Patient_Tests_Sample.xlsx')
        split_data = loader.split_into_columns()
        expected_columns = ['patient_id', 'concept', 'extraction']
        self.assertTrue(all(column in split_data.columns for column in expected_columns),
                        "Data should be split into expected columns")

    def test_pivot_data(self):

        # Test data pivoting to create patient profiles and check if the 'patient_id' column is present within the pivoted data
        loader = DataLoader('Patient_Tests_Sample.xlsx')
        loader.split_into_columns()  # Split into columns before pivoting
        pivoted_data = loader.pivot_data().data  # Access the updated data attribute after pivoting
        self.assertIn('patient_id', pivoted_data.columns, "Pivoted data should contain 'patient_id' as a column")


class TestDataCleaner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        loader = DataLoader('Patient_Tests_Sample.xlsx')
        cls.test_data = loader.get_patient_profiles()

    def test_check_missing_values(self):

        # Test that the method correctly counts missing values
        cleaner = DataCleaner(self.test_data)
        missing_values = cleaner.check_missing_values()
        self.assertIsInstance(missing_values, pd.Series, "Should return a Series")
        self.assertTrue(all(isinstance(value, int) for value in missing_values), "Series should contain integers")

    def test_check_data_size(self):
        # Test that the method correctly returns the size of the dataset
        cleaner = DataCleaner(self.test_data)
        data_size = cleaner.check_data_size()
        self.assertIsInstance(data_size, tuple, "Should return a tuple")
        self.assertEqual(len(data_size), 2, "Tuple should contain two elements")
        self.assertTrue(all(isinstance(value, int) for value in data_size), "Tuple should contain integers")

    def test_check_data_types(self):
        # Test that the method correctly returns the data types of the dataset
        cleaner = DataCleaner(self.test_data)
        data_types = cleaner.check_data_types()
        self.assertIsInstance(data_types, pd.Series, "Should return a Series")

        # Check if each value in the Series is an instance of numpy.dtype
        self.assertTrue(all(isinstance(value, np.dtype) for value in data_types), "Series should contain dtype objects")

    def test_data_info(self):
        # Test that the method correctly returns the data info
        cleaner = DataCleaner(self.test_data)
        data_info = cleaner.data_info()

        # Define the columns to be analyzed
        columns_to_analyze = ['patient_experience', 'medication', 'diagnosis']

        # Loop through each column to check for expected keys and their data types
        for column in columns_to_analyze:
            most_common_key = f'most_common_{column}'
            least_common_key = f'least_common_{column}'
            num_unique_values_key = f'Number of unique values for {column} '
            unique_values_key = f'Unique Values: {column}'

            self.assertIn(most_common_key, data_info)
            self.assertIn(least_common_key, data_info)
            self.assertIn(num_unique_values_key, data_info)
            self.assertIn(unique_values_key, data_info)

            # Check types
            self.assertIsInstance(data_info[most_common_key], tuple, f"{most_common_key} should be a tuple.")
            self.assertIsInstance(data_info[least_common_key], tuple, f"{least_common_key} should be a tuple.")
            self.assertIsInstance(data_info[num_unique_values_key], int, f"{num_unique_values_key} should be an int.")
            self.assertIsInstance(data_info[unique_values_key], set, f"{unique_values_key} should be a set.")

            # Check the content of the tuples for most and least common values
            self.assertTrue(len(data_info[most_common_key]) == 2, f"{most_common_key} tuple should have two elements.")
            self.assertTrue(len(data_info[least_common_key]) == 2,
                            f"{least_common_key} tuple should have two elements.")


class TestDataSplitter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the test data using DataLoader
        loader = DataLoader('Patient_Tests_Sample.xlsx')
        test_data = loader.get_patient_profiles()

        features = test_data[['patient_experience', 'medication']].values
        labels = test_data['diagnosis'].values

        # Instantiate DataSplitter with the loaded data
        cls.data_splitter = DataSplitter(features, labels)

    def test_split_data(self):

        X_train, X_test, y_train, y_test = self.data_splitter.split_data()

        # Check if the split sizes are correct
        self.assertEqual(X_train.shape[0], 448)
        self.assertEqual(X_test.shape[0], 112)
        self.assertEqual(y_train.shape[0], 448)
        self.assertEqual(y_test.shape[0], 112)

        # Check if the total number of samples is preserved
        self.assertEqual(X_train.shape[0] + X_test.shape[0], 560)
        self.assertEqual(y_train.shape[0] + y_test.shape[0], 560)

        # Test the split_data method with specified test_size
        X_train, X_test, y_train, y_test = self.data_splitter.split_data(test_size=0.3)

        # Check if the split sizes are correct
        self.assertEqual(X_train.shape[0], 392)
        self.assertEqual(X_test.shape[0], 168)
        self.assertEqual(y_train.shape[0], 392)
        self.assertEqual(y_test.shape[0], 168)

        # Check if the total number of samples is preserved
        self.assertEqual(X_train.shape[0] + X_test.shape[0], 560)
        self.assertEqual(y_train.shape[0] + y_test.shape[0], 560)

        # Test the split_data method with specified random_state
        X_train1, X_test1, y_train1, y_test1 = self.data_splitter.split_data(random_state=42)
        X_train2, X_test2, y_train2, y_test2 = self.data_splitter.split_data(random_state=42)

        # Check if the splits are consistent with the same random state
        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(X_test1, X_test2)
        np.testing.assert_array_equal(y_train1, y_train2)
        np.testing.assert_array_equal(y_test1, y_test2)


if __name__ == '__main__':
    unittest.main()
