import pandas as pd


def create_sample_dataset(original_file_path, sample_file_path, sample_fraction=0.1):
    """
    Creates a sample dataset from the original dataset, ensuring that the first row (headers) is included.

    Parameters:
    - original_file_path: The file path of the original dataset.
    - sample_file_path: The file path where the sampled dataset will be saved.
    - sample_fraction: The fraction of the original dataset to be used in the sample. Default is 0.1 (10%).
    """

    # Load the original dataset
    original_data = pd.read_excel(original_file_path, engine='openpyxl')

    # Ensure the first row is included by adding it to the sampled data
    first_row = original_data.iloc[0:1]
    sampled_data = original_data.sample(frac=sample_fraction,
                                        random_state=42)  # Use a fixed random state for reproducibility

    # Combine the first row with the sampled data
    sample_data_with_header = pd.concat([first_row, sampled_data], ignore_index=True)

    # Save the sampled dataset with the header included
    sample_data_with_header.to_excel(sample_file_path, index=False)


if __name__ == "__main__":
    original_file_path = '../ML_Datasets/Patient_Data.xlsx'
    sample_file_path = '../ML_Datasets/Patient_Tests_Sample.xlsx'

    # Create the sampled dataset
    create_sample_dataset(original_file_path, sample_file_path)
