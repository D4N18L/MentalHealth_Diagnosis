import os
import subprocess
import numpy as np
import streamlit as st
import joblib
from ML_Code.MentalHealth_ML import FeatureEngineer

@st.cache_data
def get_feature_engineer():
    return FeatureEngineer(None)


@st.cache_data
def load_model():
    return joblib.load('../Model_Instance/random_forest_model.pkl')


@st.cache_data
def load_label_binarizer():
    return joblib.load('../Label_Binarizer/label_binarizer.pkl')


model = load_model()
mlb = load_label_binarizer()
feature_engineer = get_feature_engineer()

def main():
    st.title('Mental Health Diagnosis Prediction')

    # Text input for patient experiences
    patient_experiences = st.text_input("Enter patient experiences")

    # Process patient experiences
    processed_experience = feature_engineer.aggregate_experiences([patient_experiences])
    processed_experience = np.atleast_2d(processed_experience)

    # Toggle for deciding whether to add medication or not
    add_medication = st.checkbox("Add medication?")

    # Conditional input field for medications
    if add_medication:
        medications = st.text_input("Enter medications")
    else:
        medications = "No_medication"

    # Process medications
    processed_medication = feature_engineer.vectorize_medication([medications])
    processed_medication = np.atleast_2d(processed_medication)

    # Prediction and display logic
    if st.button('Predict Diagnosis'):
        # Combine features using np.concatenate
        combined_features = np.concatenate((processed_experience, processed_medication), axis=1)

        # Make predictions using the trained model
        prediction = model.predict(combined_features)
        predicted_labels_tuple = mlb.inverse_transform(prediction)
        predicted_labels = predicted_labels_tuple[0] if predicted_labels_tuple else ('No diagnosis',)

        # Display predictions
        st.write("Predicted Diagnosis:", predicted_labels)


def run_streamlit():
    # Check if the environment variable is set
    if os.getenv('STREAMLIT_RUNNING') is None:

        # Set the environment variable for this process and its children
        os.environ['STREAMLIT_RUNNING'] = '1'
        # Start the Streamlit process
        subprocess.run(['streamlit', 'run', __file__])
    else:
        # The environment variable is set, meaning this is the subprocess
        # Define the Streamlit app UI
        main()


if __name__ == '__main__':
    run_streamlit()
