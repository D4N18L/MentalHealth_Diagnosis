import joblib
from ML_Code.MentalHealth_ML import DataLoader,DataCleaner,DataSplitter,FeatureEngineer,Modelling

from sklearn.preprocessing import MultiLabelBinarizer


def main(patient_file: str):
    # Load and restructure data
    load_and_restructure = DataLoader(patient_file)
    restructured_data = load_and_restructure.get_patient_profiles()

    # Clean and explore data
    clean_and_explore = DataCleaner(restructured_data)
    missing_values = clean_and_explore.check_missing_values()
    data_size = clean_and_explore.check_data_size()
    data_types = clean_and_explore.check_data_types()
    data_info = clean_and_explore.data_info()

    print("Dataset Shape:", data_size)
    print("Missing Values:", missing_values)
    print("Data Types:", data_types)
    print("Data Information:", data_info)

    # Feature Engineering
    labels = restructured_data['diagnosis']  # Assuming 'diagnosis' is the label column
    features = restructured_data.drop(['diagnosis', 'patient_id'], axis=1)  # Adjust as needed

    feature_engineer = FeatureEngineer(features)
    combined_features = feature_engineer.combine_features()

    # Label Binarization
    mlb = MultiLabelBinarizer()
    encoded_labels = mlb.fit_transform(labels)
    joblib.dump(mlb, '../Label_Binarizer/label_binarizer.pkl')  # Save the fitted MultiLabelBinarizer

    # Data Splitting
    data_splitter = DataSplitter(combined_features, encoded_labels)
    X_train, X_test, y_train, y_test = data_splitter.split_data(test_size=0.2)

    # Model Training and Evaluation
    model_selector = Modelling(X_train, y_train, X_test, y_test)

    # Tune hyperparameters for Random Forest
    param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
    model_selector.tune_hyperparameters('random_forest', param_grid_rf)

    # Train and evaluate the Random Forest model
    model_selector.train_model('random_forest')
    model_selector.evaluate_model('random_forest')

    # Retrain on full data and evaluate
    model_selector.retrain_on_full_data('random_forest')
    model_selector.evaluate_model('random_forest')

    # Save the Random Forest model
    joblib.dump(model_selector.models['random_forest'], '../Model_Instance/random_forest_model.pkl')


if __name__ == "__main__":
    patient_file = '../ML_Datasets/Patient_Data.xlsx'
    main(patient_file)
