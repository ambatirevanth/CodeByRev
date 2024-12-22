import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Function to remove outliers
def remove_outliers(data, z_thresh=3):
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    z_scores = np.abs((data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std())
    filtered_entries = (z_scores < z_thresh).all(axis=1)
    return data[filtered_entries]

# Preprocess the dataset
def preprocess_data(data):
    st.subheader("Missing Values Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    msno.matrix(data, ax=ax, fontsize=12)
    st.pyplot(fig)

    data = data.fillna(data.median(numeric_only=True))
    data = data.fillna(method="ffill").fillna(method="bfill")

    # Encode categorical features
    label_encoders = {}
    for column in data.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].astype(str))
        label_encoders[column] = le

    # Prepare the features and target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Create preprocessed DataFrame
    preprocessed_data = pd.DataFrame(X, columns=X.columns)
    preprocessed_data["Target"] = y.reset_index(drop=True)

    return X, y, preprocessed_data, label_encoders

# Display confusion matrix
def plot_confusion_matrix(y_test, y_pred, class_names):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    st.pyplot(fig)

# Train and evaluate Random Forest
def train_and_evaluate_rf(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "y_test": y_test,
        "y_pred": y_pred,
    }
    return metrics

# Visualize outliers using boxplots
def visualize_outliers(data, title):
    st.subheader(title)
    for column in data.select_dtypes(include=[np.number]).columns:
        fig, ax = plt.subplots()
        sns.boxplot(data[column], ax=ax)
        ax.set_title(f"Boxplot of {column}")
        plt.tight_layout()
        st.pyplot(fig)

# Main Streamlit app
def main():
    st.sidebar.title("Navigation")
    options = st.sidebar.radio(
        "Choose a section:",
        [
            "Introduction",
            "Feature Descriptions",
            "Before Preprocessed Dataset",
            "After Preprocessed Dataset",
            "Outlier Analysis",
            "Random Forest Training and Results",
            "Result Analysis",
        ],
        key="section_selector",
    )

    st.title("Enhanced Streamlit App: Outlier Analysis, Random Forest, and Results")
    st.write("Analyze before-preprocessed and after-preprocessed datasets separately, including outlier analysis, and evaluate results.")

    uploaded_before = st.file_uploader("Upload your before-preprocessed dataset (CSV)", type=["csv"], key="before_file")
    uploaded_after = st.file_uploader("Upload your after-preprocessed dataset (CSV)", type=["csv"], key="after_file")

    # Feature descriptions dictionary
    feature_descriptions = {
        "User_ID": "Unique identifier for each user, ensuring individuality in the dataset.",
        "Age": "Age of the user in years, representing their demographic group (18-65 years).",
        "Gender": "Gender of the user, categorized as 'Male' or 'Female'.",
        "Height_cm": "User's height in centimeters, a factor contributing to BMI and overall health metrics.",
        "Weight_kg": "User's weight in kilograms, essential for calculating BMI and monitoring health trends.",
        "BMI": "Body Mass Index, providing a health assessment based on the user's height and weight.",
        "Workout_Type": "Preferred type of workout, categorized as 'Cardio', 'Strength', or 'Yoga', indicating fitness habits.",
        "Workout_Duration_min": "Duration of the workout session in minutes, showing exercise commitment.",
        "Heart_Rate_avg": "Average heart rate during workout sessions, measured in beats per minute (BPM), reflecting cardiovascular effort.",
        "Intensity_Level": "Intensity level of the workout, classified as 'Low', 'Medium', or 'High', indicating exercise vigor.",
        "Diet_Type": "Dietary preference of the user, categorized as 'Vegan', 'Vegetarian', or 'Mixed'.",
        "Sleep_Hours": "Average number of hours the user sleeps per day, ranging from 4.0 to 9.0 hours, a measure of rest and recovery.",
        "Hydration_Liters": "Daily water intake of the user in liters, reflecting hydration levels and health awareness.",
        "Workout_Frequency": "Number of days per week the user engages in physical activity, ranging from 1 to 7 days.",
        "Calories_Burned": "Estimated level of calories burned during workouts, classified as 'Low', 'Medium', or 'High', based on intensity and duration.",
    }

    if options == "Introduction":
        st.subheader("Introduction")
        st.write(
            "This app allows you to analyze two datasets: one before preprocessing and one after preprocessing. "
            "It includes outlier analysis, Random Forest training, and result comparison with detailed metrics and visualizations."
        )

    elif options == "Feature Descriptions":
        st.subheader("Feature Descriptions")
        st.write("Below is the description of each feature in the dataset:")
        for feature, description in feature_descriptions.items():
            st.markdown(f"{feature}:** {description}**")

    elif options == "Before Preprocessed Dataset":
        st.subheader("Before Preprocessed Dataset")
        if uploaded_before:
            data_before = pd.read_csv(uploaded_before)
            st.write(data_before.head())

            st.subheader("Feature Distributions (Before Preprocessing)")
            for column in data_before.columns[:-1]:  # Exclude target column
                fig, ax = plt.subplots()
                sns.histplot(data_before[column], kde=True, bins=20, ax=ax)
                ax.set_title(f"Distribution of {column} (Before)")
                plt.tight_layout()
                st.pyplot(fig)

            st.subheader("Outlier Removal (Before Dataset)")
            filtered_before = remove_outliers(data_before)
            st.write("Dataset after removing outliers:")
            st.write(filtered_before.head())

    elif options == "After Preprocessed Dataset":
        st.subheader("After Preprocessed Dataset")
        if uploaded_after:
            data_after = pd.read_csv(uploaded_after)
            _, _, preprocessed_data, _ = preprocess_data(data_after)
            st.write(preprocessed_data.head())

            st.subheader("Feature Distributions (After Preprocessing)")
            for column in preprocessed_data.columns[:-1]:  # Exclude target column
                fig, ax = plt.subplots()
                sns.histplot(preprocessed_data[column], kde=True, bins=20, ax=ax)
                ax.set_title(f"Distribution of {column} (After)")
                plt.tight_layout()
                st.pyplot(fig)

            st.subheader("Outlier Removal (After Dataset)")
            filtered_after = remove_outliers(data_after)
            st.write("Dataset after removing outliers:")
            st.write(filtered_after.head())

    elif options == "Outlier Analysis":
        st.subheader("Outlier Analysis")
        if uploaded_before:
            data_before = pd.read_csv(uploaded_before)
            st.write("Outliers in Before-Preprocessed Dataset:")
            visualize_outliers(data_before, "Before-Preprocessed Dataset Outliers")

        if uploaded_after:
            data_after = pd.read_csv(uploaded_after)
            st.write("Outliers in After-Preprocessed Dataset:")
            visualize_outliers(data_after, "After-Preprocessed Dataset Outliers")

    elif options == "Random Forest Training and Results":
        st.subheader("Random Forest Training and Results")
        if uploaded_before:
            st.subheader("Before Dataset Random Forest Training")
            data_before = pd.read_csv(uploaded_before)
            X_before, y_before, _, _ = preprocess_data(data_before)
            metrics_before = train_and_evaluate_rf(X_before, y_before)
            st.write("### Metrics for Before Dataset")
            st.write(f"Accuracy: {metrics_before['accuracy'] * 100:.2f}%")
            st.write(f"F1 Score: {metrics_before['f1']:.2f}")
            st.write("Classification Report (Before Dataset):")
            st.dataframe(pd.DataFrame(metrics_before["classification_report"]).transpose())

        if uploaded_after:
            st.subheader("After Dataset Random Forest Training")
            data_after = pd.read_csv(uploaded_after)
            X_after, y_after, _, _ = preprocess_data(data_after)
            metrics_after = train_and_evaluate_rf(X_after, y_after)
            st.write("### Metrics for After Dataset")
            st.write(f"Accuracy: {metrics_after['accuracy'] * 100:.2f}%")
            st.write(f"F1 Score: {metrics_after['f1']:.2f}")
            st.write("Classification Report (After Dataset):")
            st.dataframe(pd.DataFrame(metrics_after["classification_report"]).transpose())

    elif options == "Result Analysis":
        st.subheader("Result Analysis")
        if uploaded_before:
            st.write("Insights from Before Dataset")
            data_before = pd.read_csv(uploaded_before)
            st.write(data_before.describe())

        if uploaded_after:
            st.write("Insights from After Dataset")
            data_after = pd.read_csv(uploaded_after)
            st.write(data_after.describe())


if __name__ == "__main__":
    main()
