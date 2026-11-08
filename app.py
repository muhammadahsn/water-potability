from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).parent / "water_potability_model.pkl"
col_list = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]
default_values = {
    "ph": 7.0,
    "Hardness": 204.8905,
    "Solids": 20791.3190,
    "Chloramines": 7.3002,
    "Sulfate": 368.5164,
    "Conductivity": 564.3087,
    "Organic_carbon": 10.3798,
    "Trihalomethanes": 86.9910,
    "Turbidity": 2.9631,
}


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Run the Classification notebook to create it."
        )
    return joblib.load(MODEL_PATH)


st.title("Water Potability Prediction")
st.write("Enter the water-quality measurements to predict whether the water is potable.")

with st.form("water_quality_form"):
    st.subheader("Water quality measurements")
    input_columns = st.columns(3)
    input_values = {}

    for index, column_name in enumerate(col_list):
        with input_columns[index % len(input_columns)]:
            input_values[column_name] = st.number_input(
                column_name.replace("_", " ").title(),
                min_value=0.0,
                max_value=14.0 if column_name == "ph" else None,
                value=default_values[column_name],
                step=0.1,
                format="%.4f",
            )

    submitted = st.form_submit_button("Predict potability")

if submitted:
    model = load_model()
    input_frame = pd.DataFrame([input_values], columns=col_list)
    prediction = model.predict(input_frame)[0]

    if prediction == 1:
        st.success("Prediction: Potable")
    else:
        st.warning("Prediction: Not potable")
