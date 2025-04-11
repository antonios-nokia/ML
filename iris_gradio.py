import gradio as gr
import joblib
import numpy as np
import pandas as pd

# Load the saved model, scaler, and label encoder
model = joblib.load("iris_logit_model.pkl")
scaler = joblib.load("iris_scaler.pkl")
encoder = joblib.load("iris_label_encoder.pkl")

# Define a function for prediction
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    # Create a NumPy array from input
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Scale the input using the saved scaler
    input_scaled = scaler.transform(input_data)

    # Predict using the trained model
    prediction = model.predict(input_scaled)

    # Convert prediction from encoded label to species name
    species_name = encoder.inverse_transform(prediction)[0]

    return f"Predicted Species: {species_name}"

# Function to handle CSV file uploads
def predict_from_csv(file):
    # Read CSV file
    df = pd.read_csv(file)
    
    # Ensure the CSV file contains only feature columns
    df_scaled = scaler.transform(df)

    # Predict using the trained model
    predictions = model.predict(df_scaled)

    # Convert predictions to species names
    predicted_species = encoder.inverse_transform(predictions)

    # Add predictions as a new column
    df["Predicted Species"] = predicted_species

    return df

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🌸 Iris Species Predictor")

    with gr.Tab("Single Prediction"):
        gr.Markdown("Enter feature values to predict the Iris species.")
        sepal_length = gr.Number(label="Sepal Length")
        sepal_width = gr.Number(label="Sepal Width")
        petal_length = gr.Number(label="Petal Length")
        petal_width = gr.Number(label="Petal Width")
        predict_btn = gr.Button("Predict")
        output_text = gr.Textbox(label="Prediction")

        predict_btn.click(predict_species, inputs=[sepal_length, sepal_width, petal_length, petal_width], outputs=output_text)

    with gr.Tab("Batch Prediction (CSV)"):
        gr.Markdown("Upload a CSV file with Iris flower measurements for bulk prediction.")
        file_input = gr.File(label="Upload CSV")
        csv_output = gr.Dataframe()
        csv_btn = gr.Button("Predict from CSV")

        csv_btn.click(predict_from_csv, inputs=file_input, outputs=csv_output)

# Run the Gradio app
demo.launch()
