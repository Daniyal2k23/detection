import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing import image
import os
from PIL import Image

# Function to load and preprocess the image
def load_and_preprocess_image(uploaded_file):
    img = Image.open(uploaded_file)
    img = img.resize((299, 299))  # Resize the image to match Xception input size
    img_array = np.array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the image for Xception
    return img_array

# Function to predict the image
def predict_image(model, uploaded_file):
    img_array = load_and_preprocess_image(uploaded_file)
    prediction = model.predict(img_array)  # Predict using the model
    return prediction

# Load the model
def load_trained_model():
    model_path = "model.h5"  # Replace with the path to your trained model file
    if os.path.exists(model_path):
        model = load_model(model_path)
        return model
    else:
        st.error("Model not found! Please upload a model file.")
        return None

# Main function for the Streamlit app
def main():
    st.title("Image Prediction App")
    
    st.write("Upload an image to make a prediction.")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        model = load_trained_model()
        
        if model is not None:
            # Display the uploaded image
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_column_width=True)
            st.write("")
            
            # Perform prediction
            prediction = predict_image(model, uploaded_file)
            
            # Display the result
            st.write(f"Prediction: {prediction}")
        else:
            st.write("Model loading failed. Try again.")
    
    st.write("Developed by [Your Name]")

# Run the app
if __name__ == '__main__':
    main()
