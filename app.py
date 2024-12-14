import numpy as np
import cv2
import streamlit as st
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

# Train the model
def train_model(X_train, y_train, X_test, y_test):
    """
    Trains and saves the CNN model.
    """
    # Check the shape of your training data for debugging
    print("Training data shape:", X_train.shape)
    print("Training labels shape:", y_train.shape)
    
    # Initialize the model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')  # Modify to the correct number of classes
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Data augmentation
    data_generator = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    
    # Fit the model
    model.fit(data_generator.flow(X_train, y_train, batch_size=64), epochs=10, validation_data=(X_test, y_test), verbose=1)

    # Save the trained model
    model.save('trained_model.h5')
    print("Model training complete and saved.")


# Predict the image using the trained model
def predict_image(image):
    """
    Predicts the label and confidence of the uploaded image using the trained model.
    """
    # Load the trained model
    model = load_model('trained_model.h5')
    
    # Preprocess the image
    image_resized = cv2.resize(image, (32, 32))  # Resize to match input shape (32x32)
    image_normalized = image_resized.astype('float32') / 255.0  # Normalize pixel values to [0, 1]
    image_expanded = np.expand_dims(image_normalized, axis=0)  # Add batch dimension
    
    # Predict the class
    prediction = model.predict(image_expanded)
    
    # Get the label with the highest confidence
    label = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][label]
    
    return label, confidence


# Streamlit application to upload images and predict
def main():
    st.title("Forest Image Detection")
    
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the image file into memory
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Show the uploaded image
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Handle prediction when button is pressed
        if st.button("Predict"):
            label, confidence = predict_image(image)
            st.write(f"**Prediction Label:** {label}")
            st.write(f"**Confidence:** {confidence * 100:.2f}%")

if __name__ == "__main__":
    main()
