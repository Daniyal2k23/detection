import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# CIFAR-10 classes
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load model function (only once)
def load_trained_model():
    if os.path.exists('cifar10_model.h5'):
        model = load_model('cifar10_model.h5')
        return model
    else:
        st.sidebar.error("Model not found. Please train the model first.")
        return None

def train_model():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # Load data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Normalize data
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Data augmentation
    data_generator = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    # Build model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(10, activation='softmax')
    ])

    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    model.fit(data_generator.flow(X_train, y_train, batch_size=64),
              epochs=10,
              validation_data=(X_test, y_test),
              verbose=1)

    # Save model
    model.save('cifar10_model.h5')
    st.sidebar.success("Model trained and saved as cifar10_model.h5")

def predict_image(image):
    model = load_trained_model()
    if model is None:
        return None, None

    # Preprocess image
    image = cv2.resize(image, (32, 32))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    return class_names[predicted_class], confidence

# Streamlit app
st.title("CIFAR-10 Image Classification")
st.sidebar.title("Options")

# Sidebar options
option = st.sidebar.selectbox("Choose an action", ("Train Model", "Predict Image"))

if option == "Train Model":
    if st.sidebar.button("Start Training"):
        train_model()

elif option == "Predict Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Read and display image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Predict
        if st.button("Predict"):
            label, confidence = predict_image(image)
            if label:
                st.write(f"**Prediction:** {label}")
                st.write(f"**Confidence:** {confidence * 100:.2f}%")
            else:
                st.write("Please train the model first.")
