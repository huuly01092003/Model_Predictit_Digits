import streamlit as st
import pandas as pd
import numpy as np
from keras.models import model_from_json
from PIL import Image, ImageOps
import io
from streamlit_drawable_canvas import st_canvas

# Function to load model from .json and .h5 files
def load_model(model_path, weights_path):
    # Load model structure from .json file
    with open(model_path, 'r') as json_file:
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
    # Load weights into the model
    loaded_model.load_weights(weights_path)
    return loaded_model

# Load all 15 models
models = []
for j in range(15):
    model_path = f"model_{j}.json"
    weights_path = f"model_{j}.h5"
    model = load_model(model_path, weights_path)
    models.append(model)

# Function to preprocess data for prediction
def preprocess_data(test_data):
    test_data = test_data / 255.0
    test_data = test_data.values.reshape(-1, 28, 28, 1)
    return test_data

# Function to preprocess individual images for prediction
def preprocess_image(image):
    image = image.resize((28, 28))
    image = image.convert('L')
    image = np.array(image) / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image

# Function to preprocess drawn images for prediction
def preprocess_drawn_image(image):
    image = image.convert('L')
    image = ImageOps.invert(image)  # Invert the colors for the drawing
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image

# Function to make predictions using ensemble of models
def predict_ensemble(models, test_data):
    results = np.zeros((test_data.shape[0], 10))
    for model in models:
        results += model.predict(test_data)
    return np.argmax(results, axis=1)

# Streamlit app
st.title("MNIST Digit Recognition")

# Sidebar menu for selection
option = st.sidebar.selectbox(
    "Select Input Method",
    ("Upload CSV File", "Upload Image Files", "Draw a Digit")
)

if option == "Upload CSV File":
    st.write("Upload a CSV file containing the test data.")

    # File uploader for CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the uploaded CSV file
        test = pd.read_csv(uploaded_file)
        X_test = preprocess_data(test)

        if st.button("Predict"):
            # Make predictions
            predictions = predict_ensemble(models, X_test)
            test['Label'] = predictions

            # Display predictions
            st.write("Predictions:")
            st.write(test)

            # Optionally, you can display the test images along with their predictions
            st.write("Test Images:")
            num_images = min(len(predictions), 40)
            for i in range(num_images):
                st.image(X_test[i].reshape((28, 28)), caption=f"Predicted Label: {predictions[i]}", use_column_width=True)

            # Add download button for CSV with predictions
            csv = test.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download CSV with Predictions", data=csv, file_name='predictions.csv', mime='text/csv')

elif option == "Upload Image Files":
    st.write("Upload individual image files for prediction.")

    # File uploader for images
    uploaded_images = st.file_uploader("Choose image files", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_images is not None and len(uploaded_images) :
        if st.button("Predict"):
            images_data = []
            predictions_list = []
            st.write("Predictions for uploaded images:")
            for uploaded_image in uploaded_images:
                image = Image.open(uploaded_image)
                preprocessed_image = preprocess_image(image)

                # Make prediction
                prediction = predict_ensemble(models, preprocessed_image)
                predictions_list.append(prediction[0])

                # Collect image data
                image_data = np.array(image.convert('L').resize((28, 28))).flatten()
                images_data.append(image_data)

                # Display the image and the prediction
                st.image(image, caption=f"Predicted Label: {prediction[0]}", use_column_width=True)

            # Create DataFrame for images data
            images_df = pd.DataFrame(images_data)
            images_df['Label'] = predictions_list

            # Add download button for CSV with predictions
            csv = images_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download CSV with Predictions", data=csv, file_name='predictions.csv', mime='text/csv')

elif option == "Draw a Digit":
    st.write("Draw a digit below for prediction.")

    # Draw canvas for drawing digits
    canvas_result = st_canvas(
        stroke_width=10,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=350,
        width=350,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        if st.button("Predict"):
            # Convert the canvas result to an image
            drawn_image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
            # Convert the image to RGB and then preprocess it
            drawn_image = drawn_image.convert('RGB')
            preprocessed_image = preprocess_drawn_image(drawn_image)

            # Make prediction
            predictions = predict_ensemble(models, preprocessed_image)

            # Display the drawn image and the prediction
            st.image(drawn_image, caption=f"Predicted Label: {predictions[0]}", use_column_width=True)

            # Convert drawn image to data and create a DataFrame
            drawn_image_data = np.array(drawn_image.convert('L').resize((28, 28))).flatten()
            drawn_df = pd.DataFrame([drawn_image_data])
            drawn_df['Label'] = predictions[0]

            # Add download button for CSV with the drawn image prediction
            csv = drawn_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download CSV with Prediction", data=csv, file_name='drawn_prediction.csv', mime='text/csv')
            #Tao chạy ngon lành mà ta