import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

#model = load_model('brain tumor detection.h5')  
model = load_model('C:\\Users\\HP\\Downloads\\EDI 5th SEM\\Brain_Tumor\\brain tumor detection.h5')

def predict(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    return prediction, img_array[0]

st.title("Brain Tumor Detection")
st.sidebar.title("Upload Image")

uploaded_file = st.sidebar.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    prediction, input_image = predict(uploaded_file)
    if prediction[0][0] > 0.5:
        result = "Tumor Detected"
        st.write(f"Prediction: {result}")

        heatmap_value = np.array([prediction[0][0]])

        heatmap = cv2.resize((255 * heatmap_value).astype(np.uint8), (input_image.shape[1], input_image.shape[0]))

        alpha_channel = heatmap / 255.0 * 0.6  

        superimposed_img = (1 - alpha_channel[:, :, np.newaxis]) * input_image + alpha_channel[:, :,
                                                                                 np.newaxis] * heatmap[:, :, np.newaxis]

        superimposed_img = cv2.normalize(superimposed_img, None, 0, 1, cv2.NORM_MINMAX)

        grayscale_img = cv2.cvtColor(superimposed_img.astype(np.uint8), cv2.COLOR_RGB2GRAY)

        _, binary_image = cv2.threshold(grayscale_img, 0, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(superimposed_img, contours, -1, (255, 0, 0), 2)

        st.image(superimposed_img, caption="Tumor Area Highlighted", use_column_width=True, clamp=True)

        print("Original Image Shape:", input_image.shape)
        print("Heatmap Shape:", heatmap.shape)
        print("Superimposed Image Shape:", superimposed_img.shape)
        print("Model Prediction:", prediction)

    else:
        result = "No Tumor"
        st.write(f"Prediction: {result}")

else:
    st.sidebar.info("Please upload an image.")

st.markdown(
    """
    <style>
        body {
            color: #1E1E1E;
            background-color: #f0f2f6;
        }
        .st-bw {
            background-color: #4CAF50;
            color: #ffffff;
            border-radius: 5px;
            padding: 8px 16px;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)
