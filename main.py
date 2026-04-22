import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

model_path = r"C:\Users\HP\Desktop\trained_fashionMNIST_model.keras"
model = tf.keras.models.load_model(model_path)


class_names = ["T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat","Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]


# function to preprocess the image


# def preprocess_image(image):
#     img = image.convert('L')        # Convert to grayscale
#     img = img.resize((28, 28))      # Resize to match model input
#     img_array = np.array(img) / 255.0
#     img_array = img_array.reshape((1, 28, 28, 1))
#     return img_array
def preprocess_image(image):
    # Convert to grayscale
    img = image.convert('L')

    # Preserve aspect ratio
    img.thumbnail((20, 20))

    # Create a black 28x28 canvas
    canvas = Image.new('L', (28, 28), color=0)
    x_offset = (28 - img.size[0]) // 2
    y_offset = (28 - img.size[1]) // 2
    canvas.paste(img, (x_offset, y_offset))

    # Convert to array
    img_array = np.array(canvas)

    # Invert colors if background is light
    if np.mean(img_array) > 127:
        img_array = 255 - img_array

    # Normalize
    img_array = img_array / 255.0

    # Reshape for CNN
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array

# Streamlit app
st.title("Fashion Item Classifier")
uploaded_image = st.file_uploader("Upload an image..", type = ['jpg', 'png', 'jpeg'])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((100,100))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            img_array = preprocess_image(image)

            # result = model.predict(img_array)
            # result = model.predict(img_array)
            # probabilities = tf.nn.softmax(result[0]).numpy()
            # predicted_class = np.argmax(probabilities)
            # confidence = probabilities[predicted_class]
            result = model.predict(img_array)
            probabilities = result[0]

            # Get the predicted class
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]

            threshold = 0.50

            if confidence >= threshold:
                st.success(f"Prediction: {class_names[predicted_class]}")
                st.info(f"Confidence: {confidence:.2%}")
            else:
                st.warning("Low confidence. Please upload a clearer image.")

            # predicted_class=np.argmax(result)
            # prediction = class_names[predicted_class]

            # st.success(f'Prediction: {prediction}')
            # st.success(f"Prediction: {class_names[predicted_class]}")
            # st.info(f"Confidence: {confidence:.2%}")

            st.subheader("Preprocessed Image")
            st.image(img_array.reshape(28, 28), clamp=True, width=150)
































