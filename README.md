# 👕 Fashion MNIST Classifier (CNN + Streamlit)

## 📌 Overview

This project is a **Deep Learning-based image classification system** that identifies fashion items using a **Convolutional Neural Network (CNN)** trained on the **Fashion MNIST dataset**.

A **Streamlit web app** is integrated to allow users to upload images and get real-time predictions.

---

## 🎯 Objective

* Build a CNN model for image classification
* Apply preprocessing and data augmentation
* Built using Streamlit (runs locally)
* Understand real-world limitations of ML models

---

## 🧠 Model Details

* **Dataset:** Fashion MNIST
* **Image Size:** 28 × 28 (grayscale)
* **Architecture:** CNN
* **Framework:** TensorFlow / Keras
* **Test Accuracy:** ~90–92%

### 🏷️ Classes

| Label | Category    |
| ----- | ----------- |
| 0     | T-shirt/Top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle Boot  |

---

## ⚙️ Features

✔ CNN-based classification
✔ Data augmentation for better generalization
✔ Image preprocessing pipeline
✔ Confidence-based prediction
✔ Interactive Streamlit UI
✔ Visualization of processed images

---

## 🏗️ Project Structure

```
FashionMNIST/
│── main.py                         # Streamlit application
│── model_training.ipynb           # Model training notebook
│── trained_fashionMNIST_model.keras # Saved trained model
│── README.md                      # Documentation
```

---

## 🚀 Setup & Execution

### 🔹 1. Install Dependencies

```
pip install tensorflow streamlit numpy pillow matplotlib
```

### 🔹 2. Run the Application

```
streamlit run main.py
```

### 🔹 3. Open in Browser

```
http://localhost:8501
```

---

## 🔄 Image Preprocessing Pipeline

Before prediction, uploaded images go through:

* Convert to **grayscale**
* Resize to **28×28**
* Maintain **aspect ratio**
* Center using padding
* **Invert colors** if required
* Normalize pixel values (0–1)

---

## 📊 Data Augmentation

To improve model performance, the following transformations were used:

* Rotation (±10°)
* Width shift (10%)
* Height shift (10%)
* Zoom (±10%)

This helps the model learn variations and reduces overfitting.

---

## ⚠️ Limitations

This model is trained on **Fashion MNIST**, which consists of:

* Low-resolution images (28×28)
* Simple black backgrounds
* Centered objects

👉 Therefore, predictions on real-world images may be inaccurate due to:

* Complex backgrounds
* Lighting variations
* High-resolution details
* Color differences

---

## 💡 Tips for Better Results

* Use **plain background images**
* Ensure the object is **centered**
* Avoid shadows and clutter
* Use **clear, high-contrast images**

---

## 📈 Performance

| Metric              | Value    |
| ------------------- | -------- |
| Training Accuracy   | ~93–95%  |
| Validation Accuracy | ~90–92%  |
| Real-world Accuracy | Moderate |

---

## 🔍 Prediction Logic

The model outputs probabilities for each class:

```python
predicted_class = np.argmax(probabilities)
confidence = probabilities[predicted_class]
```

Prediction is shown only if confidence ≥ 50%.

---

## 🧪 Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Streamlit
* PIL (Image Processing)
* Matplotlib

---

## 🚀 Future Improvements

* Use **Transfer Learning (MobileNetV2 / ResNet)**
* Train on real-world fashion datasets
* Increase input resolution (e.g., 128×128)
* Deploy online (Streamlit Cloud)

---

## 🧸 Simple Explanation

The model is trained on **simple grayscale images**, so it works best on similar inputs.
Real-world images are more complex, which can lead to incorrect predictions.

---

## 🏁 Conclusion

This project successfully demonstrates:

* CNN-based image classification
* Data preprocessing and augmentation
* Model deployment using Streamlit

It also highlights the importance of **dataset quality and real-world generalization**.

---

## 👨‍💻 Author

Developed as part of a Machine Learning / Deep Learning project.



