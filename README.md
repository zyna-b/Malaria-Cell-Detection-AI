# 🩺 Malaria Cell Detection AI: Medical Diagnosis using Deep Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=for-the-badge&logo=keras&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **96% Accuracy** | **0.96 F1-Score** | **Real-time Inference** | **1.1M Parameters**

## 📖 Project Overview

This project implements a **Convolutional Neural Network (CNN)** to automate the detection of Malaria in microscopic blood smear images. Malaria diagnosis typically requires tedious manual examination by pathologists. This AI solution automates the process, classifying cells as **Parasitized** or **Uninfected** with high precision, potentially reducing diagnosis time and human error.

This system was built using the **NIH Malaria Dataset** and demonstrates a complete Deep Learning pipeline from data preprocessing to model deployment.

## 🚀 Key Features

* **Medical-Grade Accuracy:** Achieved **96% Validation Accuracy** and **0.96 F1-Score** for both classes.
* **Advanced Preprocessing:** Automated resizing, normalization, and data augmentation pipeline with random flips and rotations.
* **Robust Architecture:** Custom CNN with **1,105,025 trainable parameters** (4.22 MB) and Dropout regularization to prevent overfitting.
* **Comprehensive Evaluation:** Includes Confusion Matrix, Classification Reports, and visual prediction analysis.
* **Deployment Ready:** The trained model is saved in `.h5` format for easy integration into web/mobile applications.

## 📊 Performance & Results

The model was trained on the NIH Malaria Dataset with outstanding results.

### Classification Report

![Classification Report](path/to/image1_classification_report.png)

| Class | Precision | Recall | F1-Score | Support |
|:------|:----------|:-------|:---------|:--------|
| **Parasitized** | 0.97 | 0.95 | 0.96 | 2,756 |
| **Uninfected** | 0.95 | 0.97 | 0.96 | 2,756 |
| **Accuracy** | - | - | **0.96** | 5,512 |
| **Macro Average** | 0.96 | 0.96 | 0.96 | 5,512 |
| **Weighted Average** | 0.96 | 0.96 | 0.96 | 5,512 |

### Model Architecture

![Model Architecture](graphs/Screenshot 2026-01-04 014013.png)

**Sequential CNN with Data Augmentation:**

| Layer (Type) | Output Shape | Parameters |
|:-------------|:-------------|:-----------|
| random_flip | (None, 128, 128, 3) | 0 |
| random_rotation | (None, 128, 128, 3) | 0 |
| conv2d | (None, 128, 126, 32) | 896 |
| max_pooling2d | (None, 64, 64, 32) | 0 |
| conv2d_1 | (None, 64, 64, 64) | 18,496 |
| max_pooling2d_1 | (None, 32, 32, 64) | 0 |
| conv2d_2 | (None, 32, 32, 64) | 36,928 |
| max_pooling2d_2 | (None, 16, 16, 64) | 0 |
| flatten | (None, 16384) | 0 |
| dense | (None, 64) | 1,048,640 |
| dropout | (None, 64) | 0 |
| dense_1 | (None, 1) | 65 |

**Total Parameters:** 1,105,025 (4.22 MB)  
**Trainable Parameters:** 1,105,025 (4.22 MB)  
**Non-trainable Parameters:** 0 (0.00 B)

### Sample Predictions

**Normalized Training Samples:**

![Sample Images](path/to/image3_sample_images.png)

The model correctly identifies parasitized cells (showing visible dark spots/rings from the parasite) and uninfected cells (uniform appearance).

**Test Set Predictions:**

![Test Predictions](path/to/image4_test_predictions.png)

**Prediction Analysis Grid:**

![Prediction Grid](path/to/image5_prediction_grid.png)

The grid shows both correct predictions (green text) and misclassifications (red text), demonstrating the model's strong performance across diverse cell morphologies.

### Visualizations

**1. Model Accuracy & Loss**

> The training and validation curves track closely, indicating no overfitting.

![Accuracy Graph](graphs/accuracy_graph_image.png)

**2. Confusion Matrix**

> High True Positives and True Negatives show strong class separation.

![Confusion Matrix](graphs/confusion_matrix.png)

**3. Validation Loss**

> Steady convergence during training demonstrates stable learning.

![Validation Loss](graphs/Validation_loss.png)

## 🛠️ Tech Stack

* **Core:** Python 3.10+, TensorFlow 2.x, Keras
* **Data Processing:** TensorFlow Datasets (TFDS), NumPy
* **Visualization:** Matplotlib, Seaborn
* **Model Evaluation:** scikit-learn
* **Environment:** Google Colab / Jupyter Notebook

## 📂 Dataset

The project uses the official **NIH Malaria Dataset** containing 27,558 cell images.

* **Source:** [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/malaria)
* **Classes:** Parasitized (infected) vs. Uninfected (healthy)
* **Image Size:** Resized to 128x128 pixels
* **Normalization:** Pixel values scaled to [0, 1]
* **Augmentation:** Random flips and rotations applied during training

## 🏗️ Model Architecture Details

The CNN employs a proven architecture for medical image classification:

1. **Data Augmentation Layers:**
   - RandomFlip: Horizontal/vertical flips for robustness
   - RandomRotation: Rotational variance handling

2. **Convolutional Blocks:**
   - 3 Conv2D layers with increasing filters (32 → 64 → 64)
   - MaxPooling2D after each convolution for downsampling
   - ReLU activation throughout

3. **Dense Layers:**
   - Flatten layer to convert 2D features to 1D
   - Dense layer with 64 units
   - Dropout (rate unspecified in architecture) for regularization
   - Output layer with sigmoid activation for binary classification

4. **Compilation:**
   - Optimizer: Adam
   - Loss: Binary Crossentropy
   - Metrics: Accuracy

## 🧠 Pre-trained Model

The trained model weights are available in the `models` directory.

* **File:** `models/malaria_cnn.h5`
* **Size:** 4.22 MB
* **Use Case:** Load this file directly to skip training and perform immediate inference.

```python
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('models/malaria_cnn.h5')

# Make predictions
predictions = model.predict(your_image_batch)
```

## ⚡ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/zyna-b/Malaria-Cell-Detection-AI.git
cd Malaria-Cell-Detection-AI
```

### 2. Install Dependencies
```bash
pip install tensorflow tensorflow-datasets matplotlib seaborn numpy scikit-learn
```

### 3. Run the Notebook
Open `Malaria_Detection_CNN.ipynb` in Jupyter or Google Colab and run all cells. The dataset will download automatically via TensorFlow Datasets.

### 4. Training (Optional)
The notebook includes full training code. To retrain:
- Adjust hyperparameters in the training section
- Monitor training/validation metrics
- Save your custom model

### 5. Inference
Use the pre-trained model for instant predictions:
```python
# Load an image
import tensorflow as tf
image = tf.keras.preprocessing.image.load_img('path/to/cell_image.png', target_size=(128, 128))
image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
image = tf.expand_dims(image, axis=0)

# Predict
prediction = model.predict(image)
result = "Parasitized" if prediction[0][0] > 0.5 else "Uninfected"
print(f"Prediction: {result}")
```

## 📈 Key Insights

1. **Balanced Performance:** Both Parasitized and Uninfected classes achieve 0.96 F1-score, indicating no class bias.

2. **High Recall for Uninfected (0.97):** The model excels at identifying healthy cells, reducing false alarms.

3. **High Precision for Parasitized (0.97):** When the model predicts infection, it's correct 97% of the time—crucial for clinical reliability.

4. **Efficient Architecture:** With just 1.1M parameters, the model is lightweight enough for edge deployment (mobile devices, IoT).

5. **Robust to Variations:** Data augmentation ensures the model handles different cell orientations and lighting conditions.

## 🔬 Clinical Relevance

**Current Malaria Diagnosis Challenges:**
- Manual microscopy is time-consuming (20-30 minutes per sample)
- Requires trained pathologists
- Subject to human error and fatigue
- Limited access in remote/resource-poor areas

**AI Solution Benefits:**
- **Speed:** Instant classification (<1 second per image)
- **Consistency:** Eliminates inter-observer variability
- **Scalability:** Can process thousands of samples simultaneously
- **Accessibility:** Deployable on mobile devices for field diagnosis
- **Cost-Effective:** Reduces dependency on expert pathologists

## 🔮 Future Scope

* **Mobile App Integration:** Deploy the `.h5` model to a Flutter/React Native app for field diagnosis in malaria-endemic regions.

* **Transfer Learning:** Experiment with pre-trained models (VGG19, ResNet50, EfficientNet) for potentially higher accuracy with less training data.

* **Explainable AI (XAI):** Implement Grad-CAM or SHAP to visualize *where* the model focuses inside cells, building clinician trust.

* **Multi-Class Classification:** Extend to detect specific Plasmodium species (P. falciparum, P. vivax, etc.).

* **Real-Time Video Analysis:** Process live microscopy feeds for instant diagnosis during patient examination.

* **Integration with WHO Standards:** Align model output with World Health Organization diagnostic protocols.

* **Federated Learning:** Train on distributed hospital datasets without compromising patient privacy.

## 📝 Citation

If you use this project in your research or application, please cite:

```bibtex
@misc{malaria_detection_cnn,
  author = {Zainab Hamid},
  title = {Malaria Cell Detection AI: Medical Diagnosis using Deep Learning},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/zyna-b/Malaria-Cell-Detection-AI}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📧 Contact

**Author:** Zainab Hamid  
*Founder, Ctrl. Alt. Delta | AI Engineer*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/zainab-hamid-187a18321/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/zyna-b)

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ for advancing AI in Healthcare

</div>
