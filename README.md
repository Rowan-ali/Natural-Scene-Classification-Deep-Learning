# 🌍 Natural Scene Image Classification

An end-to-end deep learning computer vision system for **multi-class natural scene classification**, developed using TensorFlow/Keras and deployed through an interactive Streamlit application.

The system classifies an input image into one of six natural-scene categories:

**Buildings · Forest · Glacier · Mountain · Sea · Street**

The project covers the complete machine learning workflow, including image exploration, preprocessing, data augmentation, baseline CNN development, transfer learning, controlled fine-tuning, model evaluation, error analysis, explainability using Grad-CAM, external-image inference, model export, and interactive deployment.

---

## 📌 Project Overview

Natural scene recognition is a **multi-class image classification problem** in which each input image is assigned exactly one class from six mutually exclusive scene categories.

A custom Convolutional Neural Network (CNN) was first developed as a baseline. Transfer learning was then applied using **MobileNetV2 pretrained on ImageNet**, followed by controlled fine-tuning of upper backbone layers.

The final fine-tuned MobileNetV2 substantially outperformed the baseline model and was selected as the production model.

---

## 🎯 Supported Classes

The model predicts one of the following six classes:

| Index | Class |
|---:|---|
| 0 | Buildings |
| 1 | Forest |
| 2 | Glacier |
| 3 | Mountain |
| 4 | Sea |
| 5 | Street |

> **Important:** The class order must remain unchanged because it corresponds directly to the output indices of the trained model.

---

## 🧠 Final Model

The production model uses **MobileNetV2** as a pretrained convolutional feature extractor with a custom classification head.

The model was first trained using frozen pretrained features and was subsequently improved through controlled fine-tuning using a smaller learning rate.

### Final Model Configuration

| Property | Value |
|---|---|
| Architecture | Fine-Tuned MobileNetV2 |
| Pretrained Weights | ImageNet |
| Input Size | `224 × 224 × 3` |
| Number of Classes | 6 |
| Trainable Parameters | 1,202,566 |
| Validation Accuracy | 92.78% |
| Test Accuracy | **92.17%** |
| Macro F1-Score | **92.35%** |
| Export Format | Keras `.keras` |

The final exported model is:

```text
final_mobilenetv2_transfer_model.keras
```

---

## 📊 Model Comparison

Two principal models were evaluated during development:

| Model | Input Size | Trainable Parameters | Validation Accuracy | Test Accuracy | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Baseline CNN | 150 × 150 × 3 | 423,302 | 82.28% | 82.30% | 82.50% |
| **Fine-Tuned MobileNetV2** | **224 × 224 × 3** | **1,202,566** | **92.78%** | **92.17%** | **92.35%** |

Compared with the baseline CNN, transfer learning improved:

- Validation accuracy by approximately **10.50 percentage points**
- Test accuracy by approximately **9.87 percentage points**
- Macro F1-score by approximately **9.85 percentage points**

The fine-tuned MobileNetV2 was therefore selected as the final model because it demonstrated substantially stronger predictive performance and generalization on unseen data.

---

## 🔍 Error Analysis

The final model achieved strong performance across all six classes.

The most challenging distinction was between:

```text
Glacier ↔ Mountain
```

This confusion is understandable because these classes can share similar visual characteristics, including rocky terrain, snow, elevation patterns, and natural backgrounds.

Other observed confusion occurred between:

```text
Buildings ↔ Street
```

because street scenes frequently contain buildings and other urban structures.

The strongest-performing categories included **Forest** and **Sea**, while **Glacier** and **Mountain** remained the most visually challenging pair.

---

## 🔥 Explainability with Grad-CAM

The project includes **Grad-CAM (Gradient-weighted Class Activation Mapping)** to improve model interpretability.

Grad-CAM produces a heatmap showing which spatial regions of an image contributed most strongly to the model's prediction.

For each analyzed image, the notebook displays:

1. The original image
2. The Grad-CAM activation heatmap
3. The heatmap overlaid on the original image
4. The predicted class
5. The prediction confidence

This provides qualitative evidence that the model is using meaningful visual regions rather than relying only on the final classification score.

---

## ⚙️ Training Strategy

The final training workflow incorporated several techniques designed to improve generalization and training stability.

### Data Augmentation

Training images were augmented using lightweight transformations including:

- Horizontal flipping
- Small rotations
- Small zoom transformations

Augmentation was applied only during training.

Validation and test images were not augmented.

### Transfer Learning

MobileNetV2 pretrained on ImageNet was used to leverage previously learned visual representations.

Training was performed in two stages:

**Stage 1 — Feature Extraction**

The MobileNetV2 backbone was frozen while the custom classification head was trained.

**Stage 2 — Controlled Fine-Tuning**

A limited number of upper MobileNetV2 layers were made trainable and optimization continued using a significantly smaller learning rate.

Batch Normalization layers remained frozen during controlled fine-tuning to preserve stable pretrained statistics.

### Training Callbacks

Training incorporated:

- `EarlyStopping`
- `ReduceLROnPlateau`
- Best-weight restoration

These mechanisms reduced unnecessary training and helped prevent substantial overfitting.

---

## 🖥️ Streamlit Application

A Streamlit interface is included to provide an easy-to-use inference system.

The application allows a user to:

- Upload a new natural-scene image
- Preview the uploaded image
- Generate a prediction using the final MobileNetV2 model
- View the predicted class
- View prediction confidence
- Inspect probabilities across all six classes
- View model and inference information

The interface uses the same class ordering and input configuration used during model development.

---

## 📁 Project Structure

```text
scene_classifier_streamlit/
│
├── app.py
│   └── Main Streamlit application
│
├── final_mobilenetv2_transfer_model.keras
│   └── Exported final fine-tuned model
│
├── requirements.txt
│   └── Required Python dependencies
│
├── README.md
│   └── Project documentation and setup instructions
│
└── .streamlit/
    └── config.toml
        └── Streamlit interface configuration
```

The complete development notebook contains the full training, evaluation, error-analysis, Grad-CAM, and model-selection workflow.

---

# 🚀 Local Setup and Installation

## 1. Prerequisites

The application requires Python and the dependencies listed in `requirements.txt`.

Using a virtual environment is strongly recommended to isolate project dependencies.

---

## 2. Open the Project

Open the `scene_classifier_streamlit` folder in VS Code or navigate to it using a terminal.

Example:

```powershell
cd path\to\scene_classifier_streamlit
```

---

## 3. Activate the Virtual Environment

If an existing virtual environment is already configured, activate it before running the application.

### Windows PowerShell

```powershell
path\to\.venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```cmd
path\to\.venv\Scripts\activate
```

After activation, verify the active interpreter:

```powershell
python -c "import sys; print(sys.executable)"
```

The returned path should point to the intended virtual environment.

---

## 4. Install Dependencies

If the required packages are not already installed:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The application requires the packages specified in:

```text
requirements.txt
```

including TensorFlow, Streamlit, NumPy, Pandas, and Pillow.

---

## 5. Add the Trained Model

Ensure the exported model is located in the **same directory as `app.py`**:

```text
scene_classifier_streamlit/
├── app.py
└── final_mobilenetv2_transfer_model.keras
```

The filename must remain:

```text
final_mobilenetv2_transfer_model.keras
```

because the application uses this filename when loading the model.

---

## 6. Verify TensorFlow and Streamlit

Before launching the interface, the environment can be verified using:

```powershell
python -c "import tensorflow as tf; import streamlit as st; print('TensorFlow:', tf.__version__); print('Streamlit:', st.__version__)"
```

If both versions are displayed without an import error, the environment is ready.

---

## 7. Run the Application

Start the Streamlit application with:

```powershell
python -m streamlit run app.py
```

Using `python -m streamlit` ensures that Streamlit runs using the currently selected Python environment.

After startup, Streamlit will display a local URL, typically:

```text
http://localhost:8501
```

Open the URL in a web browser if it does not open automatically.

---

# 🖼️ How to Use the Application

Once the application is running:

1. Open the Streamlit interface.
2. Upload a supported natural-scene image.
3. The image is converted to RGB.
4. The image is resized to `224 × 224`.
5. The trained MobileNetV2 model performs inference.
6. The predicted scene class and confidence are displayed.
7. The probability distribution across all six classes is shown.

Supported upload formats include:

```text
JPG
JPEG
PNG
WEBP
```

---

## 🔄 Inference Pipeline

The production inference workflow is:

```text
New Image
    │
    ▼
Image Upload
    │
    ▼
RGB Conversion
    │
    ▼
Resize to 224 × 224
    │
    ▼
Float32 Tensor
    │
    ▼
Fine-Tuned MobileNetV2
    │
    ▼
Softmax Probabilities
    │
    ├──► Predicted Class
    │
    └──► Prediction Confidence
```

---

## ⚠️ Important Preprocessing Note

The exported final model already contains its required MobileNetV2 input rescaling operation.

Therefore, the Streamlit application **does not call**:

```python
tf.keras.applications.mobilenet_v2.preprocess_input()
```

again during inference.

External images are converted to RGB, resized to `224 × 224`, converted to `float32`, and supplied to the exported model in the original image-value range expected by the integrated preprocessing layer.

Applying MobileNetV2 preprocessing a second time would create an inconsistent inference pipeline and could negatively affect predictions.

---

# 💾 Loading the Exported Model

The final model can be loaded independently from the Streamlit application using TensorFlow/Keras:

```python
import tensorflow as tf

model = tf.keras.models.load_model(
    "final_mobilenetv2_transfer_model.keras",
    compile=False
)

print(model.input_shape)
print(model.output_shape)
```

Expected input:

```text
(None, 224, 224, 3)
```

Expected output:

```text
(None, 6)
```

The six output positions correspond to:

```python
CLASS_NAMES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street",
]
```

---

## 🧪 Example Standalone Prediction

The exported model can also be used without Streamlit:

```python
import numpy as np
import tensorflow as tf
from PIL import Image

CLASS_NAMES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street",
]

model = tf.keras.models.load_model(
    "final_mobilenetv2_transfer_model.keras",
    compile=False
)

image = Image.open("example.jpg").convert("RGB")
image = image.resize((224, 224))

image_array = np.asarray(image, dtype=np.float32)
input_batch = np.expand_dims(image_array, axis=0)

probabilities = model.predict(input_batch, verbose=0)[0]

predicted_index = int(np.argmax(probabilities))
predicted_class = CLASS_NAMES[predicted_index]
confidence = float(probabilities[predicted_index])

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.2%}")
```

---

# 📈 Final Results

The final Fine-Tuned MobileNetV2 achieved:

```text
Validation Accuracy : 92.78%
Test Accuracy       : 92.17%
Macro F1-Score      : 92.35%
```

These results represent a substantial improvement over the custom baseline CNN and demonstrate the effectiveness of pretrained visual representations and controlled fine-tuning for natural-scene recognition.

---

## 🔮 Potential Future Improvements

Possible extensions include:

- Training with a larger and more diverse dataset
- Additional hyperparameter optimization
- Experimenting with EfficientNet, ResNet, or newer pretrained architectures
- More extensive controlled fine-tuning
- Improved calibration of predicted probabilities
- Additional explainability techniques
- Expanded analysis of difficult Glacier/Mountain examples
- Model compression or quantization for lightweight deployment
- Cloud deployment of the Streamlit application
- Automated testing of the inference pipeline

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow / Keras |
| Transfer Learning | MobileNetV2 |
| Image Processing | TensorFlow / Pillow |
| Data Analysis | NumPy / Pandas |
| Explainability | Grad-CAM |
| User Interface | Streamlit |
| Development | Google Colab / VS Code |
| Model Format | Keras `.keras` |

---

## 📌 Reproducibility Notes

For reliable inference:

- Keep the original class order unchanged.
- Use RGB images.
- Resize input images to `224 × 224`.
- Do not duplicate the preprocessing already embedded in the model.
- Use the exported final model rather than rebuilding its architecture manually.
- Run Streamlit using the same Python environment in which the required dependencies are installed.

---

## ⚠️ Limitations

Although the model demonstrates strong performance, predictions may be less reliable for:

- Images substantially different from the training distribution
- Images containing multiple competing scene types
- Highly ambiguous Glacier/Mountain environments
- Urban scenes where Buildings and Street characteristics overlap
- Extremely low-quality, heavily cropped, or visually obstructed images

The displayed confidence corresponds to the model's Softmax probability and should not be interpreted as a guarantee that a prediction is correct.

---

# 🏆 Conclusion

This project demonstrates a complete computer vision workflow for six-class natural scene classification.

Starting from exploratory image analysis and a custom baseline CNN, the system was improved through ImageNet-based MobileNetV2 transfer learning and controlled fine-tuning. The final model achieved **92.17% test accuracy** and a **92.35% Macro F1-score**, substantially outperforming the baseline architecture.

The project extends beyond model training by incorporating systematic evaluation, error analysis, Grad-CAM explainability, reusable inference, model export, and a Streamlit deployment interface.

The resulting system therefore represents an end-to-end workflow from **raw image data to an interpretable and deployable deep learning application**.