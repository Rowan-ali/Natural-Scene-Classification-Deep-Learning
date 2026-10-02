# 🌄 Natural Scene Classification --- Deep Learning

## 🚀 Live Application

### 👉 [Launch Natural Scene Classification](https://natural-scene-classification-deep-learning-2y3hmumqcgvuqglvsrv.streamlit.app/)

Upload a natural-scene image and receive a predicted scene category,
confidence score, and complete class-probability distribution from the
deployed fine-tuned MobileNetV2 model.

------------------------------------------------------------------------

## 📌 Project Overview

**Natural Scene Classification** is an end-to-end Computer Vision
project designed to classify outdoor images into six scene categories:

`buildings` · `forest` · `glacier` · `mountain` · `sea` · `street`

The project covers the complete deep-learning lifecycle: exploratory
image analysis, preprocessing, augmentation, baseline CNN development,
transfer learning with MobileNetV2, controlled fine-tuning, model
evaluation, error analysis, Grad-CAM explainability, model export, and
real-time deployment using Streamlit.

The final fine-tuned MobileNetV2 achieved **92.17% test accuracy** and
**92.35% macro F1-score**, substantially improving over the custom CNN
baseline.

------------------------------------------------------------------------

## 🎯 Project Objectives

-   Build a reproducible multi-class image-classification pipeline.
-   Inspect class balance, image dimensions, channels, and pixel
    distributions.
-   Apply appropriate image preprocessing and augmentation.
-   Develop a custom CNN baseline from scratch.
-   Apply transfer learning using pretrained MobileNetV2 features.
-   Fine-tune selected pretrained layers using a controlled learning
    rate.
-   Compare models using validation and untouched test-set performance.
-   Analyze class-level errors using a confusion matrix and
    classification report.
-   Explain model decisions using Grad-CAM.
-   Validate the final model on unseen external images.
-   Export the trained model for reuse.
-   Deploy the final classifier through an interactive Streamlit
    application.

------------------------------------------------------------------------

## 🗂️ Dataset

The project uses a six-class natural-scene image dataset organized into
training, testing, and prediction directories.

### Classes

  Class          Description
  -------------- -------------------------------------
  🏢 Buildings   Urban and architectural scenes
  🌲 Forest      Forest and woodland environments
  🧊 Glacier     Snow, ice, and glacier landscapes
  ⛰️ Mountain    Mountain and rocky landscapes
  🌊 Sea         Ocean and coastal scenes
  🛣️ Street      Roads and urban street environments

### Development Split

  Split          Images
  ------------ --------
  Training       11,929
  Validation      2,105
  Test            3,000

The validation set was created from the training data using a fixed
random seed, while the test set remained untouched during model
selection.

------------------------------------------------------------------------

## 🧠 Complete Computer Vision Pipeline

``` text
Natural Scene Images
        │
        ▼
Dataset Inspection & EDA
        │
        ▼
Image Preprocessing
        │
        ├── RGB Conversion
        ├── Image Resizing
        └── Pixel Scaling
        │
        ▼
Training-Time Augmentation
        │
        ├── Horizontal Flip
        ├── Small Rotation
        └── Random Zoom
        │
        ├──────────────────────────────┐
        │                              │
        ▼                              ▼
Custom CNN                    MobileNetV2 Transfer Learning
        │                              │
        ▼                              ▼
Baseline Evaluation             Frozen Feature Extraction
                                       │
                                       ▼
                              Controlled Fine-Tuning
                                       │
                                       ▼
                               Final Model Evaluation
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
               Error Analysis      Grad-CAM        External Images
                    │                  │                  │
                    └──────────────────┼──────────────────┘
                                       ▼
                                  Model Export
                                       │
                                       ▼
                              Streamlit Deployment
```

------------------------------------------------------------------------

# 🔬 Modeling Experiments

## 1. Custom CNN Baseline

A convolutional neural network was developed from scratch to establish a
strong project-specific baseline.

### Architecture

``` text
Input Image (150 × 150 × 3)
        ↓
Conv2D (32) + BatchNorm + Pooling
        ↓
Conv2D (64) + BatchNorm + Pooling
        ↓
Conv2D (128) + BatchNorm + Pooling
        ↓
Conv2D (256) + BatchNorm + Pooling
        ↓
Global Average Pooling
        ↓
Dense (128)
        ↓
Dropout
        ↓
Softmax (6 Classes)
```

### Baseline Performance

  Metric                        Score
  ---------------------- ------------
  Validation Accuracy      **82.28%**
  Test Accuracy            **82.30%**
  Macro F1-score           **82.50%**
  Trainable Parameters        423,302

The nearly identical validation and test accuracy indicates consistent
final generalization, although the training history showed some
overfitting tendency during development.

------------------------------------------------------------------------

## 2. Transfer Learning with MobileNetV2

MobileNetV2 pretrained on ImageNet was used as the feature-extraction
backbone.

The transfer-learning pipeline used:

-   Input size: **224 × 224 × 3**
-   ImageNet pretrained weights
-   Global Average Pooling
-   Dropout regularization
-   Six-class Softmax output
-   Embedded MobileNetV2-compatible rescaling
-   Frozen feature-extraction stage followed by controlled fine-tuning

### Training Strategy

The model was first trained with the pretrained backbone frozen. A
second stage then enabled controlled fine-tuning of selected upper
backbone layers while keeping Batch Normalization layers frozen.

`ReduceLROnPlateau` and `EarlyStopping` were used to improve convergence
and restore the weights associated with the strongest validation
behavior.

------------------------------------------------------------------------

## 🏆 Final Model Performance

### Fine-Tuned MobileNetV2

  Metric                           Score
  ---------------------- ---------------
  Validation Accuracy         **92.78%**
  Test Accuracy               **92.17%**
  Macro F1-score              **92.35%**
  Weighted F1-score           **92.14%**
  Test Loss                   **0.1959**
  Trainable Parameters     **1,202,566**

### Improvement Over Baseline

  Metric                  Custom CNN   Fine-Tuned MobileNetV2     Improvement
  --------------------- ------------ ------------------------ ---------------
  Validation Accuracy         82.28%               **92.78%**   **+10.50 pp**
  Test Accuracy               82.30%               **92.17%**    **+9.87 pp**
  Macro F1-score              82.50%               **92.35%**    **+9.85 pp**

Transfer learning produced a substantial improvement while preserving
strong validation-to-test consistency.

------------------------------------------------------------------------

## 📊 Per-Class Evaluation

  Class         Precision   Recall   F1-score
  ----------- ----------- -------- ----------
  Buildings        93.71%   91.99%     92.84%
  Forest           98.94%   98.73%     98.84%
  Glacier          86.86%   86.08%     86.47%
  Mountain         87.18%   85.52%     86.35%
  Sea              94.32%   97.65%     95.95%
  Street           93.10%   94.21%     93.65%

The strongest performance was observed for **forest** and **sea**. The
most challenging distinction was between **glacier** and **mountain**,
which share similar snow-covered, rocky, and large-scale landscape
features.

------------------------------------------------------------------------

# 🔎 Error Analysis

The final confusion matrix revealed several meaningful visual
ambiguities:

-   **Glacier → Mountain:** 58 images
-   **Mountain → Glacier:** 62 images
-   **Buildings → Street:** 32 images
-   **Street → Buildings:** 26 images

These errors are visually plausible because glacier and mountain scenes
often share snow and rocky terrain, while buildings and street images
frequently contain overlapping urban structures.

------------------------------------------------------------------------

# 🔥 Grad-CAM Explainability

Grad-CAM was implemented using the final MobileNetV2 convolutional
representation to inspect which image regions influenced model
predictions.

The final convolutional activation map has spatial dimensions of **7 ×
7**, so the resulting heatmaps provide coarse localization rather than
pixel-level segmentation.

### Example External Predictions

  Image       Prediction     Confidence
  ----------- ------------ ------------
  10004.jpg   Street             74.92%
  10005.jpg   Mountain           55.32%
  10012.jpg   Street             95.85%
  10013.jpg   Mountain           67.75%
  10017.jpg   Mountain           82.47%
  10021.jpg   Forest             98.40%

The Grad-CAM visualizations showed spatially meaningful attention over
relevant roads, buildings, mountain regions, and landscape structures
rather than a fixed image location.

------------------------------------------------------------------------

# 🌐 Streamlit Application

A professional Streamlit interface was developed to make the final
classifier accessible without requiring notebook execution.

### 👉 [Open the Live Streamlit Application](https://natural-scene-classification-deep-learning-2y3hmumqcgvuqglvsrv.streamlit.app/)

The application provides:

-   Image upload for JPG, JPEG, PNG, and WebP files
-   Uploaded-image preview
-   Predicted scene category
-   Prediction confidence
-   Complete six-class probability distribution
-   Model and inference information
-   Cached model loading for efficient repeated predictions
-   Responsive deployment-oriented interface

The application performs **inference only**. It does not retrain the
neural network when the application starts.

------------------------------------------------------------------------

## ⚠️ Important Preprocessing Note

The exported Keras model already contains its own MobileNetV2-compatible
preprocessing layer:

``` text
Raw RGB pixels [0, 255]
        ↓
Rescaling(1 / 127.5, offset = -1)
        ↓
Model input range [-1, 1]
```

Therefore, `tf.keras.applications.mobilenet_v2.preprocess_input()` must
**not** be applied again outside the model.

Applying preprocessing twice would change the numerical input
distribution and could significantly degrade predictions.

------------------------------------------------------------------------

# 💾 Exported Model

The final trained network is exported as:

``` text
final_mobilenetv2_transfer_model.keras
```

It can be loaded directly using:

``` python
import tensorflow as tf

model = tf.keras.models.load_model(
    "model/final_mobilenetv2_transfer_model.keras",
    compile=False
)
```

### Class Order

``` python
class_names = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street",
]
```

Preserving this exact order is essential when mapping Softmax output
indices to class labels.

------------------------------------------------------------------------

# 🖥️ Running the Application Locally

## 1. Clone the Repository

``` bash
git clone <YOUR-REPOSITORY-URL>
cd Natural-Scene-Classification-Deep-Learning
```

## 2. Create a Virtual Environment

### Windows

``` bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

``` bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

``` bash
python -m pip install -r streamlit_app/requirements.txt
```

## 4. Run Streamlit

From the repository root:

``` bash
python -m streamlit run streamlit_app/app.py
```

Streamlit will normally open the local application at
`http://localhost:8501`.

------------------------------------------------------------------------

# 📁 Repository Structure

``` text
Natural-Scene-Classification-Deep-Learning/
│
├── README.md
├── .gitignore
│
├── notebook/
│   └── Natural_Scene_Classification.ipynb
│
├── model/
│   └── final_mobilenetv2_transfer_model.keras
│
├── report/
│   └── Natural_Scene_Classification_Final_Report.pdf
│
└── streamlit_app/
    ├── app.py
    ├── requirements.txt
    └── .streamlit/
        └── config.toml
```

The structure separates experimentation, trained artifacts,
documentation, and deployment code to keep the repository clear and
maintainable.

------------------------------------------------------------------------

# 🛠️ Technology Stack

### Programming & Data Analysis

-   Python
-   NumPy
-   Pandas

### Computer Vision & Deep Learning

-   TensorFlow
-   Keras
-   Convolutional Neural Networks
-   MobileNetV2
-   Transfer Learning
-   Fine-Tuning
-   Image Augmentation

### Evaluation & Visualization

-   Matplotlib
-   scikit-learn
-   Confusion Matrix
-   Classification Report
-   Grad-CAM

### Deployment

-   Streamlit
-   Keras model serialization

------------------------------------------------------------------------

# 📈 Evaluation Metrics

Models were evaluated using:

-   **Accuracy** --- overall proportion of correctly classified images.
-   **Precision** --- reliability of predictions for each scene class.
-   **Recall** --- proportion of each true class correctly identified.
-   **F1-score** --- balance between precision and recall.
-   **Macro F1-score** --- equal-weight performance across all six
    classes.
-   **Weighted F1-score** --- class-support-weighted F1 performance.
-   **Confusion Matrix** --- detailed distribution of correct and
    incorrect predictions.
-   **Validation Loss** --- used during training and checkpoint
    restoration.

Using multiple metrics provides a more complete assessment than relying
on accuracy alone.

------------------------------------------------------------------------

# 🧪 Experimental Principles

### No Test-Based Model Selection

The test set was kept separate from model-development decisions.
Validation performance was used for model selection and training
control.

### Controlled Fine-Tuning

Transfer learning was performed in stages rather than immediately
updating the entire pretrained network with a large learning rate.

### Early Stopping

The strongest validation checkpoint was restored to reduce unnecessary
overfitting.

### Learning-Rate Scheduling

`ReduceLROnPlateau` automatically reduced the learning rate when
validation loss stopped improving.

### Reproducible Class Mapping

The six-class order is explicitly documented to ensure consistent
training and deployment behavior.

------------------------------------------------------------------------

# 💡 Key Findings

1.  **Transfer learning substantially improved classification
    performance.**\
    Fine-tuned MobileNetV2 increased test accuracy from **82.30% to
    92.17%**.

2.  **Pretrained visual representations were highly effective.**\
    ImageNet features provided a much stronger starting point than
    learning all visual representations from scratch.

3.  **Forest and sea were the strongest classes.**\
    Their visual characteristics were more distinctive in the evaluated
    test set.

4.  **Glacier and mountain were the most difficult pair.**\
    Their shared snow, rock, and landscape characteristics produced the
    largest cross-class confusion.

5.  **Urban classes also showed meaningful overlap.**\
    Buildings and street scenes can contain many of the same structural
    features.

6.  **Controlled fine-tuning improved generalization.**\
    A small learning rate and selective trainability refined pretrained
    features without discarding their useful representations.

7.  **Explainability added qualitative validation.**\
    Grad-CAM indicated that the model generally focused on semantically
    relevant image regions.

8.  **Deployment requires preprocessing consistency.**\
    Because preprocessing is embedded inside the exported model,
    inference code should supply raw resized RGB pixel values without
    applying MobileNetV2 preprocessing a second time.

------------------------------------------------------------------------

# ⚠️ Limitations

-   Visually similar scene categories can remain difficult to
    distinguish.
-   Performance reflects the distribution of the evaluated natural-scene
    dataset.
-   Grad-CAM provides coarse explanatory localization and is not a
    segmentation method.
-   Confidence scores should not be interpreted as guaranteed
    probabilities of correctness.
-   Additional evaluation on more diverse real-world images would
    provide stronger evidence of deployment robustness.

------------------------------------------------------------------------

# 🔮 Future Improvements

Potential extensions include:

-   Evaluation with additional external scene datasets
-   More targeted data for visually ambiguous class pairs
-   Comparison with EfficientNet, ResNet, and other pretrained backbones
-   Automated hyperparameter optimization
-   Advanced augmentation strategies
-   Confidence calibration
-   More extensive Grad-CAM analysis
-   REST API deployment
-   Docker containerization
-   CI/CD integration
-   Deployment monitoring

------------------------------------------------------------------------

# 📚 Project Workflow Summary

``` text
Dataset Loading
      ↓
Exploratory Data Analysis
      ↓
Image Preprocessing
      ↓
Training Augmentation
      ↓
Custom CNN Baseline
      ↓
Baseline Evaluation
      ↓
MobileNetV2 Transfer Learning
      ↓
Frozen Feature Extraction
      ↓
Controlled Fine-Tuning
      ↓
Final Test Evaluation
      ↓
Confusion Matrix & Error Analysis
      ↓
Grad-CAM Explainability
      ↓
External Image Validation
      ↓
Model Export
      ↓
Streamlit Deployment
```

------------------------------------------------------------------------

# 👩‍💻 Author

**Rowan Ali**\
Data Science Student --- Alexandria University

Areas of interest:

`Data Science` · `Machine Learning` · `Deep Learning` ·
`Computer Vision` · `Data Analytics`

------------------------------------------------------------------------

## ⭐ Project Summary

**Natural Scene Classification** demonstrates a complete Computer Vision
lifecycle---from raw image exploration and a custom CNN baseline to
pretrained feature extraction, controlled MobileNetV2 fine-tuning,
detailed evaluation, explainable AI, model export, and real-time web
deployment.

The final fine-tuned MobileNetV2 achieved **92.17% test accuracy** and
**92.35% macro F1-score**, improving test accuracy by **9.87 percentage
points** over the custom CNN baseline.

The project combines model development with experimental discipline,
transparent evaluation, visual error interpretation, explainability,
reproducible inference, and deployment-oriented engineering.

------------------------------------------------------------------------

## 🔗 Quick Links

-   **Live Application:** [Natural Scene
    Classification](https://natural-scene-classification-deep-learning-2y3hmumqcgvuqglvsrv.streamlit.app/)
-   **Notebook:** [Complete Computer Vision
    Notebook](notebook/Natural_Scene_Classification.ipynb)
-   **Final Report:** [Project
    Report](report/Natural_Scene_Classification_Final_Report.pdf)
-   **Streamlit Application:** [Deployment Source](streamlit_app/app.py)

```{=html}
<p align="center">
```
`<strong>`{=html}🌄 Natural Scene
Classification`</strong>`{=html}`<br>`{=html} From raw images to
explainable real-time deep-learning predictions.
```{=html}
</p>
```
