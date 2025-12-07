# Odia Digits and Characters Recognition

A deep learning project for recognizing and classifying Odia (Odisha) script characters and digits using Convolutional Neural Networks (CNN).

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Notebooks](#notebooks)
- [File Descriptions](#file-descriptions)
- [License](#license)

## Overview

This project implements an automated system for recognizing and classifying Odia script characters and digits. It uses deep learning with TensorFlow/Keras to train a CNN model that can identify 57 different Odia characters and digits from image inputs.

**Key Objectives:**
- Classify Odia characters and digits from image data
- Preprocess and augment training data
- Train and evaluate a CNN model
- Generate predictions with high accuracy

## Project Structure

```
Odia_project/
├── Image_generator.ipynb       # Data augmentation notebook
├── ImgToCsv.ipynb              # Image to CSV conversion
├── test2.ipynb                 # Model training and evaluation
├── model.keras                 # Trained model file
├── history.json                # Training history
├── odia_label.csv              # Character-to-label mapping
├── odia_label1.csv             # Alternative label mapping
├── data/                        # Original dataset (57 classes)
├── new_data/                   # Augmented dataset
├── new_train/                  # Training data (split)
├── new_test/                   # Testing data (split)
├── train/                       # Alternative training split
└── test/                        # Alternative test split
```

## Dataset

The dataset contains **57 classes** representing Odia characters and digits:

- **0-9**: Odia numerals (୦-୯) - 10 classes
- **10-56**: Odia alphabetic characters - 47 classes

**Dataset Structure:**
- **Original Images**: Located in `data/` directory (57 subdirectories)
- **Per Class**: 5-55 images per character (JPEG format)
- **Image Size**: Original varies, resized to 100×100 or 28×28 pixels during preprocessing
- **Augmented Dataset**: Extended in `new_data/` using transformations

## Features

### Data Augmentation
- **Rotation**: ±10 degrees
- **Width/Height Shift**: ±10%
- **Shear Range**: ±10%
- **Zoom Range**: ±10%
- **Normalization**: Rescale to 0-1 range

### Model Capabilities
- Multi-class classification (57 classes)
- Real-time image preprocessing
- High-accuracy predictions
- Confusion matrix analysis
- Classification metrics reporting

## Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook or VS Code with Jupyter extension
- TensorFlow 2.x
- Keras
- OpenCV
- Pandas, NumPy, Matplotlib

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/OA-Biswajit/Odia-digits-and-Characters-Recognition.git
   cd Odia_project
   ```

2. **Install dependencies:**
   ```bash
   pip install tensorflow keras opencv-python pandas numpy matplotlib seaborn scikit-learn
   ```

3. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```

## Usage

### 1. Data Augmentation (Image_generator.ipynb)

Generates synthetic training samples from original images to increase dataset size:

```python
# Augments each image 10 times with various transformations
# Creates 50 new images per original image
# Output: new_data/ directory with augmented images
```

**Process:**
- Load original images from `data/` directory
- Apply random transformations (rotation, shift, zoom, etc.)
- Save augmented images to `new_data/`

### 2. Data Conversion (ImgToCsv.ipynb)

Converts augmented images to CSV format for easier handling:

```python
# Reads images from new_data/
# Resizes to 100×100 pixels
# Converts to numpy arrays
# Saves as CSV with 30,000 features per image
```

**Output:**
- `new_data.csv`: Flattened image arrays (rows = images, columns = pixels)
- Dimensions: (3,135 images, 30,000 pixel values)

### 3. Model Training & Evaluation (test2.ipynb)

Builds, trains, and evaluates the CNN model:

```python
# Load data from new_train/ and new_test/
# Train CNN for 50 epochs
# Evaluate on test set
# Generate predictions and metrics
```

**Steps:**
1. Load and preprocess training/testing data
2. Build CNN model with 3 convolutional blocks
3. Compile with categorical crossentropy loss
4. Train for 50 epochs with batch size 32
5. Evaluate and visualize results

## Model Architecture

**Convolutional Neural Network (CNN)**

```
Input Layer: (28, 28, 3)
    ↓
Conv2D(32, 3×3) + ReLU
    ↓
MaxPooling2D(2×2)
    ↓
Conv2D(64, 3×3) + ReLU
    ↓
MaxPooling2D(2×2)
    ↓
Conv2D(128, 3×3) + ReLU
    ↓
MaxPooling2D(2×2)
    ↓
Flatten()
    ↓
Dense(512) + ReLU
    ↓
Dense(57) + Softmax
Output: 57 classes
```

**Model Details:**
- **Parameters**: ~500K
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy
- **Epochs**: 50
- **Batch Size**: 32

## Results

### Model Performance

- **Training Accuracy**: [See history.json for detailed metrics]
- **Validation Accuracy**: Monitored across 50 epochs
- **Test Accuracy**: Evaluated on held-out test set

### Visualization

The model generates:
- **Accuracy Plots**: Training vs. Validation accuracy
- **Loss Plots**: Training vs. Validation loss
- **Confusion Matrix**: Shows per-class classification performance
- **Classification Report**: Precision, recall, F1-score per class

### Output Files

- `history.json`: Training history with epoch-wise metrics
- `Result.xlsx`: Summary results and statistics
- `model.keras`: Trained model for inference

## Notebooks

### Image_generator.ipynb
**Purpose**: Data augmentation and synthetic data generation

**Input**: Original images from `data/` (57 directories)
**Output**: Augmented images in `new_data/`
**Key Steps**:
- Load images using Keras image utilities
- Configure ImageDataGenerator with transformations
- Generate 10 augmented versions per image
- Save to directory structure

### ImgToCsv.ipynb
**Purpose**: Image preprocessing and conversion to CSV format

**Input**: Augmented images from `new_data/`
**Output**: `new_data.csv` with flattened arrays
**Key Steps**:
- Read all images with OpenCV
- Resize to 100×100 pixels
- Convert to numpy arrays
- Flatten and save as CSV

### test2.ipynb
**Purpose**: Model training, evaluation, and analysis

**Input**: Training data from `new_train/`, test data from `new_test/`
**Output**: Trained model, metrics, visualizations
**Key Steps**:
1. Import libraries and load data
2. Configure data generators with augmentation
3. Build CNN model architecture
4. Compile model
5. Train for 50 epochs
6. Evaluate on test set
7. Generate predictions and confusion matrix
8. Visualize accuracy and loss curves

## File Descriptions

| File | Purpose |
|------|---------|
| `model.keras` | Pre-trained CNN model (inference-ready) |
| `history.json` | Training history with metrics per epoch |
| `odia_label.csv` | Mapping of class indices to Odia characters |
| `odia_label1.csv` | Alternative label mapping |
| `LICENSE` | Project license information |

## How to Make Predictions

To use the trained model for new predictions:

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = load_model('model.keras')

# Load and preprocess image
img = image.load_img('path/to/image.jpg', target_size=(28, 28))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.0  # Normalize

# Make prediction
predictions = model.predict(x)
class_index = np.argmax(predictions)
confidence = predictions[0][class_index]

print(f"Predicted class: {class_index}")
print(f"Confidence: {confidence:.2%}")
```

## Technologies Used

- **Deep Learning Framework**: TensorFlow/Keras
- **Image Processing**: OpenCV
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Evaluation**: Scikit-learn (metrics)

## Author

**Biswajit Kumar Senapati** (OA-Biswajit)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Last Updated**: December 2024
**Dataset Classes**: 57 (Odia characters and digits)
**Model Type**: Convolutional Neural Network (CNN)
