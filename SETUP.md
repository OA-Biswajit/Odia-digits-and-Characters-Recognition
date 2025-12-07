# Setup and Installation Guide

## Environment Setup

### System Requirements

- **OS**: Windows, macOS, or Linux
- **Python**: 3.7 or higher
- **RAM**: Minimum 4GB (8GB recommended for model training)
- **Storage**: 5GB+ for dataset and models

### Python Environment Setup

#### Option 1: Using Virtual Environment (Recommended)

1. **Create a virtual environment:**
   ```bash
   python -m venv odia_env
   ```

2. **Activate the environment:**
   - **Windows:**
     ```bash
     odia_env\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source odia_env/bin/activate
     ```

3. **Upgrade pip:**
   ```bash
   pip install --upgrade pip
   ```

#### Option 2: Using Conda

1. **Create a conda environment:**
   ```bash
   conda create -n odia_env python=3.9
   ```

2. **Activate the environment:**
   ```bash
   conda activate odia_env
   ```

### Dependencies Installation

Install all required packages:

```bash
pip install tensorflow>=2.10.0 keras opencv-python pandas numpy matplotlib seaborn scikit-learn jupyter
```

Or using the provided requirements.txt (if available):

```bash
pip install -r requirements.txt
```

### Verify Installation

```python
python -c "
import tensorflow as tf
import keras
import cv2
import pandas as pd
import numpy as np
print(f'TensorFlow: {tf.__version__}')
print(f'Keras: {keras.__version__}')
print('All packages installed successfully!')
"
```

## GPU Support (Optional)

For faster training on GPU:

1. **Install CUDA Toolkit** (11.2 or higher)
2. **Install cuDNN** (8.1 or higher)
3. **Verify GPU detection:**
   ```python
   python -c "
   import tensorflow as tf
   print(f'GPU Available: {tf.config.list_physical_devices(\"GPU\")}')
   "
   ```

## Jupyter Notebook Setup

### Install Jupyter

```bash
pip install jupyter jupyterlab
```

### Launch Jupyter

```bash
jupyter notebook
```

The notebook will open in your default browser at `http://localhost:8888`.

## Project Directory Structure Setup

Ensure the following directories exist in your project:

```
Odia_project/
├── data/                 # Original dataset (57 folders: 0-56)
├── new_data/             # Augmented images (will be created)
├── new_train/            # Training split (will be created)
├── new_test/             # Testing split (will be created)
├── notebooks/            # Jupyter notebooks
│   ├── Image_generator.ipynb
│   ├── ImgToCsv.ipynb
│   └── test2.ipynb
└── models/               # Trained models
    └── model.keras
```

## Troubleshooting

### Common Issues

**Issue: TensorFlow GPU not detected**
- Solution: Check CUDA and cuDNN installation
- Alternative: Use CPU (slower but functional)

**Issue: Out of Memory (OOM) errors**
- Solution: Reduce batch size in notebooks
- Reduce image dimensions if necessary

**Issue: Module not found errors**
- Solution: Ensure all packages are installed in your virtual environment
- Check Python version compatibility

**Issue: Jupyter kernel issues**
- Solution: 
  ```bash
  pip install --force-reinstall jupyter
  ```

### Performance Optimization

1. **Use GPU**: If available, enables 10x faster training
2. **Reduce batch size**: If OOM errors occur
3. **Use smaller image dimensions**: Trade-off between accuracy and speed
4. **Enable mixed precision training**: For faster computation

## Next Steps

1. Download or prepare the Odia character dataset
2. Place images in `data/0/` through `data/56/` directories
3. Run `Image_generator.ipynb` to augment the dataset
4. Run `ImgToCsv.ipynb` to convert images to CSV
5. Run `test2.ipynb` to train and evaluate the model

---

**For detailed project information**, refer to [README.md](README.md)
