# 🔐 CAPTCHA Recognition using CRNN (CTC Loss)

This project implements a CAPTCHA recognition system using a Convolutional Recurrent Neural Network (CRNN) with Connectionist Temporal Classification (CTC) loss. The system is designed to decode character sequences from distorted CAPTCHA images.

## 📁 Dataset

- **Source:** `captcha_images_v2` (provided as a ZIP file)
- **Total images:** 1040
- **Labeling:** Each image file is named with its ground truth label (e.g., `3xk9.png` has label "3xk9").

## 🏗️ Project Structure

- **Main notebook:** [capchareg-with-crnn.ipynb](capchareg-with-crnn.ipynb)
- **Dataset:** `captcha_images_v2.zip`
- **Image size:** 200×50 (width × height)

### Preprocessing

- Images are resized to (50, 200)
- Converted to grayscale
- Normalized to [0, 1] using a Rescaling layer
- Transposed so width becomes the time dimension (for CTC compatibility)

### Label Encoding

- Vocabulary is dynamically calculated from all unique characters in the dataset
- Each character is mapped to an integer for training
- Labels are padded to the maximum label length with -1

## 🧠 Model Architecture

Implemented in [capchareg-with-crnn.ipynb](capchareg-with-crnn.ipynb):

- **Input:** (50, 200, 1)
- **Convolutional Layers:**
  - Conv2D → BatchNorm → MaxPooling2D (3 blocks)
  - Last pooling uses (2, 1) to preserve width
- **Reshape & Dense:**
  - Reshape to 3D tensor for LSTM
  - Dense + Dropout
- **Bidirectional LSTM:** Captures temporal dependencies
- **Output:** Dense layer with softmax activation
- **Loss:** Custom CTC loss using TensorFlow's `ctc_batch_cost`

## 🧪 Data Split

- **Training:** ~832 images (80%)
- **Validation:** ~104 images (10%)
- **Test:** ~104 images (10%)
- Data is split using TensorFlow's `tf.data` pipeline with shuffling, batching, and prefetching.

## 🔁 Loss Function

CTC loss is implemented via a custom loss function using TensorFlow's `ctc_batch_cost`, allowing for alignment-free sequence prediction.

## 🖼️ Results

- Training and validation loss are plotted per epoch
- Decoded predictions are visualized and compared with ground-truth labels
- Accuracy is calculated on the validation and test sets

## 🚀 How to Run

1. Unzip `captcha_images_v2.zip` and update the `dataset_path` in the notebook if needed.
2. Open [capchareg-with-crnn.ipynb](capchareg-with-crnn.ipynb) in Jupyter or VS Code.
3. Run all cells to train and evaluate the model.

---

For full implementation details, see [capchareg-with-crnn.ipynb](capchareg-with-crnn.ipynb).