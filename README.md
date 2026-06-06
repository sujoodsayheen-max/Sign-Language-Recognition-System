# Sign Language Recognition System

## Overview

This project is a real-time Sign Language Recognition System that recognizes American Sign Language (ASL) alphabet gestures using computer vision and deep learning techniques.

The system uses MediaPipe to detect and extract hand landmarks from webcam input and a PyTorch neural network to classify the detected gesture into an ASL letter.

The current version focuses on alphabet recognition and serves as the first phase of a larger system that can be extended to support word and sentence translation in the future.

---

## Features

* Real-time hand gesture recognition
* Hand landmark detection using MediaPipe
* ASL alphabet classification using PyTorch
* Live webcam prediction
* Prediction smoothing for more stable results
* Training and evaluation pipeline

---

## System Workflow

1. Capture hand images using a webcam.
2. Detect hand landmarks using MediaPipe.
3. Extract 21 hand landmarks (63 features).
4. Train a neural network on the extracted features.
5. Perform real-time gesture prediction using the trained model.

---

## Technologies Used

* Python
* MediaPipe
* PyTorch
* OpenCV
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn

---

## Project Structure

```text
.
├── extract_landmarks.py
├── train_model.py
├── realtime.py
├── models/
├── results/
├── README.md
└── .gitignore
```

---

## File Description

### extract_landmarks.py

* Reads dataset images
* Detects hands using MediaPipe
* Extracts hand landmarks
* Saves landmarks to a CSV file

### train_model.py

* Loads and preprocesses data
* Trains the neural network
* Evaluates model performance
* Saves trained models and results

### realtime.py

* Loads the trained model
* Captures webcam input
* Predicts ASL gestures in real time
* Displays prediction results

---

## Model Architecture

* Input Layer: 63 Features
* Hidden Layer: 256 Neurons + ReLU
* Hidden Layer: 128 Neurons + ReLU
* Hidden Layer: 64 Neurons + ReLU
* Output Layer: ASL Classes

Batch Normalization and Dropout layers are used to improve training stability and reduce overfitting.

---

## Evaluation

Model performance is evaluated using:

* Accuracy Score
* Classification Report
* Confusion Matrix
* Training Loss Curve
* Validation Accuracy Curve

---

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Extract landmarks:

```bash
python extract_landmarks.py
```

Train the model:

```bash
python train_model.py
```

Run real-time recognition:

```bash
python realtime.py
```

---

## Future Work

* Word-level recognition
* Sentence translation
* Speech output generation
* Mobile application deployment
* Web-based interface
* Arabic Sign Language support

---

## Authors

Graduation Project – Sign Language Recognition System

Developed using Computer Vision and Deep Learning techniques.
