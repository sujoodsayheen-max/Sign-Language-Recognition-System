# ✋ Sign Language Recognition System

## 📌 Overview

The **Sign Language Recognition System** is an AI-powered real-time communication assistance project designed to help deaf and mute individuals communicate more effectively in public and private environments.

The system recognizes American Sign Language (ASL) hand gestures using computer vision and deep learning technologies. A webcam captures the user's hand movements, extracts hand landmarks using MediaPipe, and classifies the gesture into alphabet letters using a PyTorch neural network model.

This project aims to reduce communication barriers in places such as:

* Banks
* Companies
* Customer service centers
* Government institutions
* Public and private organizations

The current implementation focuses on **Alphabet-Level Recognition (Phase 1)**, while future development will expand the system toward **Word-Level and Sentence-Level Translation**.

---

# 🎯 Project Objectives

The main objectives of this project are:

* Build a real-time sign language recognition system.
* Assist deaf and mute individuals in daily communication.
* Detect and track hand landmarks accurately.
* Train a deep learning model capable of recognizing ASL alphabet gestures.
* Provide stable and fast predictions using live webcam input.
* Create a foundation for future word and sentence recognition systems.

---

# 🧠 How the System Works

The system follows a multi-stage pipeline:

## 1️⃣ Hand Detection

The webcam captures live video frames.

MediaPipe is used to:

* Detect the hand
* Track hand movement
* Extract 21 hand landmarks

Each landmark contains:

* X coordinate
* Y coordinate
* Z coordinate

This creates a total of **63 numerical features** for each gesture.

---

## 2️⃣ Feature Extraction

The extracted landmarks are converted into numerical vectors and saved into a CSV dataset.

The dataset becomes the input for training the deep learning model.

---

## 3️⃣ Model Training

A neural network was implemented using PyTorch.

The model:

* Learns gesture patterns
* Classifies ASL alphabet letters
* Improves prediction accuracy through training

### Model Architecture

* Input Layer: 63 Features
* Hidden Layer 1: 256 Neurons + ReLU
* Batch Normalization + Dropout
* Hidden Layer 2: 128 Neurons + ReLU
* Hidden Layer 3: 64 Neurons + ReLU
* Output Layer: Number of ASL Classes

---

## 4️⃣ Real-Time Prediction

The trained model is integrated with OpenCV.

The system:

* Captures webcam frames
* Extracts hand landmarks in real time
* Predicts the corresponding ASL letter
* Displays the predicted output on screen

A smoothing buffer is used to improve prediction stability and reduce flickering.

---

# 🏗️ Technologies Used

| Technology   | Purpose                         |
| ------------ | ------------------------------- |
| Python       | Main programming language       |
| MediaPipe    | Hand landmark detection         |
| PyTorch      | Deep learning model             |
| OpenCV       | Real-time webcam processing     |
| NumPy        | Numerical computations          |
| Pandas       | Data processing                 |
| Scikit-learn | Data preprocessing & evaluation |
| Matplotlib   | Visualization                   |
| Seaborn      | Confusion matrix visualization  |

---

# 📂 Project Structure

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

# 📁 Main Files Description

## 🔹 extract_landmarks.py

Responsible for:

* Reading dataset images
* Detecting hands using MediaPipe
* Extracting landmarks
* Saving numerical features into CSV format

---

## 🔹 train_model.py

Responsible for:

* Loading dataset
* Data preprocessing
* Training the neural network
* Validation and testing
* Saving trained models
* Generating evaluation metrics

---

## 🔹 realtime.py

Responsible for:

* Loading trained model
* Capturing webcam stream
* Predicting gestures in real time
* Displaying live recognition results

---

# 📊 Model Evaluation

The system performance was evaluated using:

* Accuracy Score
* Classification Report
* Confusion Matrix
* Validation Accuracy Curves
* Training Loss Curves

The model achieved stable real-time predictions with high validation performance.

---

# 🚀 How to Run the Project

## 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2️⃣ Run Landmark Extraction

```bash
python extract_landmarks.py
```

---

## 3️⃣ Train the Model

```bash
python train_model.py
```

---

## 4️⃣ Start Real-Time Recognition

```bash
python realtime.py
```

---

# 📸 System Features

✅ Real-time hand gesture recognition

✅ MediaPipe hand tracking

✅ Deep learning classification using PyTorch

✅ Stable prediction smoothing

✅ User-friendly interface

✅ Lightweight and scalable architecture

---

# 🔮 Future Improvements

Future development plans include:

* Word-Level Recognition
* Sentence Translation
* Speech Output Integration
* Mobile Application Deployment
* Web-Based Interface
* Multi-hand Detection
* Arabic Sign Language Support

---

# 🌍 Social Impact

This project aims to improve accessibility and communication for deaf and mute individuals by enabling easier interaction with employees and services without requiring a human interpreter.

The system can potentially be deployed in:

* Banks
* Hospitals
* Universities
* Customer service centers
* Government institutions
* Public service environments

---

# 👩‍💻 Authors

Graduation Project — Sign Language Recognition System

Developed using Artificial Intelligence, Computer Vision, and Deep Learning technologies.
