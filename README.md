# 🧠 Face Recognition System using Siamese Network

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

---

## 📌 Overview

This project implements a **Face Recognition / Facial Verification System** using a **Siamese Neural Network**.

Unlike traditional classification models, this system learns **similarity between images** instead of assigning fixed labels. It determines whether two facial images belong to the **same person or different individuals**.

---

## 🚀 Key Features

* 🔍 Face Verification (Same / Different person)
* 🧠 Deep Learning-based Siamese Architecture
* 📸 Custom dataset (self-collected images)
* ⚡ Embedding-based similarity comparison
* 🧪 End-to-end training in Jupyter Notebook
* 📊 Visualization of results and similarity scores

---

## 🧠 Model Architecture

The system is based on a **Siamese Neural Network**, consisting of:

* Two identical CNN networks (shared weights)
* Feature embedding generation
* Distance computation:

  * Euclidean Distance
  * Cosine Similarity
* **Contrastive Loss** for training similarity

---

## 📂 Project Structure

```
Face-Recognition/
│── app/                          # Application logic
│── application_data/             # Dataset & preprocessing
│── Facial Verification with a Siamese Network - Final.ipynb
│── Facial Recognition project_poster.pdf
│── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/face-recognition.git
cd face-recognition
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### (If requirements.txt not available)

```bash
pip install tensorflow keras opencv-python numpy matplotlib
```

---

## ▶️ Usage

### 🔹 Run the Notebook

```bash
jupyter notebook
```

Open:

```
Facial Verification with a Siamese Network - Final.ipynb
```

### 🔹 Workflow

1. Load dataset
2. Create image pairs (positive & negative)
3. Train Siamese Network
4. Generate embeddings
5. Compare two face images
6. Output similarity score

---

## 📊 How It Works

* Two images are passed into twin networks
* Each produces a feature embedding
* Distance between embeddings is computed

**Decision Rule:**

* If distance < threshold → ✅ Same Person
* Else → ❌ Different Person

---

## 📸 Dataset

* Custom dataset created using personal images
* Includes:

  * Positive pairs (same person)
  * Negative pairs (different persons)
* Preprocessing:

  * Resizing
  * Normalization

---

## 🧪 Results

* High accuracy in facial verification tasks
* Robust to:

  * Lighting variations
  * Minor pose differences
* Outputs confidence/similarity score

---

## 🛠️ Tech Stack

* **Python**
* **TensorFlow / Keras**
* **OpenCV**
* **NumPy**
* **Matplotlib**
* **Jupyter Notebook**

---

## 📌 Applications

* 🔐 Face Unlock Systems
* 🏢 Smart Attendance Systems
* 🛂 Identity Verification
* 📱 Security & Authentication

---

## 🔮 Future Improvements

* 🎥 Real-time face recognition (webcam integration)
* 🌐 Web app using Flask / FastAPI
* 📱 Mobile deployment
* 📊 Larger & diverse dataset
* ⚡ Model optimization for faster inference

## ⭐ Contribution

Contributions are welcome!
Feel free to fork the repo and submit a PR.

---

## ⭐ If you like this project

Give it a ⭐ on GitHub — it helps!
