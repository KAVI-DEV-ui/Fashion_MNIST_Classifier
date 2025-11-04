# 👕👟 Fashion MNIST Classifier (CNN from Scratch)

*A deep learning project to classify fashion items using a Convolutional Neural Network built entirely from scratch.*

---

## 📌 Project Description

This project trains a **Convolutional Neural Network (CNN)** from scratch on the **Fashion MNIST dataset**, which consists of **70,000 grayscale images** of clothing items across **10 categories** (e.g., T-shirt, Trouser, Dress, Sneaker, Bag).

Each image is **28×28 pixels** and represents a single clothing item — making it an ideal dataset for beginners exploring image classification with deep learning.

The project demonstrates the complete workflow: **data preprocessing**, **CNN architecture design**, **model training**, and **evaluation**.

---

## 🎯 What This Project Demonstrates

✅ Building and training a **CNN model from scratch** using TensorFlow/Keras
✅ Understanding how **convolutional layers** extract spatial features from images
✅ Handling **grayscale image data** in deep learning
✅ Comparing **fully connected (ANN)** vs **CNN** model performance
✅ Achieving around **90% test accuracy** with a simple architecture

---

## 🧠 Workflow

### 1. **Data Loading & Preprocessing**

* Load the **Fashion MNIST dataset** directly from Keras datasets
* Normalize pixel values (0–255 → 0–1)
* Reshape data to include channel dimension for CNN input
* Split into training and testing sets

### 2. **Model Building**

* Define a **CNN architecture** with layers such as:

  * `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, `Dropout`
* Compile model with **Adam optimizer** and **categorical crossentropy** loss

### 3. **Training**

* Train for 10–20 epochs with batch size of 64
* Monitor **accuracy and loss curves**

### 4. **Evaluation & Testing**

* Evaluate on test set
* Compare accuracy between:

  * Simple **Fully Connected Neural Network (ANN)**
  * **CNN Model**

### 5. **Visualization**

* Plot confusion matrix
* Display sample predictions with actual labels

---

## 🛠️ Tech Stack

* **Language:** Python 🐍
* **Libraries:**

  * `tensorflow` / `keras` – Deep learning framework
  * `numpy`, `matplotlib`, `seaborn` – Data analysis & visualization
  * `scikit-learn` – Confusion matrix & performance metrics

---

## 📂 Repository Structure

```
/fashion-mnist-classifier
├── mnist_ml_.py
└── README.md
```

---

## ⚙️ How to Run

1. **Clone the Repository**

   ```bash
   git clone https://github.com/KAVI-DEV-ui/fashion-mnist-classifier.git
   cd fashion-mnist-classifier
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook**

   ```bash
   jupyter notebook fashion_mnist_cnn.ipynb
   ```

4. **View Model Performance**

   * Training & validation accuracy
   * Confusion matrix
   * Sample predictions

---

## 📊 Results

| Model               | Test Accuracy | Parameters | Remarks                    |
| ------------------- | ------------- | ---------- | -------------------------- |
| Fully Connected NN  | ~85%          | 1.2M       | Baseline                   |
| CNN (2 Conv Layers) | **~90%**      | 230K       | Much better generalization |

**Example Predictions:**
🧥 T-shirt → Correct ✅
👟 Sneaker → Correct ✅
👜 Bag → Correct ✅
👗 Dress → Incorrect ❌ (Predicted: Coat)

---

## 🚀 Future Improvements

* 🔁 Add **Batch Normalization** and **Dropout** layers for better generalization
* 🧮 Experiment with **Deeper CNNs (ResNet, VGG)**
* 🌈 Visualize learned feature maps
* ☁️ Deploy model using **Streamlit** or **Gradio** for live predictions


---

## 👤 Author

**Kavi Dev**
GitHub: [KAVI-DEV-ui](https://github.com/KAVI-DEV-ui)
