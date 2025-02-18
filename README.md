# 🦷 Dental Cavity Detection

This project uses **Deep Learning (CNNs)** to detect dental cavities from X-ray or intraoral images. The model is trained using **TensorFlow/Keras** and classifies images as either **Healthy** or **Cavity**.

---

## 📂 Project Structure

```
/dental_cavity_detection
│── /dataset                  # Dataset folder
│── /models                   # Saved models
│── /scripts                  # Python scripts
│   ├── train.py              # Train & save the model
│   ├── predict.py            # Load model & predict images
│── /results                  # Folder for predictions/results
│   ├── training_history.png  # Training accuracy & loss plots
│── .gitignore                # Git ignore settings
│── requirements.txt          # Dependencies
│── README.md                 # Project documentation
```

---

## 🚀 Installation & Setup

### **1️⃣ Clone the Repository**

```bash
git clone https://github.com/onurcanersen/dental-cavity-detection
cd dental_cavity_detection
```

### **2️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

---

## 🎯 Model Training

To train the model and save it:

```bash
python scripts/train.py
```

✔️ This will create `cavity_model.h5` inside `/models`.

---

## 🔍 Predicting an Image

To classify an image:

```bash
python scripts/predict.py path/to/image.jpg
```

Example:

```bash
python scripts/predict.py dataset/test/cavity/sample.jpg
```

✔️ The output will be **"Cavity"** or **"Healthy"**.

---

## 📈 Results & Accuracy

- Model trained on **X-ray images** of healthy and cavity-affected teeth.
- Achieved **XX% accuracy** (update after training).
- Training loss & accuracy plots saved in `/results/training_history.png`.
