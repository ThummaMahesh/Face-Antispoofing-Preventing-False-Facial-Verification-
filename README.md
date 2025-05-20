
# 🛡️ Face Antispoofing - Preventing False Facial Verification

This project is designed to detect **spoofed facial inputs** such as printed photos or video replays to prevent false facial verification. It uses a **deep learning model** to classify whether a detected face is **real** or **fake**.

---

## 🚀 Getting Started

### 📁 Dataset Setup

Before running the training script:

1. Create a folder named `dataset` in your project root.
2. Inside the `dataset` folder, create the following subfolders:

```

dataset/
├── real/
└── fake/

````

3. Place real face images inside the `real/` folder.
4. Place spoofed or fake face images inside the `fake/` folder.

> Make sure the images are in `.jpg`, `.png`, or compatible formats.

---

### 🧠 Train the Model

To train the antispoofing model, run the following script:

```bash
python spoof.py
````

This script will:

* Load images from the dataset
* Train a deep learning model
* Save the trained model into the `model/` directory

---

## 📁 Project Structure

```
project/
├── dataset/
│   ├── real/
│   └── fake/
├── model/
│   └── antispoof_model.h5        # Generated after training
├── spoof.py                      # Main training script
├── README.md
└── requirements.txt
```

---

## 🧰 Technologies Used

* Python
* OpenCV
* TensorFlow / Keras
* NumPy
* Matplotlib

---

## 🔍 What It Does

* Prevents spoofing in facial recognition systems.
* Differentiates between live faces and spoofed faces (photos, videos).
* Uses convolutional neural networks (CNNs) for training.
* Flags any non-live face as **“Not a Live Face.”**

This system can be integrated into:

* Biometric authentication
* Mobile login systems
* Surveillance and security tools
* ATM and kiosk-based verification

---

## 🙌 Contributions

Contributions are welcome! You can help by:

* Improving the model accuracy
* Adding real-world datasets
* Enhancing live video integration

---

## 📧 Contact

For questions or collaboration:

📩 **[maheshthummanani@gmail.com](mailto:maheshthummanani@gmail.com)**

