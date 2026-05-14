# 🐱 Cats vs Dogs Image Classifier 🐶

A multimedia image processing and classification project that distinguishes between **Cats** and **Dogs** using manual image enhancement techniques and multiple Machine Learning models.

---

## 📁 Project Structure

```
cats-dogs-classifier/
├── main.py              # 🚀 Run everything in order
├── preprocessing.py     # 📦 Step 1: Extract zip dataset
├── image_processing.py  # 🖼️ Step 2: Brightness, Sharpening, Blurring
├── analysis.py          # 📊 Step 3: Histogram & Color Visualization
└── models.py            # 🤖 Step 4: KNN, Decision Tree, Random Forest, SVM
```

---

## ⚙️ Project Workflow

### 📦 1. Data Preparation
Images are extracted from a zip file and standardized:
- **Resize** → all images converted to the same size (100×100)
- **Grayscale** → converted from BGR to a single 2D matrix (0–255)

### 🖼️ 2. Image Enhancement
Manual filters applied pixel-by-pixel on the image matrix:

| Technique | Description |
|---|---|
| ☀️ **Brightness** | Add constant to every pixel → `newPixel = oldPixel + C` |
| 🔍 **Sharpening** | High-Pass Filter to highlight edges and fur details |
| 🌫️ **Blurring** | Average Filter (3×3 kernel) to reduce noise |

### 📊 3. Feature Extraction (Histogram)
Instead of raw pixels, each image is represented as a **1D histogram vector** of 256 values — counting how often each intensity (0–255) appears. This makes classification more robust against small image changes.

### 🤖 4. Classification Models

| Model | How it works |
|---|---|
| 🟦 **KNN** | Classifies based on distance to K nearest neighbors |
| 🌿 **Decision Tree** | Learns decision rules from histogram features |
| 🌲 **Random Forest** | Ensemble of multiple Decision Trees |
| ⚡ **SVM** | Finds optimal boundary separating cats from dogs |

---

## 📈 Results

| 🤖 Model | ✅ Accuracy | 🎯 Precision | 🔁 Recall | 📊 F1-Score |
|---|---|---|---|---|
| 🌿 Decision Tree | 53.09% | 0.51 | 0.51 | 0.49 |
| ⚡ SVM | 51.85% | 0.54 | 0.54 | 0.51 |
| 🟦 KNN | 50.62% | 0.52 | 0.52 | 0.50 |
| 🌲 Random Forest | 43.21% | 0.44 | 0.44 | 0.43 |

> 💡 **Note:** Accuracy near 50% is expected when using only grayscale histograms, since cats and dogs share similar pixel intensity distributions. Histograms discard spatial info like shapes and edges.

---

## 🖼️ Dataset

> 📂 https://drive.google.com/drive/folders/1KlTHFJ8IZPKXpmI234LUWBvCMJjCiga6?usp=sharing

---

## 🛠️ Requirements

```
opencv-python
numpy
matplotlib
scikit-learn
```

---

## 🚀 How to Run

1. Upload `Downloads2.zip` to your Colab environment
2. Run `preprocessing.py` to extract the dataset
3. Run each file in order, or simply run `main.py`

---

## 📚 Libraries Used

| Library | Purpose |
|---|---|
| `cv2` | Image reading, resizing, color conversion |
| `numpy` | Image matrix operations |
| `matplotlib` | Visualization & histogram plots |
| `sklearn` | ML models & evaluation metrics |

---

*Made with 🐱 and 🐶 by rooby*
