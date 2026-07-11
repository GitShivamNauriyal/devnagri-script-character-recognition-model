# Devanagari Handwritten Character Recognition

A robust, end-to-end Deep Learning pipeline to classify handwritten Devanagari characters (36 alphabets and 10 digits) with high accuracy. 

This project is structured for clean software engineering and data science interview presentations, demonstrating proper dataset alignment, custom OpenCV-based preprocessing (headline/Shirorekha detection), and deep CNN classification.

---

## 🌟 Project Highlights

- **Dataset Size:** 92,000 images (78,200 training, 13,800 testing) across 46 distinct classes.
- **Top Evaluation Accuracy:** **98.29% Test Accuracy** using a deep Convolutional Neural Network with Batch Normalization and Dropout regularization.
- **Shirorekha Headline Normalization:** Implemented a real-time vertical translation alignment tool using OpenCV horizontal projection profiles to normalize varied handwriting baseline margins.
- **Corrected Label Mapping:** Resolved folder name conflicts (e.g. `tabla`/`tabala`, `waw`/`va`) between Train and Test splits to ensure correct categorical label alignment.

---

## 🛠️ Pipeline Architecture & Methodology

### 1. Preprocessing: Shirorekha (Headline) Detection & Normalization
In Devanagari script, words and characters are unified by a top horizontal line called the **Shirorekha**. Variations in margins, tilts, and alignment create noise for raw CNNs. 
- **Otsu Binarization:** Images are converted to grayscale and binarized using Otsu's thresholding.
- **Horizontal Projection Profile:** Summing foreground pixels horizontally yields a density profile. The row with the peak density in the top 50% of the image represents the Shirorekha.
- **Vertical Alignment:** Instead of destructive removal (which can cut off parts of the characters), the image is shifted vertically (`cv2.warpAffine`) to align the Shirorekha to a target row (default: 6th row). This normalizes vertical alignment across all writing styles.

*A full demonstration is saved as `shirorekha_analysis.png` when running the pipeline.*

### 2. CNN Classifier Architecture
```
[Input: 32x32x3] -> Rescaling (1./255)
  ├── Conv2D (32, 3x3) -> BatchNorm -> MaxPooling2D (2x2)
  ├── Conv2D (32, 3x3) -> BatchNorm -> MaxPooling2D (2x2)
  ├── Conv2D (64, 3x3) -> BatchNorm -> MaxPooling2D (2x2)
  ├── Conv2D (64, 3x3) -> BatchNorm -> MaxPooling2D (2x2)
  ├── Flatten
  ├── Dense (128, ReLu) -> BatchNorm -> Dropout (40%)
  ├── Dense (64, ReLu)  -> BatchNorm -> Dropout (40%)
  └── Dense (46, Softmax Output)
```

---

## 🚀 How to Run Locally

### Prerequisites
Install the required libraries:
```bash
pip install tensorflow numpy matplotlib opencv-python-headless scikit-learn
```

### Run Demonstration & Preprocessing
To generate the pre-processing plots and visualize model summary without full training:
1. Open `project.py`
2. Keep `RUN_TRAINING = False`
3. Execute the script:
   ```bash
   python project.py
   ```
4. Check the generated `shirorekha_analysis.png`.

### Run Full Training & Evaluation
To perform training:
1. Edit `project.py` and set `RUN_TRAINING = True`.
2. Run the script:
   ```bash
   python project.py
   ```
3. Once training completes:
   - `model_devanagari.keras` will save the trained model.
   - `training_curves.png` will show validation accuracy/loss plots.
   - A full precision/recall classification report will print to the console.
   - `sample_predictions.png` will display predictions with Unicode Hindi glyph labels.

---

## 🎤 Interview Presentation Pointers

If presenting this project in a technical interview, emphasize:
- **Label Alignment Gotcha:** Show how you caught and fixed the Train/Test subdirectory name mismatch which would otherwise result in silent target misclassifications during training evaluation.
- **Shirorekha Normalization vs. Removal:** Explain how Shirorekha removal can distort character shape (e.g., separating loops). Detail why **Vertical Alignment (Translation Normalization)** is a much safer, more robust feature engineering strategy.
- **Regularization Strategy:** Highlight the integration of Batch Normalization after every Conv block and Dropout on Dense layers to achieve >98% generalization accuracy.
