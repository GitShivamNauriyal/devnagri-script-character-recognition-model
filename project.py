"""
Devanagari Handwritten Character Recognition
=============================================
A CNN-based classifier for 46 classes of Devanagari characters and digits.
Trained on the DevanagariHandwrittenCharacterDataset.
"""

import sys
# Force UTF-8 on Windows consoles that default to cp1252
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend so plots save to file
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Rescaling, Conv2D, BatchNormalization, MaxPooling2D,
    Flatten, Dense, Dropout,
)
from sklearn.metrics import classification_report
import cv2

# ── Configuration ───────────────────────────────────────────────────────────
RUN_TRAINING = False  # Set to True when ready to run full training

# ── Paths (relative to this script) ─────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "DevanagariHandwrittenCharacterDataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "Train")
TEST_DIR = os.path.join(DATASET_DIR, "Test")

# ── Image / training hyper-parameters ───────────────────────────────────────
IMG_SIZE = (32, 32)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 46

# ── Folder-name → Hindi-character mapping ───────────────────────────────────
# The folders are sorted alphabetically by image_dataset_from_directory, so
# the label index matches the sorted order below.  We build the mapping
# manually so every prediction can be displayed with its Hindi glyph.
#
# Sorted folder order (same in Train and Test after renaming):
#   character_10_yna   -> ञ      (index 0)
#   character_11_ta    -> ट      (index 1)
#   character_12_thaa  -> ठ      (index 2)
#   character_13_daa   -> ड      (index 3)
#   character_14_dhaa  -> ढ      (index 4)
#   character_15_ana   -> ण      (index 5)
#   character_16_tabla -> त      (index 6)
#   character_17_tha   -> थ      (index 7)
#   character_18_da    -> द      (index 8)
#   character_19_dha   -> ध      (index 9)
#   character_1_ka     -> क      (index 10)
#   character_20_na    -> न      (index 11)
#   character_21_pa    -> प      (index 12)
#   character_22_pha   -> फ      (index 13)
#   character_23_ba    -> ब      (index 14)
#   character_24_bha   -> भ      (index 15)
#   character_25_ma    -> म      (index 16)
#   character_26_yaw   -> य      (index 17)
#   character_27_ra    -> र      (index 18)
#   character_28_la    -> ल      (index 19)
#   character_29_va    -> व      (index 20)
#   character_2_kha    -> ख      (index 21)
#   character_30_shaw  -> श      (index 22)
#   character_31_shha  -> ष      (index 23)
#   character_32_sa    -> स      (index 24)
#   character_33_ha    -> ह      (index 25)
#   character_34_shra  -> क्ष    (index 26)
#   character_35_tra   -> त्र    (index 27)
#   character_36_gya   -> ज्ञ    (index 28)
#   character_3_ga     -> ग      (index 29)
#   character_4_gha    -> घ      (index 30)
#   character_5_kna    -> ङ      (index 31)
#   character_6_cha    -> च      (index 32)
#   character_7_chha   -> छ      (index 33)
#   character_8_ja     -> ज      (index 34)
#   character_9_jha    -> झ      (index 35)
#   digit_0            -> ०      (index 36)
#   digit_1            -> १      (index 37)
#   digit_2            -> २      (index 38)
#   digit_3            -> ३      (index 39)
#   digit_4            -> ४      (index 40)
#   digit_5            -> ५      (index 41)
#   digit_6            -> ६      (index 42)
#   digit_7            -> ७      (index 43)
#   digit_8            -> ८      (index 44)
#   digit_9            -> ९      (index 45)

FOLDER_TO_HINDI = {
    "character_10_yna":  "ञ",
    "character_11_ta":   "ट",
    "character_12_thaa": "ठ",
    "character_13_daa":  "ड",
    "character_14_dhaa": "ढ",
    "character_15_ana":  "ण",
    "character_16_tabla":"त",
    "character_17_tha":  "थ",
    "character_18_da":   "द",
    "character_19_dha":  "ध",
    "character_1_ka":    "क",
    "character_20_na":   "न",
    "character_21_pa":   "प",
    "character_22_pha":  "फ",
    "character_23_ba":   "ब",
    "character_24_bha":  "भ",
    "character_25_ma":   "म",
    "character_26_yaw":  "य",
    "character_27_ra":   "र",
    "character_28_la":   "ल",
    "character_29_va":   "व",
    "character_2_kha":   "ख",
    "character_30_shaw": "श",
    "character_31_shha": "ष",
    "character_32_sa":   "स",
    "character_33_ha":   "ह",
    "character_34_shra": "क्ष",
    "character_35_tra":  "त्र",
    "character_36_gya":  "ज्ञ",
    "character_3_ga":    "ग",
    "character_4_gha":   "घ",
    "character_5_kna":   "ङ",
    "character_6_cha":   "च",
    "character_7_chha":  "छ",
    "character_8_ja":    "ज",
    "character_9_jha":   "झ",
    "digit_0": "०",
    "digit_1": "१",
    "digit_2": "२",
    "digit_3": "३",
    "digit_4": "४",
    "digit_5": "५",
    "digit_6": "६",
    "digit_7": "७",
    "digit_8": "८",
    "digit_9": "९",
}


def get_hindi_labels(class_names):
    """Return a list of Hindi glyphs aligned with the class_names list."""
    return [FOLDER_TO_HINDI[name] for name in class_names]


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Loading datasets …")
print("=" * 60)

training_dataset = tf.keras.utils.image_dataset_from_directory(
    directory=TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=42,
)

testing_dataset = tf.keras.utils.image_dataset_from_directory(
    directory=TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
)

# The class names are identical for both datasets (alphabetically sorted
# subfolder names).  Build the Hindi label list once.
class_names = training_dataset.class_names
hindi_labels = get_hindi_labels(class_names)

print(f"\nClasses ({len(class_names)}): {class_names}")
print(f"Hindi labels: {hindi_labels}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. SHIROREKHA (HEADLINE) DETECTION & NORMALIZATION  (OpenCV)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Shirorekha (headline) detection & normalization …")
print("=" * 60)


def detect_shirorekha(img_gray, top_fraction=0.5):
    """Detect the Shirorekha (horizontal headline) in a Devanagari character.

    Uses a horizontal projection profile: for each row, count the number of
    foreground (dark) pixels.  The Shirorekha is the row with the highest
    projection value in the upper portion of the image.

    Returns
    -------
    shirorekha_row : int
        Row index of the detected headline.
    projection : np.ndarray
        Full horizontal projection profile (one value per row).
    """
    # Binarise: Otsu threshold (foreground = 0/black on white bg)
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Horizontal projection – sum of foreground pixels per row
    projection = np.sum(binary, axis=1) / 255  # normalise to pixel count

    # Search only in the upper portion of the image
    search_limit = int(len(projection) * top_fraction)
    shirorekha_row = int(np.argmax(projection[:search_limit]))

    return shirorekha_row, projection


def align_shirorekha(img_gray, shirorekha_row, target_row=6):
    """Align the image vertically so the detected Shirorekha is shifted to target_row.

    This acts as a vertical translation normalizer, reducing variances caused by
    varying top margins in different handwriting styles.
    """
    shift = target_row - shirorekha_row
    h, w = img_gray.shape
    # Transformation matrix for translation: [1 0 dx], [0 1 dy]
    M = np.float32([[1, 0, 0], [0, 1, shift]])
    # Translate the image, padding with white background (255)
    aligned = cv2.warpAffine(img_gray, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return aligned


# ── Demonstrate on sample images from the training set ──────────────────
sample_folders = ["character_1_ka", "character_20_na", "character_16_tabla",
                  "character_25_ma", "character_33_ha"]

fig, axes = plt.subplots(len(sample_folders), 4, figsize=(14, 3 * len(sample_folders)))
column_titles = ["Original", "Binary (Otsu)", "Projection Profile", "Vertically Aligned"]

for row_idx, folder in enumerate(sample_folders):
    folder_path = os.path.join(TRAIN_DIR, folder)
    # Pick the first image in the folder
    img_file = sorted(os.listdir(folder_path))[0]
    img_path = os.path.join(folder_path, img_file)

    # Read as grayscale
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        continue

    # Detect
    shiro_row, proj = detect_shirorekha(gray)
    # Align
    aligned = align_shirorekha(gray, shiro_row)
    # Binary for display
    _, binary_disp = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ── Plot ──
    # Original
    axes[row_idx, 0].imshow(gray, cmap="gray")
    axes[row_idx, 0].axhline(y=shiro_row, color="red", linewidth=1, label="Shirorekha")
    axes[row_idx, 0].set_ylabel(FOLDER_TO_HINDI.get(folder, folder), fontsize=14)
    axes[row_idx, 0].set_xticks([]); axes[row_idx, 0].set_yticks([])

    # Binary
    axes[row_idx, 1].imshow(binary_disp, cmap="gray")
    axes[row_idx, 1].set_xticks([]); axes[row_idx, 1].set_yticks([])

    # Projection profile (horizontal bar chart)
    axes[row_idx, 2].barh(range(len(proj)), proj, color="steelblue", height=1)
    axes[row_idx, 2].axhline(y=shiro_row, color="red", linewidth=1)
    axes[row_idx, 2].invert_yaxis()
    axes[row_idx, 2].set_xlabel("Pixel count")

    # Aligned
    axes[row_idx, 3].imshow(aligned, cmap="gray")
    axes[row_idx, 3].set_xticks([]); axes[row_idx, 3].set_yticks([])

    # Column titles on first row
    if row_idx == 0:
        for col, title in enumerate(column_titles):
            axes[0, col].set_title(title, fontsize=12, fontweight="bold")

plt.suptitle("Shirorekha (Headline) Detection & Translation Normalization",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
shiro_path = os.path.join(SCRIPT_DIR, "shirorekha_analysis.png")
plt.savefig(shiro_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  → Shirorekha analysis saved to {shiro_path}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. BUILD MODEL
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Building CNN model …")
print("=" * 60)

model = Sequential([
    # Rescale pixel values from [0, 255] to [0, 1]
    Rescaling(1.0 / 255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

    # Block 1
    Conv2D(32, (3, 3), strides=1, activation="relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

    # Block 2
    Conv2D(32, (3, 3), strides=1, activation="relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

    # Block 3
    Conv2D(64, (3, 3), strides=1, activation="relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

    # Block 4
    Conv2D(64, (3, 3), strides=1, activation="relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

    # Flatten + fully connected
    Flatten(),
    Dense(128, activation="relu", kernel_initializer="he_uniform"),
    BatchNormalization(),
    Dropout(0.4),

    Dense(64, activation="relu", kernel_initializer="he_uniform"),
    BatchNormalization(),
    Dropout(0.4),

    # Output
    Dense(NUM_CLASSES, activation="softmax"),
])

model.summary()

if RUN_TRAINING:
    # ═══════════════════════════════════════════════════════════════════════════
    # 4. COMPILE & TRAIN
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Compiling & training …")
    print("=" * 60)

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    history = model.fit(
        training_dataset,
        validation_data=testing_dataset,
        epochs=EPOCHS,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. PLOT TRAINING CURVES
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Saving training curves …")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"],      "ro-", label="Train")
    axes[0].plot(history.history["val_accuracy"],  "bs--", label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Training & Validation Accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].grid(True)

    # Loss
    axes[1].plot(history.history["loss"],      "ro-", label="Train")
    axes[1].plot(history.history["val_loss"],  "bs--", label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training & Validation Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(SCRIPT_DIR, "training_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  → Saved to {plot_path}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 6. EVALUATE ON TEST SET
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Evaluating on test set …")
    print("=" * 60)

    test_loss, test_acc = model.evaluate(testing_dataset)
    print(f"\nTest accuracy : {test_acc:.4f}")
    print(f"Test loss     : {test_loss:.4f}")

    # ── Classification report ───────────────────────────────────────────────
    # Collect all true labels and predictions across the full test set.
    y_true = []
    y_pred = []

    for images, labels in testing_dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\n" + "-" * 60)
    print("Classification Report")
    print("-" * 60)
    print(classification_report(y_true, y_pred, target_names=hindi_labels))

    # ═══════════════════════════════════════════════════════════════════════════
    # 7. SAMPLE PREDICTION
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Sample predictions …")
    print("=" * 60)

    # Grab one batch from the test set
    for images, labels in testing_dataset.take(1):
        preds = model.predict(images, verbose=0)
        # Show first 5 samples
        fig, axes = plt.subplots(1, 5, figsize=(15, 4))
        for i in range(5):
            true_idx = np.argmax(labels[i].numpy())
            pred_idx = np.argmax(preds[i])
            true_label = hindi_labels[true_idx]
            pred_label = hindi_labels[pred_idx]
            correct = "✓" if true_idx == pred_idx else "✗"

            axes[i].imshow(images[i].numpy().astype("uint8"))
            axes[i].set_title(
                f"True: {true_label}\nPred: {pred_label} {correct}",
                fontsize=11,
            )
            axes[i].axis("off")

        plt.tight_layout()
        sample_path = os.path.join(SCRIPT_DIR, "sample_predictions.png")
        plt.savefig(sample_path, dpi=150)
        plt.close()
        print(f"  → Saved to {sample_path}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 8. SAVE MODEL
    # ═══════════════════════════════════════════════════════════════════════════
    model_path = os.path.join(SCRIPT_DIR, "model_devanagari.keras")
    model.save(model_path)
    print(f"\n✅ Model saved to {model_path}")
    print("Done!")
else:
    print("\nTraining skipped as RUN_TRAINING = False. Preprocessing demo successfully generated.")

