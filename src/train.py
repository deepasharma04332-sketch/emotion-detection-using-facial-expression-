"""
train.py
Trains the CNN on the FER-2013 dataset.

Expected folder structure (after downloading FER-2013 from Kaggle):

data/
    train/
        Angry/
        Disgust/
        Fear/
        Happy/
        Sad/
        Surprise/
        Neutral/
    test/
        Angry/
        Disgust/
        Fear/
        Happy/
        Sad/
        Surprise/
        Neutral/

Run from project root:
    python src/train.py
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from model import build_model, IMG_SIZE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
TEST_DIR = os.path.join(BASE_DIR, "data", "test")
MODEL_OUT = os.path.join(BASE_DIR, "model", "emotion_model.h5")

BATCH_SIZE = 64
EPOCHS = 50


def get_data_generators():
    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    test_aug = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_aug.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )

    test_gen = test_aug.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    return train_gen, test_gen


def main():
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

    train_gen, test_gen = get_data_generators()
    print("Class label mapping:", train_gen.class_indices)

    model = build_model()
    model.summary()

    callbacks = [
        ModelCheckpoint(MODEL_OUT, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=test_gen,
        callbacks=callbacks,
    )

    print(f"\nTraining complete. Best model saved to: {MODEL_OUT}")
    return history


if __name__ == "__main__":
    main()
