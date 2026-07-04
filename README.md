Emotion Detection Using Facial Expression--------!

A deep learning project that detects human emotions in real time using a webcam feed. The model is a Convolutional Neural Network (CNN) trained from scratch on the FER-2013 dataset, and it classifies faces into one of seven emotions: **Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral**.

Overview---------!
This project combines classical computer vision (face detection) with deep learning (emotion classification) to build an end-to-end pipeline: a webcam frame comes in, a face is located, and a trained CNN predicts the emotion being expressed, all in real time.

Tech Stack---------!

- **Python**
- **TensorFlow / Keras** – for building and training the CNN
- **OpenCV** – for face detection (Haar Cascade) and real-time video processing
- **NumPy, Matplotlib, scikit-learn** – supporting libraries for data handling and visualization

How It Works--------!

1. Face Detection – OpenCV's Haar Cascade classifier locates faces in each webcam frame.
2. Preprocessing – The detected face is cropped, converted to grayscale, resized to 48x48 pixels, and normalized.
3. Emotion Classification** – The preprocessed face is passed through a trained CNN, which outputs a probability for each of the 7 emotion classes.
4. Display – A bounding box is drawn around the detected face along with the predicted emotion label and confidence score.

Model Architecture--------!

The CNN consists of three convolutional blocks (with batch normalization, max pooling, and dropout for regularization), followed by fully connected layers ending in a softmax output over 7 classes. It was trained using the Adam optimizer with categorical cross-entropy loss, along with early stopping and learning rate reduction to prevent overfitting.

Dataset---------!

The model is trained on **FER-2013**, a widely used benchmark dataset for facial emotion recognition consisting of 48x48 grayscale images labeled across 7 emotion categories. The dataset is publicly available on Kaggle: [FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013).

Results---------!

The model was trained on Google Colab using a GPU runtime and achieved a **best validation accuracy of ~61.7%**. This is consistent with typical results for CNNs trained from scratch on FER-2013, which is a known to be a challenging dataset since some images are low-resolution, ambiguous, or mislabeled. Accuracy could likely be improved further with deeper architectures, transfer learning, or additional data augmentation.

Project Structure--------!


emotion-detection-project/
├── data/
│   ├── train/              # Training images, organized by emotion
│   └── test/                # Testing/validation images, organized by emotion
├── model/
│   └── emotion_model.h5     # Trained CNN model
├── src/
│   ├── model.py              # CNN architecture definition
│   ├── train.py              # Training script
│   ├── webcam_detect.py      # Real-time webcam emotion detection
│   └── prepare_dataset.py    # Converts FER-2013 CSV into image folders (if needed)
├── requirements.txt
└── README.md


Setup & Usage--------!

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Get the dataset

Download FER-2013 from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013). Depending on the format you receive:

- **CSV format** (`fer2013.csv`): place it in `data/fer2013.csv` and run:
  ```bash
  python src/prepare_dataset.py
  ```
  This converts the CSV into properly organized `data/train/` and `data/test/` folders.

- Folder format (already split into `train/`/`test/` with emotion subfolders): copy them directly into `data/train` and `data/test`.

3. Train the model

```bash
cd src
python train.py
```

The best-performing model (based on validation accuracy) is automatically saved to `model/emotion_model.h5`. Training was originally run on Google Colab with a GPU to keep training time reasonable; running on CPU is possible but considerably slower.

4. Run real-time detection

```bash
python webcam_detect.py
```

This opens your webcam, detects faces, and overlays the predicted emotion and confidence score in real time. Press **`q`** to exit.

Notes-------!

- This project currently runs locally and has not been deployed as a web or cloud service.
- The FER-2013 dataset is inherently noisy, so an accuracy in the 60-65% range is expected and considered reasonable for a CNN trained from scratch, even though it may seem low at first glance.
- Future improvements could include experimenting with deeper architectures, transfer learning from pretrained models, or training on a larger/cleaner emotion dataset.

