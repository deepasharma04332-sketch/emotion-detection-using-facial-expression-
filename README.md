# Emotion Detection Using Facial Expression

CNN (Convolutional Neural Network) based deep learning project jo webcam se real-time facial
expressions detect karta hai aur 7 emotions classify karta hai: Angry, Disgust, Fear, Happy,
Sad, Surprise, Neutral.

**Tech stack:** Python, OpenCV, TensorFlow/Keras, CNN

---

## Folder Structure

```
emotion-detection-project/
├── data/
│   ├── train/          # training images, emotion-wise subfolders
│   └── test/            # testing images, emotion-wise subfolders
├── model/
│   └── emotion_model.h5 # trained model (generated after training)
├── src/
│   ├── model.py          # CNN architecture
│   ├── train.py          # training script
│   ├── webcam_detect.py  # real-time webcam emotion detection
│   └── prepare_dataset.py # converts FER2013 CSV into image folders
├── requirements.txt
└── README.md
```

---

## Step 1: Setup

```bash
pip install -r requirements.txt
```

## Step 2: Dataset Download (FER-2013)

Tujhe dataset kahan se milega, yahan steps hain:

1. Kaggle pe jaa: https://www.kaggle.com/datasets/msambare/fer2013
   (Kaggle account free hai, login karna padega)
2. Download button dabaa aur zip file unzip kar.
3. Do possible cases:
   - **Case A:** Tujhe sirf `fer2013.csv` file milegi (ek hi CSV mein sab images pixel-format
     mein hoti hain). Isko `data/fer2013.csv` mein rakh de, fir run kar:
     ```bash
     python src/prepare_dataset.py
     ```
     Yeh automatically `data/train/` aur `data/test/` folders bana dega emotion-wise.
   - **Case B:** Tujhe already `train/` aur `test/` folders milte hain emotion subfolders ke
     saath (Angry, Happy, etc). Unko seedha `data/train` aur `data/test` mein copy kar de,
     prepare_dataset.py skip kar.

Agar Kaggle pe access nahi hai ya account nahi banaya, "FER2013 dataset download" Google pe
search karke kisi bhi mirror se bhi le sakti hai — but Kaggle wala sabse reliable hai.

## Step 3: Model Train Karna

```bash
cd src
python train.py
```

- Yeh CNN ko `data/train` pe train karega, `data/test` pe validate karega.
- Best model automatically `model/emotion_model.h5` mein save hoga.
- ~50 epochs ka default hai, EarlyStopping laga hai isliye agar improvement rukk jaye to
  training apne aap stop ho jayegi.
- CPU pe training slow hogi (kaafi der lag sakti hai). Better hai Google Colab use kare
  (free GPU milta hai) — sirf `src/` folder upload kar Colab pe aur same commands chalaa.

## Step 4: Real-Time Webcam Detection

Training complete hone ke baad:

```bash
python webcam_detect.py
```

- Webcam open hoga, face detect hoga (green/colored box ke saath), aur uske upar emotion
  label + confidence % dikhega.
- 'q' dabaakar window close kar sakti hai.

---

## Notes

- Yeh project abhi deploy nahi kiya gaya hai — locally hi run hota hai.
- Agar accuracy kam lage, training epochs badha sakti hai ya data augmentation aur tweak kar
  sakti hai (`train.py` mein `ImageDataGenerator` params).
- FER-2013 dataset thoda noisy hai (kuch images blurry/mislabeled bhi hain), isliye typical
  accuracy 65-70% range mein hoti hai achhe CNN models ke liye bhi — yeh normal hai, isko
  README/resume mein honestly mention karna.
