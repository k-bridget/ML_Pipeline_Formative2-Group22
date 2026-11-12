# Multimodal User Identity & Product Recommendation System (Group 22)

A sequential multimodal authentication and personalized product recommendation pipeline combining **Facial Recognition**, **Voiceprint Verification**, and **Tabular Modeling**.

## System Architecture & Security Flow

1. **Facial Recognition Model:** Extracts facial embeddings/histograms and verifies user identity.
2. **Voiceprint Verification Model:** Validates spoken authorization phrases (*"Yes, approve"*, *"Confirm transaction"*).
3. **Product Recommendation Model:** Predicts the customer's preferred product category (`Books`, `Clothing`, `Electronics`, `Groceries`, `Sports`) from merged social media and transaction records.

---

## Repository Structure

```text
├── data/
│   ├── raw/
│   │   ├── customer_social_profiles.csv
│   │   ├── customer_transactions.csv
│   │   ├── images/                      # 3 members × 3 expressions (Neutral, Smiling, Surprised)
│   │   └── audio/                       # 3 members × 2 phrases (_approve.mp3, _transaction.mp3)
│   └── processed/
│       ├── merged_customer_data.csv     # 213 merged & feature-engineered records (21 columns)
│       ├── image_features.csv           # 45 extracted facial feature vectors (9 originals + 36 augmented)
│       └── audio_features.csv           # 18 extracted acoustic feature vectors (6 originals + 12 augmented)
├── notebooks/
│   ├── 01_data_merge_and_eda.ipynb       # Task 1: Cleaning, inner merge, EDA plots & feature engineering
│   ├── 03_image_processing.ipynb        # Task 2: Image loading, display, augmentation & DeepFace/histogram extraction
│   ├── 02_sound_data_collection_and_processing.ipynb        # Task 3: Audio waveforms, spectrograms, noise/pitch/stretch & MFCC extraction
│   └── 04_model_training.ipynb          # Task 4: Training & evaluation (Accuracy, Macro F1, Log Loss)
├── scripts/
│   ├── process_audio.py                 # Deterministic audio augmentation & feature extraction utilities
│   ├── train_models.py                  # End-to-end training & model serialization script
│   └── app.py                           # Command-line system simulation (authorized & unauthorized flows)
├── models/
│   ├── face_model.joblib                # Logistic Regression facial classifier + StandardScaler
│   ├── voice_model.joblib               # Random Forest voiceprint classifier + StandardScaler
│   ├── product_recommender.joblib       # XGBoost multiclass product recommendation classifier
│   └── *_encoder.joblib / *_features.joblib
├── README.md

### Setup Instruction

# 1. Clone the repository

git clone https://github.com/k-bridget/ML_Pipeline_Formative2-Group22.git
cd ML_Pipeline_Formative2-Group22

# 2. Install dependencies

pip install pandas numpy scikit-learn xgboost librosa soundfile opencv-python deepface matplotlib seaborn joblib
