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
│       ├── merged_customer_data.csv     
│       ├── image_features.csv           
│       └── audio_features.csv           
├── notebooks/
│   ├── 01_data_merge_and_eda.ipynb      
│   ├── 03_image_processing.ipynb        
│   ├── 02_sound_data_collection_and_processing.ipynb        
│   └── 04_model_training.ipynb          
├── scripts/
│   ├── process_audio.py                 
│   ├── train_models.py                  
│   └── app.py                           
├── models/
│   ├── face_model.joblib                
│   ├── voice_model.joblib               
│   ├── product_recommender.joblib      
│   └── *_encoder.joblib / *_features.joblib
├── README.md

### Setup Instruction

# 1. Clone the repository

git clone https://github.com/k-bridget/ML_Pipeline_Formative2-Group22.git
cd ML_Pipeline_Formative2-Group22

# 2. Install dependencies

pip install pandas numpy scikit-learn xgboost librosa soundfile opencv-python deepface matplotlib seaborn joblib
