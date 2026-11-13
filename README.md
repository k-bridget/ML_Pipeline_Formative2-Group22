# Multimodal User Identity & Product Recommendation System (Group 22)

A sequential multimodal authentication and personalized product recommendation pipeline combining **Facial Recognition**, **Voiceprint Verification**, and **Tabular Modeling**.

## Group Membera

1. Merci Nzabinesha
2. Yves Niyigena
4. Bridget Karungi

## System Architecture 

1. **Facial Recognition Model:** Extracts facial embeddings/histograms and verifies user identity.
2. **Voiceprint Verification Model:** Validates spoken authorization phrases (*"Yes, approve"*, *"Confirm transaction"*).
3. **Product Recommendation Model:** Predicts the customer's preferred product category (`Books`, `Clothing`, `Electronics`, `Groceries`, `Sports`) from merged social media and transaction records.



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

python -m pip install -r requirements-demo.txt

On macOS, install the OpenMP runtime required by XGBoost before running the demo:

```bash
brew install libomp
```

## System Demonstration

Use Python 3.10-3.13, then run the command-line demonstration from the repository root after installing the dependencies. The scikit-learn version is pinned to match the serialized model artifacts:

```bash
python scripts/app.py \
	--image "data/raw/images/Bridget/neutral.jpeg" \
	--audio "data/raw/audio/Bridget_approve.mp3" \
	--face-user Bridget \
	--customer-id A151
```

The saved voice model verifies speaker identity, then the app checks that it matches the supplied face identity before loading the product recommender. A successful authentication prints the top three product categories for the selected customer. To simulate an unauthorized cross-modal attempt, pair Bridget's image with another member's audio:

```bash
python scripts/app.py \
	--image "data/raw/images/Bridget/neutral.jpeg" \
	--audio "data/raw/audio/Merci_approve.mp3" \
	--face-user Bridget \
	--customer-id A151
```

That attempt is denied when the predicted voice identity does not match the face identity. A voice-confidence rejection can also be exercised with a stricter threshold, for example `--voice-threshold 0.99`.

To simulate a face rejection before voice validation, use an identity that is not in the enrolled class list:

```bash
python scripts/app.py \
	--image "data/raw/images/Bridget/neutral.jpeg" \
	--audio "data/raw/audio/Bridget_approve.mp3" \
	--face-user Unauthorized
```

Here `--face-user Unauthorized` is a decision-logic fixture; it does not claim that the saved image model recognized an unknown person.

**Demonstration limitations:** `--face-user` is an explicit identity fixture for exercising the sequential decision logic, not image-model inference. This checkout has no image feature extraction script or `image_features.csv`; its raw images are only Bridget's, and the face model expects 560 engineered values. Supply `--face-features-csv` with the image pipeline's output to run the face model. The saved voice model classifies the speaker; it does not recognize whether the spoken words actually say "Yes, approve" or "Confirm transaction". The product recommendation is a prediction from an existing customer row, not a prediction using image or voice features.
