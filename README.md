# Cognitive-Workload-Estimation
Real-time Mental State Detection using EEG and Eye Blink Analysis

## Overview
This project estimates cognitive load (Relaxed/Focused/Confused)
using multimodal AI combining EEG brainwave features and 
real-time eye blink detection via webcam.

## Accuracy
96.15% mean accuracy (5-fold cross validation)

## Technologies
- Python
- TensorFlow / Keras
- OpenCV
- Streamlit
- Scikit-learn

## How to Run
pip install -r requirements.txt
streamlit run dashboard.py

## Project Structure
cognitive_load_project/
├── dashboard.py
├── mlp_model.py
├── validate.py
├── data/
├── models/
└── output/
