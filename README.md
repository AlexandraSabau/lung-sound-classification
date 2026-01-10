# Lung Sound Classification and Anomaly Detection

## Description
This project focuses on the automatic analysis of lung sounds using deep learning techniques.
Audio recordings are transformed into Log-Mel spectrograms and classified using a Convolutional Neural Network (CNN).

## Project Objectives
- Lung sound classification into multiple respiratory conditions
- Feature extraction using Log-Mel spectrograms
- CNN-based multi-class classification
- Reformulation as anomaly detection (Normal vs Abnormal)

## Dataset
The dataset consists of lung sound recordings in WAV format, associated with medical diagnoses such as:
Healthy, COPD, Pneumonia, URTI, Bronchiectasis and Bronchiolitis.

## Methods
- Audio preprocessing and normalization
- Log-Mel spectrogram generation
- CNN architecture with Batch Normalization and Dropout
- Model evaluation using precision, recall, F1-score
- Anomaly detection formulation

## Tools & Technologies
- Python
- Librosa
- NumPy
- TensorFlow / Keras
- Matplotlib

## Results
- Multi-class classification accuracy: ~89%
- Anomaly detection accuracy (Normal vs Abnormal): ~98%

## Documentation
The full project report is available in the `report` folder.

