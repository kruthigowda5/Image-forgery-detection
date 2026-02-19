## AI-Based Image Forgery Detection
An end-to-end deep learning–based image forgery detection system that classifies images as REAL or FORGED using a ResNet-18 model trained on the CASIA2 dataset and deployed through an interactive Streamlit web application.
##  Overview
Digital images can be easily manipulated using modern editing tools, making manual authenticity verification difficult and unreliable.
This project provides an automated solution that analyzes images and detects tampering using deep learning–based forensic features.
The system allows users to upload one or multiple images and instantly view:
Prediction (Real / Forged)
Confidence score
Downloadable PDF report
Scan history
##  Features
Deep learning–based forgery detection
Batch image upload
Confidence score for each prediction
Calibrated decision threshold for reliable classification
Downloadable PDF scan report
Scan history with delete option
Light / Dark theme toggle
Modular design for easy model replacement
##  Model Details
Architecture: ResNet-18
Framework: PyTorch
Dataset: CASIA2
Input size: 224 × 224
Preprocessing: High-pass filtering to highlight manipulation artifacts
Output: Forgery probability with threshold-based classification
## System Workflow
User uploads image(s) through the web interface
Images are preprocessed and resized
The trained CNN model generates a forgery probability
Threshold logic determines the final label
Results are displayed with confidence score
A PDF report can be downloaded
## Tech Stack
Machine Learning
PyTorch
timm
OpenCV
NumPy

Frontend / App Interface
Streamlit

Utilities
ReportLab (PDF generation)
##  Performance
The model is validated on the CASIA2 benchmark dataset and achieves strong performance in distinguishing real and tampered images within this controlled environment.
## Limitations
The current model is trained on a benchmark dataset, and real-world image generalization is affected by domain shift such as different compression levels, editing tools, and resolutions.
Improving cross-domain robustness is part of future work.
##  Future Improvements
Training on larger and more diverse datasets
Cloud deployment
Tampered region localization
User authentication system
## Run Locally
git clone https://github.com/kruthigowda5/Image-forgery-detection.git
cd Image-forgery-detection
pip install -r requirements.txt
streamlit run app.py
## Author
Kruthi P Gowda
https://github.com/kruthigowda5







The threshold was calibrated using validation data and confusion matrix analysis to balance false positives and false negatives.
