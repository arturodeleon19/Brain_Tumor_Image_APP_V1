🧠 Brain Tumor MRI Classification

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://braintumorimageappv1-eq3v9mhdlpn642eskvcajk.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-green.svg)](https://lightgbm.readthedocs.io/)
[![Framework](https://img.shields.io/badge/App-Streamlit-red.svg)](https://streamlit.io/)


📌 Project Overview

This project applies deep learning to medical imaging by classifying brain MRI scans into two categories: Tumor or No Tumor. The end product is a lightweight, interactive web application built with Streamlit, designed to process uploaded MRI scans and output real-time predictions with confidence scores.

This project was specifically architected to bypass standard deployment constraints (GitHub's 100MB file limit and Streamlit Cloud's 1GB RAM limit) by utilizing an optimized EfficientNetB0 model via Transfer Learning.

🚀 Web App Demonstration

The trained model is deployed via a Streamlit web interface. Users can either upload their own MRI scans or test the model using a pre-loaded selection of validation images.

🗄️ Dataset

The model was trained on a dataset of ~2,146 Brain MRI images.

Format: Images were annotated using the COCO Segmentation format (JSON).

Preprocessing: A custom TensorFlow dataset pipeline was built to dynamically parse the JSON annotations, map categorical IDs to binary labels, and resize the images to 224x224x3 on the fly using asynchronous prefetching (tf.data.AUTOTUNE).

🧠 Modeling Strategy & Development

    Baseline CNN

Initially, a custom Convolutional Neural Network (CNN) was built to establish a performance baseline.

Architecture: 3 Conv2D/MaxPooling blocks + Dense layers with 50% Dropout.

Baseline Metrics: Accuracy: 99.80% | Val_Accuracy: 99.07% | Val_Loss: 0.0402

Limitation: While highly accurate, custom CNNs on small medical datasets can be prone to overfitting and lack the generalized feature-extraction capabilities of established architectures.

    Transfer Learning: EfficientNetB0

To ensure robust real-world performance while maintaining a tiny file footprint (~20MB) for deployment, the model was upgraded to EfficientNetB0.

The Preprocessing Catch: During initial transfer learning, the model degraded to a ~51% accuracy (random guessing). This was diagnosed as a preprocessing mismatch: EfficientNet handles [0, 255] input scaling internally. Once manual 1/255.0 normalization was removed from the pipeline, the model converged immediately.

Fine-Tuning (Phase 2): After training the dense classification head, the top 20 layers of the EfficientNet base were unfrozen. The model was fine-tuned using a severely reduced learning rate (1e-5) and Early Stopping to prevent weights degradation.

💻 Local Setup & Installation

To run this application locally on your machine:

Clone the repository:

git clone https://github.com/yourusername/brain_tumor_project.git cd brain_tumor_project

Install the required dependencies: It is recommended to use a virtual environment (like Anaconda, Python 3.11).

pip install -r requirements.txt

Run the Streamlit app:

streamlit run project.py

📂 Repository Structure

brain_tumor_project/ │ ├── project.py # Streamlit application script ├── brain_tumor_EfficientB0_FT.keras # Fine-tuned EfficientNet model weights ├── requirements.txt # Python dependencies ├── README.md # Project documentation └── sample_images/ # Pre-loaded MRIs for app testing ├── sample_01.jpg
├── sample_02.jpg
└── ...

📈 Future Improvements

Multi-Class Classification: Expanding the dataset to classify specific tumor types (e.g., Glioma, Meningioma, Pituitary).

Segmentation: Transitioning from image-level classification to drawing bounding boxes/polygons around the tumor using the original COCO annotations.
