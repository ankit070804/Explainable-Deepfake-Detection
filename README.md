Explainable Deepfake Detection System
Dataset : https://www.kaggle.com/datasets/ucimachinelearning/deep-fake-dataset
Overview

Deepfake technology has advanced rapidly, making it increasingly difficult to distinguish manipulated media from authentic content. This project presents an Explainable Deepfake Detection System that classifies facial images as either Real or Fake while providing visual explanations for its predictions using Grad-CAM (Gradient-weighted Class Activation Mapping). The primary objective is to improve transparency in AI-based deepfake detection by allowing users to understand which facial regions influenced the model's decision.

Features

The system is built using the EfficientNet-B0 deep learning architecture and is capable of classifying facial images into real and fake categories with an associated confidence score. To improve interpretability, the application generates Grad-CAM heatmaps that highlight the regions of the image responsible for the model's prediction. An interactive web interface developed using Streamlit enables users to upload images, obtain predictions instantly, and visualize the corresponding explanations.

Technologies Used

The project is developed in Python using PyTorch and Torchvision for deep learning, OpenCV for image processing, Pillow for image handling, NumPy for numerical computations, Streamlit for the web interface, and the Grad-CAM library for explainable AI visualizations. GPU acceleration through CUDA is supported for faster inference when compatible hardware is available.

Project Structure

The repository contains the Streamlit application, the trained EfficientNet-B0 model, supporting assets, sample images, and the required dependency files. The application loads the trained model, preprocesses uploaded images, performs inference, generates Grad-CAM visualizations, and displays the prediction along with its confidence score through the web interface.

Installation

Clone the repository using Git, navigate to the project directory, and create a Python virtual environment if desired. Install all required dependencies using the provided requirements.txt file. Once the dependencies are installed, launch the application by running streamlit run app.py. The application will open in your default web browser, where users can upload an image for analysis.

Working

When an image is uploaded, it is resized and converted into a tensor before being passed to the trained EfficientNet-B0 model. The model predicts whether the image is real or fake and computes a confidence score using the softmax function. Grad-CAM is then applied to the final convolutional layer of the network to generate a heatmap illustrating the regions that contributed most significantly to the prediction. The original image and the Grad-CAM visualization are displayed side by side, allowing users to better understand the model's reasoning.

Explainability

Unlike conventional image classification systems that only provide a prediction, this project integrates Grad-CAM to improve the interpretability of the model. The generated heatmaps visually indicate the facial regions that most influenced the model's decision, helping users evaluate the reliability of the prediction and increasing trust in the system.

Future Improvements

The current implementation focuses on image-based deepfake detection. Future enhancements include extending the system to support video deepfake detection, real-time webcam analysis, automatic face detection prior to classification, additional explainability techniques such as Grad-CAM++ and Score-CAM, improved robustness against unseen deepfake generation methods, model optimization for edge devices, and deployment through a REST API for broader accessibility.

Limitations

The system currently supports only image-based deepfake detection, and its performance depends on the quality and diversity of the training dataset. Although Grad-CAM provides valuable visual explanations, it should be considered an interpretability tool rather than definitive evidence of image manipulation.
