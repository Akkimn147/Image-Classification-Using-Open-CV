# Image-Classification-Using-Open-CV
In this Machine learning project, I have classify some celebrities, restricting classification to only 6 people,
1) Angelina Jolie
2) Jethalal
3) Milana Nagaraj
4) Simrat Kaur
5) Jason Statham
6) Hande Ercel

Project Components
Face and Eye Detection: OpenCV is used for detecting faces and eyes in an input image. This is achieved through pre-trained Haar Cascade classifiers.

Cropping and Preprocessing: Once the faces and eyes are detected, the regions of interest (ROIs) are cropped from the original image. These ROIs are then converted to grayscale for further processing.

Base64 Encoding: The grayscale ROIs are encoded in Base64 format, which is a widely-used method for representing binary data as text. This encoding facilitates easy data storage and transmission.

Image Classification Model: A machine learning model is built to classify the grayscale ROIs. The model may use various algorithms such as deep learning (e.g., convolutional neural networks) or traditional machine learning (e.g., SVM, k-Nearest Neighbors).


Also created a streamlit app, which can be deployed over any cloud platform like Azure, AWS etc