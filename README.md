# Accident-detection-system
This is a Machine Learning project which has been designed to identify vehicles moving on a highway and detect collision if it happens.

1. What is an accident detection system ??

Each year, over 1.34 million individuals lose their lives in car accidents. The project's goals 
are, in a nutshell, real-time crash detection and vehicle identification. In order to solve the issue, our 
suggested strategy combines computer vision and machine learning. The Google Colab servers, Tensorflow, and Pytorch 
frameworks were used as development tools.

3. Prerequisites

To use this project Python Version > 3.6 is recommended.
To contribute to this project, knowledge of basic python scripting, Machine Learning, and Deep Learning will help.

4. Description

This program includes 4 things.

data: Kaggle dataset on Accident Detection from CCTV footage.
accident-classification.ipynb: This is a jupyter notebook that generates a model to classify the above data. This file generates two important files model.json and model_weights.h5.
detection.py: This file loads the Accident Detection system with the help of model.json and model_weights.h5 files.
camera.py: It packs the camera and executes the detection.py file on the video dividing it frame by frame and displaying the percentage of the prediction in the accident (if present) in the frame.

5. Future work 

We can use an alarm system that can call the nearest police station in case of an accident and also alert them of the severity of the accident.
