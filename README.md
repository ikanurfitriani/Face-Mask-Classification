# Face-Mask-Classification
This repository contains a Streamlit application for classifying face masks based on user image uploads. The prediction is made using a pre-trained deep learning model.

## Deployment
Link deployment for public:
https://face-mask-classification-by-ika.streamlit.app/

## Contents
- `app.py`: The main Streamlit application script.
- `mask_classifier.h5`: The trained deep learning model used for classification.
- `Face_Mask_Classification-Ika_Nurfitriani.ipynb`: A Jupyter Notebook used for model training and evaluation.
- `requirements.txt`: To specify the Python packages and their versions that are required to run application.

## Installation
To run this application, you'll need to have Python installed along with the necessary libraries. Ensure you have the following libraries installed:

- streamlit
- numpy
- tensorflow
- others

You can install these libraries using the following command:
```
pip install -r requirements.txt
```

Ensure that you have the following files in your working directory:
- `app.py`
- `mask_classifier.h5`
- `Face_Mask_Classification-Ika_Nurfitriani.ipynb`
- `requirements.txt`

## Running the Application
To start the Streamlit application, use the following command:
```
streamlit run app.py
```
This will launch the application locally. Open the provided URL in your web browser to interact with the face mask classification.

## Usage
1. User Input: Upload image
2. Prediction: After uploading the image, the prediction result will show.

## Screen Capture
The following is a screen capture from the Face Mack Classification App:
- `SS1`
<img src="screenshots/SS-Prediction1.png" alt="SS1" width="800"> 

- `SS2`
<img src="screenshots/SS-Prediction2.png" alt="SS1" width="800">

## Author
[@Ika Nurfitriani](https://github.com/ikanurfitriani)