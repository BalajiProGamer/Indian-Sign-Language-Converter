# Indian Sign Language Converter

## Description

The Indian Sign Language Converter project uses machine learning to recognize and convert hand gestures into Indian Sign Language. It collects hand gesture data, trains a Random Forest classifier, and enables real-time gesture recognition using a webcam.

## Features

- Collects hand gesture data from a webcam.
- Processes and prepares the data for training.
- Trains a Random Forest classifier to recognize gestures.
- Real-time gesture recognition with feedback displayed on the screen.
- Saves the trained model for future use.

## Prerequisites

Make sure you have the following libraries installed:

- Python 3.x
- OpenCV
- MediaPipe
- NumPy
- scikit-learn
- Matplotlib

You can install the required libraries using pip:

```bash
pip install opencv-python mediapipe numpy scikit-learn matplotlib

 # Project Structure

 Indian-Sign-Language-Converter/
│
├── collect_imgs.py           # Script to collect hand gesture images.
├── create_dataset.py         # Script to create a dataset from collected images.
├── inference_classifier.py    # Script for real-time gesture recognition.
├── train_classifier.py        # Script to train the Random Forest classifier.
├── data.pickle               # Pickle file containing training data.
├── model.p                   # Pickle file containing the trained model.
├── data/                     # Directory containing collected images organized by class.
└── README.md                 # Project documentation.

 ## Usage
 
 1. Collect Images: Run the collect_imgs.py script to capture images of hand gestures. Press "Q" to start collecting data for each gesture class.

 * python collect_imgs.py

 2. Create Dataset: Use the create_dataset.py script to process the collected images and create a dataset for training.

 * python create_dataset.py

 3.Train Classifier: Train the Random Forest classifier using the train_classifier.py script.

 * python train_classifier.py

 4. Real-time Inference: Run the inference_classifier.py script to start recognizing gestures in real time using your webcam.

 * python inference_classifier.py

 ## License

 This project is open-source and available under the [MIT License](LICENSE).