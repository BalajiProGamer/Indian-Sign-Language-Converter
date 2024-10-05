
# Indian Sign Language Converter

## Description

The Indian Sign Language Converter project aims to recognize and translate hand gestures into corresponding Indian Sign Language characters using machine learning. It collects hand gesture data, trains a Random Forest classifier, and performs real-time recognition through a webcam.

## Features

- Collects hand gesture data using a webcam.
- Creates a dataset for training from captured images.
- Trains a Random Forest classifier to recognize hand gestures.
- Real-time gesture recognition and prediction with visual feedback.
- Save and load trained models for future use.

## Prerequisites

Before running the project, ensure you have the following libraries installed:

- Python 3.x
- OpenCV
- MediaPipe
- NumPy
- scikit-learn
- Matplotlib

You can install them via pip:

```bash
pip install opencv-python mediapipe numpy scikit-learn matplotlib
```

## Project Structure

```
Indian-Sign-Language-Converter/
│
├── collect_imgs.py           # Script for capturing hand gesture images
├── create_dataset.py         # Script for creating dataset from images
├── train_classifier.py       # Script for training the Random Forest classifier
├── inference_classifier.py   # Script for real-time gesture recognition
├── data.pickle               # Pickle file containing dataset
├── model.p                   # Pickle file containing the trained model
├── data/                     # Directory containing collected images organized by classes
└── README.md                 # Project documentation
```

## Dataset

A sample dataset has been included in the `data/` folder, organized by classes. Each subfolder represents a different class of hand gestures (e.g., sign language characters), and contains images collected for training.

If you'd like to use this sample dataset for training, simply follow the steps in the `create_dataset.py` script to generate the dataset file (`data.pickle`). You can also collect new images by running the `collect_imgs.py` script and placing them in the `data/` directory.

Feel free to modify the dataset or add more classes as needed!

## Usage

### 1. Collect Hand Gesture Images

Run the `collect_imgs.py` script to capture images of hand gestures. You will need to press "Q" to start capturing images for each class.

```bash
python collect_imgs.py
```

### 2. Create Dataset

Use the `create_dataset.py` script to process the collected images and create a dataset for training the model.

```bash
python create_dataset.py
```

### 3. Train the Classifier

Train the Random Forest classifier using the `train_classifier.py` script.

```bash
python train_classifier.py
```

### 4. Real-time Gesture Recognition

After training, run the `inference_classifier.py` script for real-time gesture recognition using your webcam.

```bash
python inference_classifier.py
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenCV](https://opencv.org/) for image and video processing.
- [MediaPipe](https://mediapipe.dev/) for hand tracking and gesture recognition.
- [scikit-learn](https://scikit-learn.org/) for machine learning algorithms.
