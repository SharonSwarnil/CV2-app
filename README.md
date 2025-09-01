# Real-Time Static Hand Gesture Recognition

## Developer Information
Name: Sharon Swarnil Choudhary

## Project Overview
This project implements a real-time hand gesture recognition system using a webcam and a custom-trained CNN model. The application can recognize four static hand gestures: Open Palm, Fist, Peace Sign (V-sign), and Thumbs Up.

## Technology Justification

### Custom CNN Model
I implemented a custom Convolutional Neural Network (CNN) for gesture recognition because:

1. **No External Dependencies**: This approach doesn't rely on external libraries like MediaPipe that may have compatibility issues with newer Python versions.

2. **Flexibility**: A custom model can be easily adapted to recognize additional gestures with more training data.

3. **Performance**: CNNs are well-suited for image classification tasks and can achieve high accuracy with sufficient training data.

4. **Offline Operation**: The model runs completely offline once trained, with no need for internet connectivity.

### Hand Detection Approach
For hand detection, I used a combination of:

1. **Color-based Segmentation**: Using HSV color space to detect skin tones, which works well for hand detection in controlled environments.

2. **Contour Analysis**: Identifying the largest skin-colored contour to locate the hand in the frame.

3. **Morphological Operations**: Cleaning up the segmentation mask to reduce noise and improve detection accuracy.

## Model Architecture
The CNN model has the following architecture:
- Input: 64x64x3 RGB images
- Conv2D (32 filters, 3x3 kernel) + ReLU
- MaxPooling2D (2x2)
- Conv2D (64 filters, 3x3 kernel) + ReLU
- MaxPooling2D (2x2)
- Conv2D (64 filters, 3x3 kernel) + ReLU
- Flatten
- Dense (64 units) + ReLU
- Dropout (0.5)
- Dense (4 units) + Softmax (output layer)

## Setup and Execution Instructions

### Prerequisites
- Python 3.8 or higher
- Webcam

### Installation
1. Clone this repository: