# gesture_recognition.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model

class HandGestureRecognizer:
    def __init__(self, model_path='model/gesture_model.h5'):
        # Load the trained model
        self.model = load_model(model_path)
        self.gesture_names = ['Open Palm', 'Fist', 'Peace Sign', 'Thumbs Up']
        
    def preprocess_frame(self, frame, bbox=None):
        """Preprocess frame for model prediction"""
        if bbox is not None:
            x, y, w, h = bbox
            # Extract ROI with padding
            margin = 20
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(w + 2 * margin, frame.shape[1] - x)
            h = min(h + 2 * margin, frame.shape[0] - y)
            roi = frame[y:y+h, x:x+w]
        else:
            # Use the entire frame
            roi = frame
        
        # Resize to model input size
        roi = cv2.resize(roi, (64, 64))
        
        # Normalize pixel values
        roi = roi.astype('float32') / 255.0
        
        # Add batch dimension
        roi = np.expand_dims(roi, axis=0)
        
        return roi
    
    def detect_hand(self, frame):
        """Simple hand detection using skin color and motion"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define skin color range
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create skin mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Filter by area
            if cv2.contourArea(largest_contour) > 1000:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(largest_contour)
                return (x, y, w, h), largest_contour
        
        return None, None
    
    def recognize_gesture(self, frame, bbox):
        """Recognize gesture using the trained model"""
        # Preprocess the frame
        processed_frame = self.preprocess_frame(frame, bbox)
        
        # Make prediction
        predictions = self.model.predict(processed_frame, verbose=0)
        gesture_idx = np.argmax(predictions[0])
        confidence = predictions[0][gesture_idx]
        
        return self.gesture_names[gesture_idx], confidence
    
    def process_frame(self, frame):
        """Process a single frame to detect hands and recognize gestures"""
        # Detect hand
        bbox, contour = self.detect_hand(frame)
        
        gesture = "No hand detected"
        confidence = 0
        
        if bbox is not None:
            x, y, w, h = bbox
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Recognize gesture
            gesture, confidence = self.recognize_gesture(frame, bbox)
            
            # Draw contour
            cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)
        
        # Display gesture text
        cv2.putText(frame, f"Gesture: {gesture} ({confidence:.2f})", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame

def main():
    # Initialize gesture recognizer
    recognizer = HandGestureRecognizer()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Starting hand gesture recognition. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Process frame
        processed_frame = recognizer.process_frame(frame)
        
        # Display the processed frame
        cv2.imshow('Hand Gesture Recognition', processed_frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()