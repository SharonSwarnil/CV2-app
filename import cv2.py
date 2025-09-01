# train_model.py
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Create synthetic training data since we don't have a real dataset
def create_synthetic_data(num_samples=1000, img_size=(64, 64)):
    gestures = ['open_palm', 'fist', 'peace_sign', 'thumbs_up']
    X = []
    y = []
    
    for gesture_idx, gesture in enumerate(gestures):
        for i in range(num_samples):
            # Create a blank image
            img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
            
            # Draw different shapes based on the gesture
            if gesture == 'open_palm':
                # Draw a circle (palm) with lines (fingers)
                center = (img_size[1]//2, img_size[0]//2)
                cv2.circle(img, center, 15, (255, 255, 255), -1)
                for angle in range(0, 360, 72):
                    end_x = center[0] + int(20 * np.cos(np.radians(angle)))
                    end_y = center[1] + int(20 * np.sin(np.radians(angle)))
                    cv2.line(img, center, (end_x, end_y), (255, 255, 255), 3)
            
            elif gesture == 'fist':
                # Draw a circle (fist)
                center = (img_size[1]//2, img_size[0]//2)
                cv2.circle(img, center, 20, (255, 255, 255), -1)
            
            elif gesture == 'peace_sign':
                # Draw a circle (palm) with two prominent lines
                center = (img_size[1]//2, img_size[0]//2)
                cv2.circle(img, center, 15, (255, 255, 255), -1)
                for angle in [45, 135]:
                    end_x = center[0] + int(25 * np.cos(np.radians(angle)))
                    end_y = center[1] + int(25 * np.sin(np.radians(angle)))
                    cv2.line(img, center, (end_x, end_y), (255, 255, 255), 4)
            
            elif gesture == 'thumbs_up':
                # Draw a circle with one prominent line for thumb
                center = (img_size[1]//2, img_size[0]//2)
                cv2.circle(img, center, 15, (255, 255, 255), -1)
                end_x = center[0] + 20
                end_y = center[1] - 25
                cv2.line(img, center, (end_x, end_y), (255, 255, 255), 4)
            
            # Add some noise and variations
            noise = np.random.randint(0, 50, (img_size[0], img_size[1], 3), dtype=np.uint8)
            img = cv2.add(img, noise)
            
            # Apply slight affine transformation
            rows, cols = img.shape[:2]
            M = np.float32([[1, 0, np.random.randint(-5, 5)], [0, 1, np.random.randint(-5, 5)]])
            img = cv2.warpAffine(img, M, (cols, rows))
            
            X.append(img)
            y.append(gesture_idx)
    
    return np.array(X), np.array(y)

def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def main():
    # Create synthetic data
    print("Creating synthetic training data...")
    X, y = create_synthetic_data(num_samples=500)
    
    # Normalize pixel values
    X = X.astype('float32') / 255.0
    
    # Convert labels to categorical
    y = to_categorical(y, 4)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create model
    input_shape = X_train.shape[1:]
    model = create_model(input_shape, 4)
    
    # Define callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5)
    ]
    
    # Train model
    print("Training model...")
    history = model.fit(X_train, y_train,
                        epochs=50,
                        batch_size=32,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks,
                        verbose=1)
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {accuracy*100:.2f}%")
    
    # Save model
    os.makedirs('model', exist_ok=True)
    model.save('model/gesture_model.h5')
    print("Model saved as 'model/gesture_model.h5'")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved as 'training_history.png'")

if __name__ == "__main__":
    main()