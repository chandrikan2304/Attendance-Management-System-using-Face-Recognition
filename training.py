import cv2
import os
import numpy as np
from PIL import Image

# Initialize the recognizer and detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    # Filter image files
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    faceSamples = []
    Ids = []
    
    for imagePath in imagePaths:
        try:
            # Convert image to grayscale
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')
            Id = int(os.path.split(imagePath)[-1].split(".")[1])  # Extract ID from file name

            # Detect faces in the image
            faces = detector.detectMultiScale(imageNp)
            for (x, y, w, h) in faces:
                faceSamples.append(imageNp[y:y+h, x:x+w])
                Ids.append(Id)
        except Exception as e:
            print(f"Error processing file {imagePath}: {e}")
            continue
    
    return faceSamples, Ids

# Train the recognizer
faces, Ids = getImagesAndLabels('TrainingImage')
if len(faces) > 0 and len(Ids) > 0:
    recognizer.train(faces, np.array(Ids))
    recognizer.save('TrainingImageLabel/trainner.yml')
    print("Training complete. Model saved to 'TrainingImageLabel/trainner.yml'")
else:
    print("No valid training data found. Ensure the TrainingImage folder contains properly formatted images.")
