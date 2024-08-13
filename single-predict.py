import numpy as np
from PIL import Image
import tensorflow as tf
import os
from PIL import Image
import numpy as np
import utils
import model as m
import matplotlib.pyplot as plt
import tqdm as tqdm
from tqdm.notebook import trange
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# This was generated by ChatGPT
def predict_with_augmentation(image_path, model, num_augmentations=10):
    """
    Predict the class of a single image using a pre-trained model with data augmentation.
    
    Args:
    image_path (str): Path to the image file.
    model (tf.keras.Model): Trained Keras model.
    num_augmentations (int): Number of augmented images to generate for prediction.

    Returns:
    dict: Average prediction results including class and average probability.
    """
    # Load and process the image
    img = Image.open(image_path)
    img = img.resize((380, 380))  # Resize image to match the model's expected input
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = img_array[np.newaxis, ...]  # Add batch dimension

    # Data augmentation generator
    datagen = ImageDataGenerator(
        rotation_range=60,
        brightness_range=(0.6, 1.3),
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=None,
        data_format='channels_last',
    )

    # Create augmented images and predict
    predictions = []
    i = 0
    for x in datagen.flow(img_array, batch_size=1):
        preds = model.predict(x)
        predictions.append(preds)
        i += 1
        if i >= num_augmentations:
            break
    
    # Calculate average prediction
    average_prediction = np.mean(predictions, axis=0)
    predicted_class = 'Malignant' if average_prediction[0][0] > 0.5 else 'Benign'
    probability = average_prediction[0][0]

    return {
        'class': predicted_class,
        'probability': float(probability)
    }

def predict_single_image(image_path, model):
    """
    Predict the class of a single image using a pre-trained model.
    
    Args:
    image_path (str): Path to the image file.
    model (tf.keras.Model): Trained Keras model.

    Returns:
    dict: Prediction results including class and probabilities.
    """
    # Load and process the image
    img = Image.open(image_path)
    img = img.resize((380, 380))  # Correct resizing to match the model's expected input
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = img_array[np.newaxis, ...]  # Add batch dimension

    # Check if the model expects a different input, and adjust accordingly
    if model.input_shape[-1] == 1:  # Model expects grayscale images
        img_array = img_array.mean(axis=3, keepdims=True)
    
    # Make prediction
    predictions = model.predict(img_array)
    
    # Optionally, process the predictions to a more friendly format
    # For example, if binary classification:
    predicted_class = 'Malignant' if predictions[0][0] > 0.5 else 'Benign'
    probability = predictions[0][0]
    
    return {
        'class': predicted_class,
        'probability': probability
    }


job_id = os.getenv('SLURM_JOB_ID')
if not job_id:
    job_id = '9999999'

slimweights = '/work/cse479/egunde/ModelWeights/AEslim2_weights.h5' 
fullweights = '/work/cse479/egunde/ModelWeights/E2E_weights_6241243.h5'
imgpath = '/work/cse479/egunde/test-imgs/1.jpg'

def main():

    AE = m.buildAE_slim()
    AE.load_weights(slimweights)
    encoder = AE.layers[0]
        
    classifier = m.buildXClassifier()

    E2E = tf.keras.Sequential([encoder, classifier])
    E2E.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=['acc', tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), tf.keras.metrics.AUC(curve='PR', from_logits=False)]
    )

    E2E.load_weights(fullweights)
    result = predict_with_augmentation(imgpath, E2E, num_augmentations=50)
    print(result)


if __name__ == '__main__':
    main()