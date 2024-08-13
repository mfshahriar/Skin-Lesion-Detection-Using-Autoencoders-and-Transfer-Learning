# Creating a TensorFlow dataset from our images
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Rescale Dimensions
WIDTH = 380
HEIGHT = 380
# Center Crop - proportion to keep
CENTER_FRACTION = .9


# function to map the img file to the scaled, cropped tensor representation we want
def processImage(img, augmented=False):
    if augmented:  # passed as a tensor already
        img = tf.cast(img, dtype=tf.float16) / 255
        return img[0]
    
    # from directory -> read the jpg and convert it to a tf tensor
    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, channels=3)
    # then crop and rescale
    img = tf.image.central_crop(img, CENTER_FRACTION) 
    img = tf.image.resize(img, (WIDTH, HEIGHT))
    img = tf.image.convert_image_dtype(img, tf.float16)
    img = img / 255
    return img


def plotHistory(history,j):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.savefig(f'modelResultsPlot{j}.png', bbox_inches='tight')
    plt.show()

def plot_history_v2(history,j):
    for key in history.history.keys():
        plt.figure(figsize=(8, 5))
        plt.plot(history.history[key], label=key)
        plt.title(f'Training {key}')
        plt.xlabel('Epochs')
        plt.ylabel(key)
        plt.legend()
        plt.savefig(f'modelResultsPlot{j}.png', bbox_inches='tight')
        plt.show()
    
    
def getCI(accuracy, test_size):
    offset = 1.96 * np.sqrt(accuracy * (1-accuracy) / test_size)  # 95% CI
    CI = (accuracy-offset, accuracy+offset)
    return CI
    
    
def splitDataset(dataset, train_ratio=.8):
    num_items = dataset.cardinality().numpy() # assumes tf dataset
    train_size = int(num_items * train_ratio)
    
    train_dataset = dataset.take(train_size)
    validation_dataset = dataset.skip(train_size)
    
    train_dataset = train_dataset.cache()
    validation_dataset = validation_dataset.cache()
    
    return train_dataset, validation_dataset


def get_data_as_df(
    csvpath='../dataset/Combined_ISIC2020_BCN_HAM_metadata.csv', 
    datapath='../dataset/Combined_ISIC2020_BCN_HAM/'):  
    
    labels_df = pd.read_csv(csvpath)
   
    labels_df['filename'] = labels_df['image_name'].map(lambda x: os.path.join(datapath, x)) + '.jpg'
    labels_df = labels_df[['filename', 'target']]
    
    malignant_images = labels_df[ (labels_df['target'] == 1) ]
    benign_images = labels_df[ (labels_df['target'] == 0) ]
    
    return malignant_images, benign_images


# should remove the generator init to a separate function - TODO - fine for now though :p
def createDisjointDatasets():  #    
    malignant_images, benign_images = get_data_as_df()

    val_size = 1024
    test_size = 2048

    train_start = (val_size // 2) + (test_size // 2)
    val_end = (val_size // 2)
    
    mal_train = malignant_images[train_start:]
    ben_train = benign_images[train_start:]
    
    mal_val = malignant_images[:val_end]
    ben_val = benign_images[:val_end]
    val = pd.concat([ben_val,mal_val])
    
    mal_test = malignant_images[val_end:train_start]
    ben_test = benign_images[val_end:train_start]
    test = pd.concat([mal_test,ben_test])
    
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=60,
        brightness_range=(0.6, 1.3),
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=None,
        data_format='channels_last',
    )
    
    def gen():
        generator = image_generator.flow_from_dataframe(
            mal_train,
            x_col='filename',
            y_col='target',
            target_size=(380,380),
            color_mode='rgb',
            batch_size=1,
            class_mode='raw',
            shuffle=True,
            seed=2025,
            interpolation='bilinear'
        )
        for batch_image, batch_label in generator:
            yield batch_image, batch_label
            
    augmented_ds =  tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None, 380, 380, 3), dtype=tf.float16),
            tf.TensorSpec(shape=(None,), dtype=tf.int64))
    )
    
    valSet = tf.data.Dataset.from_tensor_slices((val['filename'], val['target']))
    testSet = tf.data.Dataset.from_tensor_slices((test['filename'], test['target']))
    malignantSet = tf.data.Dataset.from_tensor_slices((mal_train['filename'], mal_train['target']))
    
    benignSet = tf.data.Dataset.from_tensor_slices((ben_train['filename'], ben_train['target']))
    
    
    return malignantSet, benignSet, augmented_ds, valSet, testSet



     # adds augmented samples of malignant class images to a given dataset #
# assumes tf objects

# need to edit to give both train and test images - for autoencoders - 
def createUnsupervisedDataset(path): 
    PATH = path
    image_paths = [os.path.join(PATH, file) for file in os.listdir(PATH)]
    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    image_dataset = image_dataset.shuffle(128)
    image_dataset = image_dataset.map(processImage)
    image_dataset = image_dataset.map(lambda x: (x, x))
    return image_dataset
    
    
    
    
    
    