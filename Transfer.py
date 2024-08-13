import tensorflow as tf
from tensorflow.keras import applications as app
import pandas as pd
import model as m
import utils
import pickle


checkpoint_path = '/home/cse479/lukew/Project/ModelWeights/TResNet50V2_weights.h5'
datapath = '/work/cse479/lukew/ISIC2020'
csvpath = '/work/cse479/lukew/ISIC_2020_Training_GroundTruth.csv'
logpath = '/home/cse479/lukew/Project/trainLog.csv'
historypath = '/home/cse479/lukew/Project/TResNet50V2_trainHistory.pkl'

EPOCHS = 20
STEPS = 600
PATIENCE = 6
LR = .001
BATCHSIZE = 32

# XCEweights = 'home/cse479/lukew/Project/ModelWeights/TXception_weights.h5'
# Res50weights = '/home/cse479/lukew/Project/ModelWeights/TRes50_weights.h5'
# AE_weights = '/home/cse479/lukew/Project/ModelWeights/AEslim2_weights.h5'
# Xception by default takes 299x299x3 images

# TEffNetB4 = tf.keras.applications.EfficientNetB4(
#     include_top=False,
#     weights='imagenet',
#     input_shape=(380,380,3),
#     pooling='max'
# )

ResNet = app.ResNet50V2(
    include_top=False,
    weights='imagenet',
    input_shape=(380,380,3),
    pooling='max',
)

# Xception = app.Xception(
#     include_top=False,
#     weights='imagenet',
#     input_shape=(380,380,3),
#     pooling='max',
# )

XClassifier = m.buildXClassifier()

resnet_classweights = {0:.75, 1:18}  # weight to apply to each cluster
effnet_classweights = {0:1, 1:14}  # adjusted based on performance


# AE = m.buildAE_slim()
# AE.load_weights(AE_weights)
# encoder = AE.layers[0]
# classifier = m.buildXClassifier()

# model = tf.keras.Sequential([
#     encoder,
#     tf.keras.layers.Reshape((1,-1)),
#     tf.keras.layers.Flatten(),
#     classifier
# ])

model = tf.keras.Sequential([ResNet, XClassifier])

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR,),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='Precision'), 
        tf.keras.metrics.Recall(name='Recall'), 
        tf.keras.metrics.AUC(curve='PR', name='PR-AUC'), 
        tf.keras.metrics.AUC(curve='ROC', name='ROC-AUC'),
    ]
)

# model.load_weights(XCEweights)
model.layers[0].trainable = False
model.summary()

malign, benign, augment, val, test = utils.createDisjointDatasets()

malign = malign.map(lambda x,y: (utils.processImage(x), y)).shuffle(256)
benign = benign.map(lambda x,y: (utils.processImage(x), y)).shuffle(256)
augment = augment.map(lambda x,y: (utils.processImage(x, augmented=True), y[0])).shuffle(256)

sample_weights = [.05, .85, .1]  # boost appearance of positive samples - still use weighted loss
train = tf.data.experimental.sample_from_datasets(
    [malign.repeat(), benign, augment.repeat()], sample_weights, stop_on_empty_dataset=True
)

train = train.batch(BATCHSIZE).prefetch(tf.data.AUTOTUNE)
val = val.map(lambda x,y: (utils.processImage(x), y)).batch(BATCHSIZE).prefetch(tf.data.AUTOTUNE)
test = test.map(lambda x,y: (utils.processImage(x), y)).batch(BATCHSIZE).prefetch(tf.data.AUTOTUNE)


CALLBACKS = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, save_best_only=True, monitor='val_loss', verbose=1),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=PATIENCE,),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', patience=PATIENCE//3),
    tf.keras.callbacks.CSVLogger(
        logpath, separator=',',append=False
    )
]


history = model.fit(
    train.take(STEPS),
    epochs=EPOCHS,
    callbacks=CALLBACKS,
    validation_data=val,
    class_weight=resnet_classweights)

with open(historypath, 'wb') as file:
    pickle.dump(history.history, file)
    
results = model.evaluate(test, return_dict=True, verbose=1)


