import tensorflow as tf
import numpy as np
import matplotlib as plt
import utils
import model as m

LR = 0.0001
EPOCHS = 50
PATIENCE = 10
BATCHSIZE = 32

p1 = '/work/cse479/lukew/ISIC2020'
p2 = '/work/cse479/lukew/ISIC_test/ISIC_2020_Test_Input'
checkpoint_path = '/home/cse479/lukew/Project/AEslim2_weights.h5'

AE = m.buildAE_slim()

AE.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss='mse'
)

# create the dataset to pretrain on 
ISIC_ds1 = utils.createUnsupervisedDataset(p1)
ISIC_ds2 = utils.createUnsupervisedDataset(p2)
# then we just merge them
ds = ISIC_ds1.concatenate(ISIC_ds2)

train_ds, val_ds = utils.splitDataset(ds, .80)
train_ds = train_ds.batch(BATCHSIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCHSIZE).prefetch(tf.data.AUTOTUNE)

Callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_best_only=True, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE,),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=PATIENCE//3),
]

history = AE.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=Callbacks,
    validation_data=val_ds,
)

utils.plotHistory(history)

