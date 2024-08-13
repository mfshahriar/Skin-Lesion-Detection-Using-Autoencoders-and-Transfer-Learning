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

job_id = os.getenv('SLURM_JOB_ID')
if not job_id:
    job_id = '9999999'

checkpoint_path = f'/work/cse479/egunde/ModelWeights/E2E_weights_{job_id}.h5'
datapath = '../dataset/Combined_ISIC2020_BCN_HAM/'
csvpath = '../dataset/Combined_ISIC2020_BCN_HAM_metadata.csv'
bweights = '/work/cse479/egunde/ModelWeights/ModelCPs_weights.h5'
slimweights = '/work/cse479/egunde/ModelWeights/AEslim2_weights.h5' 

BATCHSIZE = 32
EPOCHS = 50
PATIENCE = 15
LR = .001

def main():
    malign, benign, augment, val, test = utils.createDisjointDatasets()

    malign = malign.map(lambda x,y: (utils.processImage(x), y)).shuffle(256)
    benign = benign.map(lambda x,y: (utils.processImage(x), y)).shuffle(256)
    augment = augment.map(lambda x,y: (utils.processImage(x, augmented=True), y[0])).shuffle(256)  # through tf pipeline handles diff

    print(f"Malignant instances: {len(malign)}")
    print(f"Benign instances: {len(benign)}")
    print(f"Val size: {len(val)}")
    print(f"Test size: {len(test)}")

    sample_weights = [.15, .6, .25]  # boost appearance of positive samples - still use weighted loss
    train = tf.data.experimental.sample_from_datasets(
        [malign.repeat(), benign.repeat(), augment.repeat()], sample_weights, stop_on_empty_dataset=True
    )

    for i in range(3): # not great but it works
        if i == 0:
            print(f"Malignant weight: {sample_weights[i]}")
        elif i == 1:
            print(f"Benign weight: {sample_weights[i]}")
        elif i == 2:
            print(f"Augmented weight: {sample_weights[i]}")


    train = train.batch(BATCHSIZE).prefetch(tf.data.AUTOTUNE)
    val = val.map(lambda x,y: (utils.processImage(x), y)).batch(BATCHSIZE).prefetch(tf.data.AUTOTUNE)
    test = test.map(lambda x,y: (utils.processImage(x), y)).batch(BATCHSIZE).prefetch(tf.data.AUTOTUNE)


    AE = m.buildAE_slim()
    AE.load_weights(slimweights)
    encoder = AE.layers[0]
        
    classifier = m.buildXClassifier()

    E2E = tf.keras.Sequential([encoder, classifier])
    E2E.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss='binary_crossentropy',
        metrics=['acc', tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), tf.keras.metrics.AUC(curve='PR', from_logits=False)]
    )

    E2E.layers[0].trainable = False        

    Callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE,),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=PATIENCE//3),
        m.unlockFrozenWeights(patience=int(np.sqrt(PATIENCE)), layers=[0])  # layers expects list of layer indices
    ]

    history = E2E.fit(
        train,
        steps_per_epoch=128,
        validation_steps=32,
        epochs=EPOCHS,
        callbacks=Callbacks,
        validation_data=val,
        verbose=1
    ) 

    utils.plot_history_v2(history,0)

    test_results = E2E.evaluate(test, return_dict=True)

    CI = utils.getCI(test_results['acc'], test.cardinality().numpy())
    print(CI)

if __name__ == '__main__':
    main()