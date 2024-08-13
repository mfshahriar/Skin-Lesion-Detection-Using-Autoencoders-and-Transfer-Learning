# testing the idea for a malignant and benign split set achieving better results than a single AE

# so we'll take the AE we previously trained and retrain a separate instance on
# both Malignant and Benign set then we compare

# also worth exploring is training the autoencoders in parallel with the classification task. Which is where unfreezing weights comes in later, I feel that pretraining to an extent is beneficial for time purposes, but there is no reason that pretraining would be required

# if this fails I am doing transfer learning lmao
# well the auto encoders work, but they need to bottleneck small enough.
# further, generalization doesn't follow, they REQUIRE fine-tuning with the classification task, since it isn't dealing probabilistically I doubt the validity of this approach for this reasoning. It may succeed simply because deep networks are powerful
# classification still an extreme challenge, need to report f1

import tensorflow as tf
import utils
import model as m
import numpy as np
import pandas as pd

EPOCHS=100
PATIENCE=25
LR=.001


malignant_ds, benign_ds = utils.createDisjointDatasets()

# AE_slim bottlenecks to 

def main():
    checkpoint_path = '/home/cse479/lukew/Project/malignantAE_weights.h5'
    malignantAE = m.createAE_slim()
    
    malignantAE.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss=tf.keras.losses.MSE(),
        metrics=['acc']
    )
    
    Callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=PATIENCE),
    ]
    # the AE's are pretraining, we just let them run and take them when they start to plateau
    malignantAE.fit(
        malignant_ds,
        epochs=EPOCHS,
        callbacks=Callbacks
    )
    
    
if __name__=='__main__':
    main()


