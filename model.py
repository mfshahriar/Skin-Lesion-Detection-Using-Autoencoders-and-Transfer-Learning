import tensorflow as tf
import numpy as np
from keras import backend as K


class ResBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, stride=1):
        super().__init__()
        self.stride = stride

        # Both self.conv1 and self.down_conv layers downsample the input when stride != 1
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            padding="same")

        if self.stride != 1:
            self.down_conv = tf.keras.layers.Conv2D(filters=filter_num,
                                                    kernel_size=(1, 1),
                                                    strides=stride,
                                                    padding="same")
            self.down_bn = tf.keras.layers.BatchNormalization()
            
    def __call__(self, x, is_training=None):
        identity = x
        is_training = tf.keras.backend.learning_phase() if is_training is None else is_training
        
        if self.stride != 1:
            identity = self.down_conv(identity)
            identity = self.down_bn(identity, training=is_training)

        x = self.bn1(x, training=is_training)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        
        
        x = self.bn2(x, training=is_training)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        return x + identity
    

def buildAE_slim():
    conv_encoder = tf.keras.Sequential([
        tf.keras.layers.Conv2D(input_shape=[380,380,3], filters=32, kernel_size=1, strides=1, padding='same'),
        ResBlock(32),
        ResBlock(32),
        ResBlock(64,2),
        ResBlock(64),
        ResBlock(128,2),
        ResBlock(128),
        ResBlock(256,2),
        ResBlock(256),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=1, padding='same', activation=tf.nn.swish),
        tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1, padding='same', activation=tf.nn.swish),
        tf.keras.layers.MaxPool2D()
    ])

    conv_decoder = tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='valid'),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='valid', use_bias=0, activation=tf.nn.swish),
        tf.keras.layers.Conv2DTranspose(filters=64,kernel_size=3, strides=2, padding='valid'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2DTranspose(filters=64,kernel_size=5, strides=2, padding='valid'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, padding='valid', use_bias=0, activation=tf.nn.swish), 
        tf.keras.layers.Conv2DTranspose(filters=32,kernel_size=7, strides=2, padding='valid'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(filters=16, kernel_size=7, strides=1, padding='valid', use_bias=0, activation=tf.nn.swish),
        tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=5, strides=2, padding='valid'),
        tf.keras.layers.Conv2D(filters=8, kernel_size=5, strides=2, padding='valid', use_bias=0, activation=tf.nn.swish),
        tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=4, strides=2, padding='valid'),
    ])

    model = tf.keras.Sequential([conv_encoder, conv_decoder])
    return model
    

def buildClassifier_slim():
    
    inputs = tf.keras.Input(shape=(24,24,8))

    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(400, activation='relu', kernel_regularizer='L1L2')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(.2)(x)
    x = tf.keras.layers.Dense(300, activation='relu', kernel_regularizer='L2')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(.2)(x)
    x = tf.keras.layers.Dense(300, activation='relu', kernel_regularizer='L2')(x)
    x = tf.keras.layers.Dense(100, activation='relu', kernel_regularizer='L1L2')(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    classifier = tf.keras.Model(inputs=inputs, outputs=out, name='flat_classifier')
    return classifier


def buildXClassifier():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1500, activation='relu'),
        tf.keras.layers.Dropout(.4),
        tf.keras.layers.Dense(1250, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(.3),
        tf.keras.layers.Dense(750, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(.3),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    
    return model

# unlock weights callback, after a patience period resume training the encoder.
class unlockFrozenWeights(tf.keras.callbacks.Callback):
    def __init__(self, patience, layers):
        super().__init__()
        self.patience = patience  # how many epochs to wait
        self.layers = layers  # layers to unlock
        
    def on_train_begin(self, logs=None):
        self.wait = 0  # 
        self.whenUnlocked = 0
    
    def on_epoch_end(self, epoch, logs=None):
        self.wait += 1  # not monitoring, just a raw patience period
        if self.wait >= self.patience:
            self.whenUnlocked = epoch  # unlocked after that epoch concluded
            for idx in self.layers:
                self.model.layers[idx].trainable = True

    def on_train_end(self, logs=None):
        print(f'Locked Model weights unlocked after epoch: {self.whenUnlocked}')
    
    
# the following is not my code
# """"https://github.com/umbertogriffo/focal-loss-keras/blob/master/src/loss_function/losses.py""""
def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.

      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)

      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.

    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)

    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss

    return binary_focal_loss_fixed
