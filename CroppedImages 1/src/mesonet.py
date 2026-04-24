import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

IMG_SIZE = 128

class Meso4:
    def __init__(self, lr=0.001):
        self.model = self.build_model()
        self.model.compile(
            optimizer=Adam(lr),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

    def build_model(self):
        x = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

        x1 = Conv2D(8, (3,3), padding="same", activation="relu")(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D((2,2), padding="same")(x1)

        x2 = Conv2D(8, (5,5), padding="same", activation="relu")(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D((2,2), padding="same")(x2)

        x3 = Conv2D(16, (5,5), padding="same", activation="relu")(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D((2,2), padding="same")(x3)

        x4 = Conv2D(16, (5,5), padding="same", activation="relu")(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D((4,4), padding="same")(x4)

        y = Flatten()(x4)
        y = Dropout(0.3)(y)
        y = Dense(16)(y)
        y = LeakyReLU(0.1)(y)
        y = Dropout(0.3)(y)
        y = Dense(1, activation="sigmoid")(y)

        return Model(inputs=x, outputs=y)
