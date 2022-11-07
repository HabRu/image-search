import tensorflow as tf
from keras.applications import VGG19
import os

class Model:
    model = None

    @staticmethod
    def get_model():

        if Model.model is None == False:
            print("-not init")
            return Model.model

        print('init')

        if os.path.exists('./data/my_model.h5') == False:
            base_model = VGG19(weights='imagenet')
            base_model.save("./data/my_model.h5")
        else:
            base_model = tf.keras.models.load_model('./data/my_model.h5', compile=False)

        # Read about fc1 here http://cs231n.github.io/convolutional-networks/
        Model.model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

        return Model.model