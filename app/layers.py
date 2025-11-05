# Custom L1 Distance layer module 
# Required for loading the custom Siamese model

# Import dependencies
import tensorflow as tf
from tensorflow.keras.layers import Layer

@tf.keras.utils.register_keras_serializable()
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, inputs):
        # Ensure inputs are tensors
        input_embedding, validation_embedding = tf.convert_to_tensor(inputs[0]), tf.convert_to_tensor(inputs[1])
        return tf.math.abs(input_embedding - validation_embedding)
