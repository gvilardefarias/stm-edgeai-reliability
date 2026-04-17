import tensorflow as tf

# Use tf.keras.utils instead of keras.saving for older TF versions
@tf.keras.utils.register_keras_serializable(package="Custom", name="AverageVoterLayer")
class AverageVoterLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AverageVoterLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # inputs is a list of 2 tensors: [out_1, out_2]
        return (inputs[0] + inputs[1]) / 2.0   # <-- average instead of median

    def get_config(self):
        return super(AverageVoterLayer, self).get_config()