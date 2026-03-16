import tensorflow as tf

# Use tf.keras.utils instead of keras.saving for older TF versions
@tf.keras.utils.register_keras_serializable(package="Custom", name="MedianVoterLayer")
class MedianVoterLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MedianVoterLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # inputs is a list of 3 tensors: [out_1, out_2, out_3]
        stacked = tf.stack(inputs, axis=-1)
        sorted_vals = tf.math.top_k(stacked, k=2).values
        return sorted_vals[..., 1]

    def get_config(self):
        base_config = super(MedianVoterLayer, self).get_config()
        return dict(list(base_config.items()))