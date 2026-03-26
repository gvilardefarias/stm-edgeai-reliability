import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="Custom", name="AdaptiveClipper")
class AdaptiveClipper(tf.keras.layers.Layer):
    """Custom Keras layer implementing an Adaptive Clipper (capped ReLU / HardTanH).

    This layer clips the input to the range [0, max_value], effectively acting as
    a ReLU with an upper bound. The max_value is determined per-layer during the
    profiling phase of the hardening pipeline, based on the maximum activation
    observed over representative data.

    This is equivalent to: output = min(max(input, 0), max_value)
    """

    def __init__(self, max_value=6.0, **kwargs):
        super(AdaptiveClipper, self).__init__(**kwargs)
        self.max_value = float(max_value)

    def build(self, input_shape):
        # Store max_value as a non-trainable weight so it is exported
        # into the model file and visible to ST Edge AI as a weight tensor.
        self.clip_max = self.add_weight(
            name='clip_max',
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.max_value),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, inputs, **_):
        # Capped ReLU: clamp to [0, clip_max]
        return tf.clip_by_value(inputs, 0.0, self.clip_max)

    def get_config(self):
        base_config = super(AdaptiveClipper, self).get_config()
        base_config.update({'max_value': self.max_value})
        return base_config
