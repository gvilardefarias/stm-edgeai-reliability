import tensorflow as tf
import numpy as np

# Use tf.keras.utils instead of keras.saving for your TF version
@tf.keras.utils.register_keras_serializable(package="Custom", name="TMRBiasLayer")
class TMRBiasLayer(tf.keras.layers.Layer):
    def __init__(self, original_bias=None, **kwargs):
        super(TMRBiasLayer, self).__init__(**kwargs)
        # Store original bias as a numpy array for configuration
        if original_bias is not None:
            self.original_bias = np.array(original_bias, dtype=np.float32)
        else:
            self.original_bias = None

    def build(self, input_shape):
        shape = self.original_bias.shape if self.original_bias is not None else (input_shape[-1],)
        init_val = self.original_bias if self.original_bias is not None else np.zeros(shape)
        
        # Inject epsilons to bypass STEdgeAI deduplication
        init_val_1 = (init_val + 1e-6).astype(np.float32) if self.original_bias is not None else init_val
        init_val_2 = (init_val + 2e-6).astype(np.float32) if self.original_bias is not None else init_val
        init_val_3 = (init_val + 3e-6).astype(np.float32) if self.original_bias is not None else init_val
        
        self.b1 = self.add_weight(name='b1', shape=shape,
                                  initializer=tf.keras.initializers.Constant(init_val_1), 
                                  trainable=False)
        self.b2 = self.add_weight(name='b2', shape=shape, 
                                  initializer=tf.keras.initializers.Constant(init_val_2), 
                                  trainable=False)
        self.b3 = self.add_weight(name='b3', shape=shape, 
                                  initializer=tf.keras.initializers.Constant(init_val_3), 
                                  trainable=False)
        super().build(input_shape)

    def call(self, inputs, **_):
        # Majority vote on the 3 biases
        b1_eq_b2 = tf.equal(self.b1, self.b2)
        b1_eq_b3 = tf.equal(self.b1, self.b3)
        b2_eq_b3 = tf.equal(self.b2, self.b3)
        
        voted_bias = tf.where(tf.logical_or(b1_eq_b2, b1_eq_b3), self.b1, 
                       tf.where(b2_eq_b3, self.b2, self.b1))
        
        return inputs + voted_bias

    def get_config(self):
            # Explicitly DO NOT include original_bias here to keep the Keras JSON config clean
            # and prevent ST Edge AI from trying to parse it as a tensor.
            base_config = super(TMRBiasLayer, self).get_config()
            return dict(list(base_config.items()))