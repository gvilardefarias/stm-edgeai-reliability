import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="Custom", name="MajorityVoterLayer")
class MajorityVoterLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MajorityVoterLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Expecting a list of 3 tensors
        o1, o2, o3 = inputs
        
        # Exact equality check in Python
        o1_eq_o2 = tf.equal(o1, o2)
        o1_eq_o3 = tf.equal(o1, o3)
        o2_eq_o3 = tf.equal(o2, o3)
        
        # If (o1 == o2) or (o1 == o3) -> o1
        # Else if (o2 == o3) -> o2
        # Else -> o1 (fallback)
        res = tf.where(tf.logical_or(o1_eq_o2, o1_eq_o3), o1, 
                       tf.where(o2_eq_o3, o2, o1))
        return res

    def get_config(self):
        base_config = super(MajorityVoterLayer, self).get_config()
        return dict(list(base_config.items()))