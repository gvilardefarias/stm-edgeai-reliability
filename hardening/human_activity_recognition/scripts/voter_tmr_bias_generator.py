import os
import tensorflow as tf
import numpy as np

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 1. Define the Custom Layer for Bias TMR
# We register it so Keras knows how to save/load it properly.
class TMRBiasLayer(tf.keras.layers.Layer):
    def __init__(self, original_bias, **kwargs):
        super(TMRBiasLayer, self).__init__(**kwargs)
        # Store original bias as a numpy array for configuration
        self.original_bias = np.array(original_bias, dtype=np.float32)

    def build(self, input_shape):
        # Create 3 physically separate copies of the bias as layer weights
        # We set trainable=False because we are patching a pre-trained model
        self.b1 = self.add_weight(name='b1', shape=self.original_bias.shape, 
                                  initializer=tf.keras.initializers.Constant(self.original_bias), 
                                  trainable=False)
        self.b2 = self.add_weight(name='b2', shape=self.original_bias.shape, 
                                  initializer=tf.keras.initializers.Constant(self.original_bias), 
                                  trainable=False)
        self.b3 = self.add_weight(name='b3', shape=self.original_bias.shape, 
                                  initializer=tf.keras.initializers.Constant(self.original_bias), 
                                  trainable=False)

    def call(self, inputs, **_):
        # Majority vote on the 3 biases
        b1_eq_b2 = tf.equal(self.b1, self.b2)
        b1_eq_b3 = tf.equal(self.b1, self.b3)
        b2_eq_b3 = tf.equal(self.b2, self.b3)
        
        # If b1 matches b2 or b3, use b1. Else if b2 matches b3, use b2. Else fallback to b1.
        voted_bias = tf.where(tf.logical_or(b1_eq_b2, b1_eq_b3), self.b1, 
                       tf.where(b2_eq_b3, self.b2, self.b1))
        
        # Add the voted bias to the incoming convolution feature map
        return inputs + voted_bias

    def get_config(self):
        # Crucial for ST Edge AI to see the parameters (like Netron)
        base_config = super(TMRBiasLayer, self).get_config()
        config = {"original_bias": self.original_bias.tolist()}
        return dict(list(base_config.items()) + list(config.items()))


def build_bias_tmr_model(original_model_path, target_layer_name, save_path):
    print(f"Loading original model: {original_model_path}")
    model = tf.keras.models.load_model(original_model_path)
    
    input_layer = tf.keras.Input(shape=model.input_shape[1:])
    x = input_layer
    
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
            
        if layer.name == target_layer_name:
            print(f"--- APPLYING BIAS TMR TO: {target_layer_name} ---")
            
            # 1. Extract weights and biases from the original layer
            layer_weights = layer.get_weights()
            if len(layer_weights) < 2:
                print(f"Error: Layer {target_layer_name} does not have a bias.")
                return
            
            conv_weights = layer_weights[0]
            conv_bias = layer_weights[1]
            
            # 2. Rebuild the Conv2D layer WITHOUT the bias
            config = layer.get_config()
            config['use_bias'] = False
            config['name'] = f"{layer.name}_nobias"
            conv_nobias_layer = layer.__class__.from_config(config)
            
            # 3. Pass input through the bias-less convolution and set its weights
            x = conv_nobias_layer(x)
            conv_nobias_layer.set_weights([conv_weights])
            
            # 4. Apply the Custom TMR Bias Voter Layer
            tmr_bias_layer = TMRBiasLayer(original_bias=conv_bias, name=f"{layer.name}_tmr_bias")
            x = tmr_bias_layer(x)
            
            print("--- BIAS SEPARATED AND VOTER ADDED ---")
            
        else:
            config = layer.get_config()
            new_layer = layer.__class__.from_config(config)
            x = new_layer(x)
            new_layer.set_weights(layer.get_weights())
            
    tmr_model = tf.keras.Model(inputs=input_layer, outputs=x)
    print("\nSaving new Bias TMR model to:", save_path)
    tmr_model.save(save_path, save_format='h5')
    print("Done!")
    return tmr_model

if __name__ == "__main__":
    # Ensure these match your environment
    base_model = "/home/apo/stm32ai-modelzoo/human_activity_recognition/st_gmp/ST_pretrainedmodel_public_dataset/WISDM/st_gmp_wl_24/st_gmp_wl_24.keras"
    target = "conv2d" # Changed from "conv2d_bias" to target the layer itself
    output_h5 = "/home/apo/stm32ai-modelzoo/human_activity_recognition/HAR_bias_tmr_1.h5"
    
    if os.path.exists(base_model):
        build_bias_tmr_model(base_model, target, output_h5)
    else:
        print(f"Error: Could not find {base_model}")