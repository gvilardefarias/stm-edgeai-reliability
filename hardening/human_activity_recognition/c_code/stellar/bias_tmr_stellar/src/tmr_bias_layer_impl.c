#include "ai_layer_custom_interface.h"
#define AI_LAYER_CUSTOM_CONV2D_TMR_BIAS_ID (3)

void custom_init_TMRBiasLayer(ai_layer* layer)
{
  ai_layer_custom* l = ai_layer_custom_get(layer);
  (void)ai_layer_get_tensor_in(l, 0);
  (void)ai_layer_get_tensor_out(l, 0);
  ai_layer_custom_release(layer);
}

void custom_forward_TMRBiasLayer(ai_layer* layer)
{
  ai_layer_custom* l = ai_layer_custom_get(layer);
  ai_tensor* t_in0 = ai_layer_get_tensor_in(l, 0);
  ai_tensor* t_out0 = ai_layer_get_tensor_out(l, 0);

  ai_tensor* t_weight0 = ai_layer_get_tensor_weights(l, 0);
  ai_tensor* t_weight1 = ai_layer_get_tensor_weights(l, 1);
  ai_tensor* t_weight2 = ai_layer_get_tensor_weights(l, 2);

  if (l->id == AI_LAYER_CUSTOM_CONV2D_TMR_BIAS_ID)
  {
      ai_float *d_in  = ai_tensor_get_data(t_in0).float32;
      ai_float *d_out = ai_tensor_get_data(t_out0).float32;

      ai_float *b1 = ai_tensor_get_data(t_weight0).float32;
      ai_float *b2 = ai_tensor_get_data(t_weight1).float32;
      ai_float *b3 = ai_tensor_get_data(t_weight2).float32;

      ai_size total_size = ai_tensor_get_data_size(t_in0);
      ai_size channels = ai_tensor_get_data_size(t_weight0);
      ai_size spatial_size = total_size / channels;

      ai_float voted_bias[channels]; 
      
      for(int c = 0; c < channels; c++) {
          if (b1[c] == b2[c] || b1[c] == b3[c]) {
              voted_bias[c] = b1[c];
          } else if (b2[c] == b3[c]) {
              voted_bias[c] = b2[c];
          } else {
              voted_bias[c] = b1[c];
          }
      }

      for(ai_size i = 0; i < spatial_size; i++) {
          for(int c = 0; c < channels; c++) {
              d_out[i * channels + c] = d_in[i * channels + c] + voted_bias[c];
          }
      }
  }
  ai_layer_custom_release(layer);
}
#undef AI_LAYER_CUSTOM_CONV2D_TMR_BIAS_ID
