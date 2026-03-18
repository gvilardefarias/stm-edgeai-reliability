#include "ai_layer_custom_interface.h"

/* Initialization Function */
void custom_init_MedianVoterLayer(ai_layer* layer)
{
  ai_layer_custom* l = ai_layer_custom_get(layer);
  (void)ai_layer_get_tensor_in(l, 0);
  (void)ai_layer_get_tensor_in(l, 1);
  (void)ai_layer_get_tensor_in(l, 2);
  (void)ai_layer_get_tensor_out(l, 0);
  ai_layer_custom_release(layer);
}

/* Forward Pass Function */
void custom_forward_MedianVoterLayer(ai_layer* layer)
{
  ai_layer_custom* l = ai_layer_custom_get(layer);
  
  ai_tensor* t_in0 = ai_layer_get_tensor_in(l, 0);
  ai_tensor* t_in1 = ai_layer_get_tensor_in(l, 1);
  ai_tensor* t_in2 = ai_layer_get_tensor_in(l, 2);
  ai_tensor* t_out0 = ai_layer_get_tensor_out(l, 0);

  // Safety check to prevent segfaults if a tensor is null
  if (!t_in0 || !t_in1 || !t_in2 || !t_out0) {
      ai_layer_custom_release(layer);
      return;
  }

  // CORRECT ST Edge AI Data Access macro
  ai_float *d_in0 = ai_tensor_get_data(t_in0).float32;
  ai_float *d_in1 = ai_tensor_get_data(t_in1).float32;
  ai_float *d_in2 = ai_tensor_get_data(t_in2).float32;
  ai_float *d_out = ai_tensor_get_data(t_out0).float32;

  ai_size size = ai_tensor_get_data_size(t_in0);

  // Find the median
  for(ai_size i = 0; i < size; i++) {
      float v1 = d_in0[i];
      float v2 = d_in1[i];
      float v3 = d_in2[i];
      float median;
      
      if (v1 > v2) {
          if (v2 > v3) { median = v2; } 
          else if (v1 > v3) { median = v3; } 
          else { median = v1; }
      } else { 
          if (v1 > v3) { median = v1; } 
          else if (v2 > v3) { median = v3; } 
          else { median = v2; }
      }
      d_out[i] = median;
  }

  ai_layer_custom_release(layer);
}