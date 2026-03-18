#include "ai_layer_custom_interface.h"

/* Initialization Function */
void custom_init_MedianVoterLayer(ai_layer* layer)
{
  ai_layer_custom* l = ai_layer_custom_get(layer);
  
  /* Suppress unused variable warnings for init */
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
  
  // Get the 3 input tensors (the 3 TMR branches)
  ai_tensor* t_in0 = ai_layer_get_tensor_in(l, 0);
  ai_tensor* t_in1 = ai_layer_get_tensor_in(l, 1);
  ai_tensor* t_in2 = ai_layer_get_tensor_in(l, 2);
  
  // Get the output tensor
  ai_tensor* t_out0 = ai_layer_get_tensor_out(l, 0);

  // Cast tensor data to float arrays
  ai_float *d_in0 = (ai_float *)t_in0->data;
  ai_float *d_in1 = (ai_float *)t_in1->data;
  ai_float *d_in2 = (ai_float *)t_in2->data;
  ai_float *d_out = (ai_float *)t_out0->data;

  // Get the total number of elements in the feature map
  ai_size size = ai_tensor_get_data_size(t_in0);

  // Iterate through every element to find the median
  for(ai_size i = 0; i < size; i++) {
      float v1 = d_in0[i];
      float v2 = d_in1[i];
      float v3 = d_in2[i];
      
      float median;
      
      // Efficiently find the median of 3 floats
      if (v1 > v2) {
          if (v2 > v3) {
              median = v2;
          } else if (v1 > v3) {
              median = v3;
          } else {
              median = v1;
          }
      } else { // v1 <= v2
          if (v1 > v3) {
              median = v1;
          } else if (v2 > v3) {
              median = v3;
          } else {
              median = v2;
          }
      }
      
      d_out[i] = median;
  }

  ai_layer_custom_release(layer);
}