#include "ai_layer_custom_interface.h"
/* Init: remove the 3rd input tensor */
void custom_init_AverageVoterLayer(ai_layer* layer)
{
  ai_layer_custom* l = ai_layer_custom_get(layer);
  (void)ai_layer_get_tensor_in(l, 0);
  (void)ai_layer_get_tensor_in(l, 1);   // only 2 inputs now
  (void)ai_layer_get_tensor_out(l, 0);
  ai_layer_custom_release(layer);
}

/* Forward: average instead of median */
void custom_forward_AverageVoterLayer(ai_layer* layer)
{
  ai_layer_custom* l = ai_layer_custom_get(layer);

  ai_tensor* t_in0 = ai_layer_get_tensor_in(l, 0);
  ai_tensor* t_in1 = ai_layer_get_tensor_in(l, 1);   // only 2 inputs
  ai_tensor* t_out0 = ai_layer_get_tensor_out(l, 0);

  if (!t_in0 || !t_in1 || !t_out0) {
      ai_layer_custom_release(layer);
      return;
  }

  ai_float *d_in0 = ai_tensor_get_data(t_in0).float32;
  ai_float *d_in1 = ai_tensor_get_data(t_in1).float32;
  ai_float *d_out  = ai_tensor_get_data(t_out0).float32;

  ai_size size = ai_tensor_get_data_size(t_in0);

  for (ai_size i = 0; i < size; i++) {
      d_out[i] = (d_in0[i] + d_in1[i]) * 0.5f;      // average of 2
  }

  ai_layer_custom_release(layer);
}