/**
  ******************************************************************************
  * @file    adaptive_clipper_impl.c
  * @brief   ST Edge AI Custom Layer Implementation for AdaptiveClipper
  *
  * Implements a capped ReLU (HardTanH) for deployment on STM32.
  * For each element: output = min(max(input, 0), max_value)
  *
  * The max_value is stored as a single-element weight tensor exported
  * from the Keras AdaptiveClipper layer.
  ******************************************************************************
  */

#include "ai_layer_custom_interface.h"

/* ---- Initialization ---- */
void custom_init_AdaptiveClipper(ai_layer* layer)
{
  ai_layer_custom* l = ai_layer_custom_get(layer);
  (void)ai_layer_get_tensor_in(l, 0);
  (void)ai_layer_get_tensor_out(l, 0);
  ai_layer_custom_release(layer);
}

/* ---- Forward Pass ---- */
void custom_forward_AdaptiveClipper(ai_layer* layer)
{
  ai_layer_custom* l = ai_layer_custom_get(layer);

  ai_tensor* t_in0  = ai_layer_get_tensor_in(l, 0);
  ai_tensor* t_out0 = ai_layer_get_tensor_out(l, 0);

  /* Weight tensor 0 holds the clip_max scalar exported by the Keras layer */
  ai_tensor* t_clip_max = ai_layer_get_tensor_weights(l, 0);

  if (!t_in0 || !t_out0 || !t_clip_max) {
      ai_layer_custom_release(layer);
      return;
  }

  ai_float *d_in  = ai_tensor_get_data(t_in0).float32;
  ai_float *d_out = ai_tensor_get_data(t_out0).float32;

  /* Retrieve the scalar ceiling value */
  ai_float clip_max = ai_tensor_get_data(t_clip_max).float32[0];

  ai_size size = ai_tensor_get_data_size(t_in0);

  /* Capped ReLU: output = min(max(input, 0), clip_max) */
  for (ai_size i = 0; i < size; i++) {
      ai_float val = d_in[i];

      if (val < 0.0f) {
          val = 0.0f;
      } else if (val > clip_max) {
          val = clip_max;
      }

      d_out[i] = val;
  }

  ai_layer_custom_release(layer);
}
