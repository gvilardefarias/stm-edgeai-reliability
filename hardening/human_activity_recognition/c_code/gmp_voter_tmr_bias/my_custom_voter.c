/**
  ******************************************************************************
  * @file    human_activity_recognition_custom_layers.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-03-02T15:50:16+0100
  * @brief   AI Tool Automatic Code Generator for Custom Layers Implementation
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#include "ai_layer_custom_interface.h"
#define AI_LAYER_CUSTOM_CONV2D_VOTER_ID           (5)

/* USER CODE BEGINS HERE */
/* Helper functions can be defined here if needed */
/* USER CODE ENDS HERE */

/*****************************************************************************/

/* Layer Init Function #0 */
void custom_init_MajorityVoterLayer(ai_layer* layer)
{
  ai_layer_custom* l = ai_layer_custom_get(layer);
  
  /* Suppress unused variable warnings for init */
  (void)ai_layer_get_tensor_in(l, 0);
  (void)ai_layer_get_tensor_in(l, 1);
  (void)ai_layer_get_tensor_in(l, 2);
  (void)ai_layer_get_tensor_out(l, 0);

  switch (l->id)
  {
    case AI_LAYER_CUSTOM_CONV2D_VOTER_ID:
    {
      /* USER CODE BEGINS HERE */
      /* Initialization code goes here if needed (empty for this voter) */
      /* USER CODE ENDS HERE */
    } break;
  
    default: break;
  }

  ai_layer_custom_release(layer);
}
/*****************************************************************************/

/* Layer Forward Function #0 */
void custom_forward_MajorityVoterLayer(ai_layer* layer)
{
  ai_layer_custom* l = ai_layer_custom_get(layer);
  
  // Get the 3 input tensors
  ai_tensor* t_in0 = ai_layer_get_tensor_in(l, 0);
  ai_tensor* t_in1 = ai_layer_get_tensor_in(l, 1);
  ai_tensor* t_in2 = ai_layer_get_tensor_in(l, 2);
  
  // Get the output tensor
  ai_tensor* t_out0 = ai_layer_get_tensor_out(l, 0);

  switch (l->id)
  {
    case AI_LAYER_CUSTOM_CONV2D_VOTER_ID:
    {
      /* USER CODE BEGINS HERE */
      // Cast tensor data to float arrays using the ->data property
      ai_float *d_in0 = (ai_float *)t_in0->data;
      ai_float *d_in1 = (ai_float *)t_in1->data;
      ai_float *d_in2 = (ai_float *)t_in2->data;
      ai_float *d_out = (ai_float *)t_out0->data;

      // Get the total number of elements in the feature map
      ai_size size = ai_tensor_get_data_size(t_in0);

      // Iterate through every float in the tensor and apply strict majority voting
      for(ai_size i = 0; i < size; i++) {
          float v1 = d_in0[i];
          float v2 = d_in1[i]; // Fixed typo from v-4 to v2
          float v3 = d_in2[i];
          
          if (v1 == v2 || v1 == v3) {
              d_out[i] = v1;
          } else if (v2 == v3) {
              d_out[i] = v2;
          } else {
              d_out[i] = v1; // Fallback to branch 1 if all 3 are completely different
          }
      }
      /* USER CODE ENDS HERE */
    } break;
  
    default: break;
  }

  ai_layer_custom_release(layer);
}

#undef AI_LAYER_CUSTOM_CONV2D_VOTER_ID