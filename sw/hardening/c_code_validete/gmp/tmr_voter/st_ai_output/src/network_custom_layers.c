

/**
  ******************************************************************************
  * @file    network_custom_layers.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-04-01T14:54:29+0200
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
#define AI_LAYER_CUSTOM_CONV2D_1_VOTER_ID           (7)

/* USER CODE BEGINS HERE */

/* USER CODE ENDS HERE */


/*****************************************************************************/

/* Layer Init Function #0 */
void custom_init_MajorityVoterLayer(ai_layer* layer)
{
  ai_layer_custom* l = ai_layer_custom_get(layer);
  ai_tensor* t_in0 = ai_layer_get_tensor_in(l, 0);
ai_tensor* t_in1 = ai_layer_get_tensor_in(l, 1);
ai_tensor* t_in2 = ai_layer_get_tensor_in(l, 2);

  ai_tensor* t_out0 = ai_layer_get_tensor_out(l, 0);

  
  switch (l->id)
  {
    case AI_LAYER_CUSTOM_CONV2D_1_VOTER_ID:
    {
      
      /* USER CODE BEGINS HERE */

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
  ai_tensor* t_in0 = ai_layer_get_tensor_in(l, 0);
ai_tensor* t_in1 = ai_layer_get_tensor_in(l, 1);
ai_tensor* t_in2 = ai_layer_get_tensor_in(l, 2);

  ai_tensor* t_out0 = ai_layer_get_tensor_out(l, 0);

  
  
  switch (l->id)
  {
    case AI_LAYER_CUSTOM_CONV2D_1_VOTER_ID:
    {
      
      /* USER CODE BEGINS HERE */

      /* USER CODE ENDS HERE */

    } break;
  
    default: break;
  }


  ai_layer_custom_release(layer);
}


#undef AI_LAYER_CUSTOM_CONV2D_1_VOTER_ID
