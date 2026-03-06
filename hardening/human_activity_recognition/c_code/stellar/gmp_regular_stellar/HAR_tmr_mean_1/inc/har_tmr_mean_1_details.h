/**
  ******************************************************************************
  * @file    har_tmr_mean_1.h
  * @date    2026-03-05T11:45:17+0100
  * @brief   ST.AI Tool Automatic Code Generator for Embedded NN computing
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
#ifndef STAI_HAR_TMR_MEAN_1_DETAILS_H
#define STAI_HAR_TMR_MEAN_1_DETAILS_H

#include "stai.h"
#include "layers.h"

const stai_network_details g_har_tmr_mean_1_details = {
  .tensors = (stai_tensor[7]) {
   { .size_bytes = 288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (int32_t[4]){1, 24, 3, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "serving_default_input_layer0_output" },
   { .size_bytes = 3840, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (int32_t[4]){1, 20, 3, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv2d_0_output" },
   { .size_bytes = 3840, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (int32_t[4]){1, 20, 3, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "eltwise_1_output" },
   { .size_bytes = 3840, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (int32_t[4]){1, 20, 3, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "eltwise_2_output" },
   { .size_bytes = 64, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (int32_t[4]){1, 1, 1, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv2d_6_output" },
   { .size_bytes = 16, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (int32_t[2]){1, 4}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "gemm_8_output" },
   { .size_bytes = 16, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (int32_t[2]){1, 4}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "nl_9_output" }
  },
  .nodes = (stai_node_details[6]){
    {.id = 0, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (int32_t[1]){0}}, .output_tensors = {1, (int32_t[1]){1}} }, /* conv2d_0 */
    {.id = 1, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {2, (int32_t[2]){1, 1}}, .output_tensors = {1, (int32_t[1]){2}} }, /* eltwise_1 */
    {.id = 2, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {2, (int32_t[2]){2, 1}}, .output_tensors = {1, (int32_t[1]){3}} }, /* eltwise_2 */
    {.id = 7, .type = AI_LAYER_OPTIMIZED_CONV2D_TYPE, .input_tensors = {1, (int32_t[1]){3}}, .output_tensors = {1, (int32_t[1]){4}} }, /* conv2d_6 */
    {.id = 8, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (int32_t[1]){4}}, .output_tensors = {1, (int32_t[1]){5}} }, /* gemm_8 */
    {.id = 9, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (int32_t[1]){5}}, .output_tensors = {1, (int32_t[1]){6}} } /* nl_9 */
  },
  .n_nodes = 6
};
#endif