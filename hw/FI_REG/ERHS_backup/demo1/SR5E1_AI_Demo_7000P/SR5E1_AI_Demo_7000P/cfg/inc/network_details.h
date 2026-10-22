/**
  ******************************************************************************
  * @file    network.h
  * @date    2026-10-12T12:12:02+0200
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
#ifndef STAI_NETWORK_DETAILS_H
#define STAI_NETWORK_DETAILS_H

#include "stai.h"
#include "layers.h"

const stai_network_details g_network_details = {
  .tensors = (const stai_tensor[6]) {
   { .size_bytes = 320, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 20, 4}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "input_0_output" },
   { .size_bytes = 2560, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 20, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv1d_1_output" },
   { .size_bytes = 2560, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 20, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "activation_1_output" },
   { .size_bytes = 1280, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 10, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "max_pooling1d_1_output" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "lstm_1_output0" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "dense_1_output" }
  },
  .nodes = (const stai_node_details[5]){
    {.id = 1, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){1}} }, /* conv1d_1 */
    {.id = 2, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){2}} }, /* activation_1 */
    {.id = 3, .type = AI_LAYER_POOL_TYPE, .input_tensors = {1, (const int32_t[1]){2}}, .output_tensors = {1, (const int32_t[1]){3}} }, /* max_pooling1d_1 */
    {.id = 4, .type = AI_LAYER_LSTM_TYPE, .input_tensors = {1, (const int32_t[1]){3}}, .output_tensors = {1, (const int32_t[1]){4}} }, /* lstm_1 */
    {.id = 5, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){4}}, .output_tensors = {1, (const int32_t[1]){5}} } /* dense_1 */
  },
  .n_nodes = 5
};
#endif

