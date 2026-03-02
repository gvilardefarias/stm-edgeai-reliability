/**
  ******************************************************************************
  * @file    human_activity_recognition.h
  * @date    2026-03-02T16:03:18+0100
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
#ifndef STAI_HUMAN_ACTIVITY_RECOGNITION_DETAILS_H
#define STAI_HUMAN_ACTIVITY_RECOGNITION_DETAILS_H

#include "stai.h"
#include "layers.h"

const stai_network_details g_human_activity_recognition_details = {
  .tensors = (const stai_tensor[10]) {
   { .size_bytes = 288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 24, 3, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "input_layer_output" },
   { .size_bytes = 288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 24, 3, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "batch_normalization_output" },
   { .size_bytes = 3840, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 20, 3, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv2d_tmr3_output" },
   { .size_bytes = 3840, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 20, 3, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv2d_tmr2_output" },
   { .size_bytes = 3840, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 20, 3, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv2d_tmr1_output" },
   { .size_bytes = 3840, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 20, 3, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv2d_voter_output" },
   { .size_bytes = 3072, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 16, 3, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv2d_1_output" },
   { .size_bytes = 64, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 1, 1, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "global_max_pooling2d_pool_output" },
   { .size_bytes = 16, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 4}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "dense_output" },
   { .size_bytes = 16, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 4}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "activation_output" }
  },
  .nodes = (const stai_node_details[9]){
    {.id = 1, .type = AI_LAYER_BN_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){1}} }, /* batch_normalization */
    {.id = 4, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){2}} }, /* conv2d_tmr3 */
    {.id = 3, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){3}} }, /* conv2d_tmr2 */
    {.id = 2, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){4}} }, /* conv2d_tmr1 */
    {.id = 5, .type = AI_LAYER_CUSTOM_TYPE, .input_tensors = {3, (const int32_t[3]){4, 3, 2}}, .output_tensors = {1, (const int32_t[1]){5}} }, /* conv2d_voter */
    {.id = 7, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){5}}, .output_tensors = {1, (const int32_t[1]){6}} }, /* conv2d_1 */
    {.id = 8, .type = AI_LAYER_POOL_TYPE, .input_tensors = {1, (const int32_t[1]){6}}, .output_tensors = {1, (const int32_t[1]){7}} }, /* global_max_pooling2d_pool */
    {.id = 10, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){7}}, .output_tensors = {1, (const int32_t[1]){8}} }, /* dense */
    {.id = 11, .type = AI_LAYER_SM_TYPE, .input_tensors = {1, (const int32_t[1]){8}}, .output_tensors = {1, (const int32_t[1]){9}} } /* activation */
  },
  .n_nodes = 9
};
#endif

