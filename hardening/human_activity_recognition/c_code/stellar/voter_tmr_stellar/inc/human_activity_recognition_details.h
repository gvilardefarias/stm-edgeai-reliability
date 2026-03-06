/**
  ******************************************************************************
  * @file    human_activity_recognition.h
  * @date    2026-03-06T11:10:24+0100
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
  .tensors = (stai_tensor[9]) {
   { .size_bytes = 288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (int32_t[4]){1, 24, 3, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "input_1_output" },
   { .size_bytes = 3840, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (int32_t[4]){1, 20, 3, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv2d_output" },
   { .size_bytes = 3072, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (int32_t[4]){1, 16, 3, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv2d_1_tmr3_output" },
   { .size_bytes = 3072, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (int32_t[4]){1, 16, 3, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv2d_1_tmr2_output" },
   { .size_bytes = 3072, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (int32_t[4]){1, 16, 3, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv2d_1_tmr1_output" },
   { .size_bytes = 3072, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (int32_t[4]){1, 16, 3, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv2d_1_voter_output" },
   { .size_bytes = 64, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (int32_t[4]){1, 1, 1, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "global_max_pooling2d_pool_output" },
   { .size_bytes = 16, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (int32_t[2]){1, 4}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "dense_output" },
   { .size_bytes = 16, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (int32_t[2]){1, 4}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "activation_output" }
  },
  .nodes = (stai_node_details[8]){
    {.id = 3, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (int32_t[1]){0}}, .output_tensors = {1, (int32_t[1]){1}} }, /* conv2d */
    {.id = 6, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (int32_t[1]){1}}, .output_tensors = {1, (int32_t[1]){2}} }, /* conv2d_1_tmr3 */
    {.id = 5, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (int32_t[1]){1}}, .output_tensors = {1, (int32_t[1]){3}} }, /* conv2d_1_tmr2 */
    {.id = 4, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (int32_t[1]){1}}, .output_tensors = {1, (int32_t[1]){4}} }, /* conv2d_1_tmr1 */
    {.id = 7, .type = AI_LAYER_CUSTOM_TYPE, .input_tensors = {3, (int32_t[3]){4, 3, 2}}, .output_tensors = {1, (int32_t[1]){5}} }, /* conv2d_1_voter */
    {.id = 8, .type = AI_LAYER_POOL_TYPE, .input_tensors = {1, (int32_t[1]){5}}, .output_tensors = {1, (int32_t[1]){6}} }, /* global_max_pooling2d_pool */
    {.id = 10, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (int32_t[1]){6}}, .output_tensors = {1, (int32_t[1]){7}} }, /* dense */
    {.id = 11, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (int32_t[1]){7}}, .output_tensors = {1, (int32_t[1]){8}} } /* activation */
  },
  .n_nodes = 8
};
#endif