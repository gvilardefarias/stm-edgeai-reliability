/****************************************************************************
*
* Copyright (c) 2026 STMicroelectronics - All Rights Reserved
*
* License terms: STMicroelectronics Proprietary in accordance with licensing
* terms SLA0098 at www.st.com.
*
* THIS SOFTWARE IS DISTRIBUTED "AS IS," AND ALL WARRANTIES ARE DISCLAIMED,
* INCLUDING MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
*
*****************************************************************************/

/**
 * @file    app_stellar-studio-ai.h
 * @author  MDRF GPAM team
 * @brief   AI Validation application (ST AI embedded legacy/c-api)
 *
 * @addtogroup APP_STELLAR-STUDIO-AI
 * @{
 */

#ifndef _APP_STELLAR_STUDIO_AI_H_
#define _APP_STELLAR_STUDIO_AI_H_

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stai.h"
#include "network.h"
#include "network_data.h"

#define MIN_HEAP_SIZE 0x2000
#define MIN_STACK_SIZE 0x2000


#define STAI_MNETWORK_DATA_ACTIVATIONS_INT_SIZE 0

#if STAI_NETWORK_ACTIVATIONS_SIZE > STAI_MNETWORK_DATA_ACTIVATIONS_INT_SIZE
#undef STAI_MNETWORK_DATA_ACTIVATIONS_INT_SIZE
#define STAI_MNETWORK_DATA_ACTIVATIONS_INT_SIZE STAI_NETWORK_ACTIVATIONS_SIZE
#endif 

/* IO buffers ----------------------------------------------------------------*/

extern stai_ptr data_ins[];
extern stai_ptr data_outs[];

extern stai_ptr data_activations[];
extern stai_ptr data_states[];

#if 0
int STELLAR_AI_Init(void);
int STELLAR_AI_Process(void);
#endif

/* USER CODE BEGIN includes */
/* USER CODE END includes */

/* Multiple network support --------------------------------------------------*/

typedef struct {
    const char *name;
    stai_size (*get_context_size)(void);
    stai_return_code (*init)(stai_network* network);
    stai_return_code (*deinit)(stai_network* network);
    stai_return_code (*run)(stai_network* network, const stai_run_mode mode);
    stai_return_code (*get_info)(stai_network* network, stai_network_info* info);
    stai_return_code (*get_error)(stai_network* network);

    stai_return_code (*get_activations)(stai_network* network, stai_ptr* activations, stai_size *n_activations);
    stai_return_code (*get_states)(stai_network* network, stai_ptr* states, stai_size *n_states);
    stai_return_code (*get_weights)(stai_network* network, stai_ptr* weights, stai_size *n_weights);
    stai_return_code (*get_inputs)(stai_network* network, stai_ptr* inputs, stai_size* n_inputs);
    stai_return_code (*get_outputs)(stai_network* network, stai_ptr* outputs, stai_size* n_outputs);

    stai_return_code (*set_activations)(stai_network* network, const stai_ptr* activations, const stai_size n_activations);
    stai_return_code (*set_states)(stai_network* network, const stai_ptr* states, const stai_size n_states);
    stai_return_code (*set_weights)(stai_network* network, const stai_ptr* weights, const stai_size n_weights);
    stai_return_code (*set_inputs)(stai_network* network, const stai_ptr* inputs, const stai_size n_inputs);
    stai_return_code (*set_outputs)(stai_network* network, const stai_ptr* outputs, const stai_size n_outputs);
    stai_return_code (*set_callback)(stai_network* network, const stai_event_cb cb, void* cb_cookie);

    stai_network *context;
    stai_size n_inputs;
    stai_size n_outputs;
    stai_size n_activations;
    stai_size n_states;
    stai_size n_weights;

    uintptr_t *weights_addr;
    
    const stai_network_details *details;

} stai_network_entry_t;

#define STAI_MNETWORK_NUMBER  (1)
#define STAI_MNETWORK_IN_1_SIZE_BYTES 1
#define STAI_MNETWORK_IN_NUM STAI_NETWORK_IN_NUM
#define DEF_DATA_IN \
  STAI_ALIGNED(32) static uint8_t data_in_1[STAI_MNETWORK_IN_1_SIZE_BYTES]; \
  stai_ptr data_ins[] = { \
    data_in_1, \
  }; \

#define STAI_MNETWORK_OUT_1_SIZE_BYTES 1
#define STAI_MNETWORK_OUT_NUM STAI_NETWORK_OUT_NUM
#define DEF_DATA_OUT \
  STAI_ALIGNED(32) static uint8_t data_out_1[STAI_MNETWORK_OUT_1_SIZE_BYTES]; \
  stai_ptr data_outs[] = { \
    data_out_1, \
  }; \

STAI_API_DECLARE_BEGIN

STAI_API_ENTRY
const char* stai_mnetwork_find(const char *name, int idx);

STAI_API_ENTRY
stai_return_code stai_mnetwork_init(const char *name, stai_ptr* network);

STAI_API_ENTRY
stai_return_code stai_mnetwork_deinit(stai_ptr network);

STAI_API_ENTRY
stai_return_code stai_mnetwork_get_info(stai_ptr network, stai_network_info* report);

STAI_API_ENTRY
stai_return_code stai_mnetwork_get_error(stai_ptr network);

STAI_API_ENTRY
stai_return_code stai_mnetwork_run(stai_ptr network);

STAI_API_ENTRY
stai_return_code stai_mnetwork_set_activations(stai_ptr network, const stai_ptr* activations);

STAI_API_ENTRY
stai_return_code stai_mnetwork_set_states(stai_ptr network, const stai_ptr* states);

STAI_API_ENTRY
stai_return_code stai_mnetwork_set_weights(stai_ptr network, const stai_ptr* weights);

STAI_API_ENTRY
stai_return_code stai_mnetwork_set_inputs(stai_ptr network, const stai_ptr* inputs);

STAI_API_ENTRY
stai_return_code stai_mnetwork_set_outputs(stai_ptr network, const stai_ptr* outputs);

STAI_API_ENTRY
stai_return_code stai_mnetwork_set_callback(stai_ptr network, const stai_event_cb cb, void* cb_cookie);

STAI_API_ENTRY
const stai_network_details *stai_mnetwork_get_details(stai_ptr network);

STAI_API_ENTRY
stai_return_code stai_mnetwork_get_activations(stai_ptr network, stai_ptr* activations);

STAI_API_ENTRY
stai_return_code stai_mnetwork_get_states(stai_ptr network, stai_ptr* states);

STAI_API_ENTRY
stai_return_code stai_mnetwork_get_weights(stai_ptr network, stai_ptr* weights);

STAI_API_ENTRY
stai_ptr *stai_mnetwork_get_weights_ext(stai_ptr network);

STAI_API_ENTRY
stai_return_code stai_mnetwork_get_inputs(stai_ptr network, stai_ptr* inputs);

STAI_API_ENTRY
stai_return_code stai_mnetwork_get_outputs(stai_ptr network, stai_ptr* outputs);

STAI_API_ENTRY
stai_ptr stai_mnetwork_get_internal_handle(const stai_ptr network);

STAI_API_DECLARE_END

#ifdef __cplusplus
}
#endif


/**
 * @brief   Validate serial.
 * @details Specify the serial used for host to target communication on which
 *          the validate on target procedure is based on.
 */
#define STELLAR_AI_VALIDATE_SERIAL             SR5E1_UART3

#endif /* _APP_STELLAR_STUDIO_AI_H_ */

/** @} */
