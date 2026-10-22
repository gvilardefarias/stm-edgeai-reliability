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
 * @file    app_stellar-studio-ai.c
 * @author  MDRF GPAM team
 * @brief   Entry point for the Stellar AI test application
 *
 * @addtogroup APP_STELLAR-STUDIO-AI
 * @{
 */

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include <string.h>
#include "app_stellar-studio-ai.h"
#include "stellar_ai_validate.h"
#include "aiTestUtility.h"
#include "aiValidation.h"

#if 0
int STELLAR_AI_Init(void)
{
  int ret = 0;
  /* SR5E1_UART_Init(); */
  /* SR6X_UART_Init(); */
  aiValidationInit();
  /* USER CODE BEGIN */
  /* USER CODE END */
  return ret;
}

int STELLAR_AI_Process(void)
{
  int ret = aiValidationProcess();
  port_hal_delay(1000); /* delay 1s */
  /* USER CODE BEGIN */
  /* USER CODE END */
  return ret;
}
#endif

/* IO buffers ----------------------------------------------------------------*/

DEF_DATA_IN

DEF_DATA_OUT


/* Activations buffers allocation --------------------------------------------*/

#undef  STAI_MNETWORK_ACTIVATION_1_SIZE_BYTES
#define STAI_MNETWORK_ACTIVATION_1_SIZE_BYTES STAI_NETWORK_ACTIVATION_1_SIZE_BYTES
#undef  STAI_MNETWORK_ACTIVATION_2_SIZE_BYTES
#define STAI_MNETWORK_ACTIVATION_2_SIZE_BYTES STAI_NETWORK_ACTIVATION_2_SIZE_BYTES
#undef  STAI_MNETWORK_ACTIVATION_3_SIZE_BYTES
#define STAI_MNETWORK_ACTIVATION_3_SIZE_BYTES STAI_NETWORK_ACTIVATION_3_SIZE_BYTES
#undef  STAI_MNETWORK_ACTIVATION_4_SIZE_BYTES
#define STAI_MNETWORK_ACTIVATION_4_SIZE_BYTES STAI_NETWORK_ACTIVATION_4_SIZE_BYTES

#if STAI_NETWORK_ACTIVATIONS_NUM == 0
/* Dummy buffer is allocated */
STAI_ALIGNED(32) static uint8_t activations_1[4];
stai_ptr data_activations[] = {
  activations_1
};
#elif STAI_NETWORK_ACTIVATIONS_NUM == 1
#if STAI_NETWORK_ACTIVATION_1_SIZE_BYTES == 0
/* Dummy buffer is allocated */
STAI_ALIGNED(32) static uint8_t activations_1[4];
#else
STAI_ALIGNED(32) static uint8_t activations_1[STAI_MNETWORK_ACTIVATION_1_SIZE_BYTES];
#endif
stai_ptr data_activations[] = {
  activations_1
};
#elif STAI_NETWORK_ACTIVATIONS_NUM == 2
STAI_ALIGNED(32) static uint8_t activations_1[STAI_MNETWORK_ACTIVATION_1_SIZE_BYTES];
STAI_ALIGNED(32) static uint8_t activations_2[STAI_MNETWORK_ACTIVATION_2_SIZE_BYTES];
stai_ptr data_activations[] = {
  activations_1,
  activations_2
};
#elif STAI_NETWORK_ACTIVATIONS_NUM == 3
STAI_ALIGNED(32) static uint8_t activations_1[STAI_MNETWORK_ACTIVATION_1_SIZE_BYTES];
STAI_ALIGNED(32) static uint8_t activations_2[STAI_MNETWORK_ACTIVATION_2_SIZE_BYTES];
STAI_ALIGNED(32) static uint8_t activations_3[STAI_MNETWORK_ACTIVATION_3_SIZE_BYTES];
stai_ptr data_activations[] = {
  activations_1,
  activations_2,
  activations_3
};
#elif STAI_NETWORK_ACTIVATIONS_NUM == 4
STAI_ALIGNED(32) static uint8_t activations_1[STAI_MNETWORK_ACTIVATION_1_SIZE_BYTES];
STAI_ALIGNED(32) static uint8_t activations_2[STAI_MNETWORK_ACTIVATION_2_SIZE_BYTES];
STAI_ALIGNED(32) static uint8_t activations_3[STAI_MNETWORK_ACTIVATION_3_SIZE_BYTES];
STAI_ALIGNED(32) static uint8_t activations_4[STAI_MNETWORK_ACTIVATION_4_SIZE_BYTES];
stai_ptr data_activations[] = {
  activations_1,
  activations_2,
  activations_3,
  activations_4
};
#else
#error Maximum of 4 activations buffer are supported for the network named network
#endif

/* States buffers allocation -------------------------------------------------*/

#undef  STAI_MNETWORK_STATE_1_SIZE_BYTES
#define STAI_MNETWORK_STATE_1_SIZE_BYTES STAI_NETWORK_STATE_1_SIZE_BYTES
#undef  STAI_MNETWORK_STATE_2_SIZE_BYTES
#define STAI_MNETWORK_STATE_2_SIZE_BYTES STAI_NETWORK_STATE_2_SIZE_BYTES
#undef  STAI_MNETWORK_STATE_3_SIZE_BYTES
#define STAI_MNETWORK_STATE_3_SIZE_BYTES STAI_NETWORK_STATE_3_SIZE_BYTES
#undef  STAI_MNETWORK_STATE_4_SIZE_BYTES
#define STAI_MNETWORK_STATE_4_SIZE_BYTES STAI_NETWORK_STATE_4_SIZE_BYTES

#if STAI_NETWORK_STATES_NUM == 0
/* Dummy buffer is allocated */
STAI_ALIGNED(32) static uint8_t states_1[4];
stai_ptr data_states[] = {
  states_1
};
#elif STAI_NETWORK_STATES_NUM == 1
#if STAI_NETWORK_STATE_1_SIZE_BYTES == 0
/* Dummy buffer is allocated */
STAI_ALIGNED(32) static uint8_t states_1[4];
#else
STAI_ALIGNED(32) static uint8_t states_1[STAI_MNETWORK_STATE_1_SIZE_BYTES];
#endif
stai_ptr data_states[] = {
  states_1
};
#elif STAI_NETWORK_STATES_NUM == 2
STAI_ALIGNED(32) static uint8_t states_1[STAI_MNETWORK_STATE_1_SIZE_BYTES];
STAI_ALIGNED(32) static uint8_t states_2[STAI_MNETWORK_STATE_2_SIZE_BYTES];
stai_ptr data_states[] = {
  states_1,
  states_2
};
#elif STAI_NETWORK_STATES_NUM == 3
STAI_ALIGNED(32) static uint8_t states_1[STAI_MNETWORK_STATE_1_SIZE_BYTES];
STAI_ALIGNED(32) static uint8_t states_2[STAI_MNETWORK_STATE_2_SIZE_BYTES];
STAI_ALIGNED(32) static uint8_t states_3[STAI_MNETWORK_STATE_3_SIZE_BYTES];
stai_ptr data_states[] = {
  states_1,
  states_2,
  states_3
};
#elif STAI_NETWORK_STATES_NUM == 4
STAI_ALIGNED(32) static uint8_t states_1[STAI_MNETWORK_STATE_1_SIZE_BYTES];
STAI_ALIGNED(32) static uint8_t states_2[STAI_MNETWORK_STATE_2_SIZE_BYTES];
STAI_ALIGNED(32) static uint8_t states_3[STAI_MNETWORK_STATE_3_SIZE_BYTES];
STAI_ALIGNED(32) static uint8_t states_4[STAI_MNETWORK_STATE_4_SIZE_BYTES];
stai_ptr data_states[] = {
  states_1,
  states_2,
  states_3,
  states_4
};
#else
#error Maximum of 4 state buffer are supported for the network named network
#endif

/* Multiple network support --------------------------------------------------*/

#include <string.h>

STAI_NETWORK_CONTEXT_DECLARE(_context_network, STAI_NETWORK_CONTEXT_SIZE)
uintptr_t stai_network_weights_addr[STAI_NETWORK_WEIGHTS_NUM + 1 ];
extern const stai_network_details g_network_details;

static const stai_network_entry_t networks[STAI_MNETWORK_NUMBER] = {
{
	.name = (const char *)STAI_NETWORK_MODEL_NAME,
	.get_context_size = stai_network_get_context_size,
	.init = stai_network_init,
	.deinit = stai_network_deinit,
	.run = stai_network_run,
	.get_info = stai_network_get_info,
	.get_error = stai_network_get_error,
	
	.get_activations = stai_network_get_activations,
	.get_states = stai_network_get_states,
	.get_weights = stai_network_get_weights,
	.get_inputs = stai_network_get_inputs,
	.get_outputs = stai_network_get_outputs,
	
	.set_activations = stai_network_set_activations,
	.set_states = stai_network_set_states,
	.set_weights = stai_network_set_weights,
	.set_inputs = stai_network_set_inputs,
	.set_outputs = stai_network_set_outputs,
	.set_callback = stai_network_set_callback,
	
	.context = _context_network,
	
	.n_inputs = STAI_NETWORK_IN_NUM,
	.n_outputs = STAI_NETWORK_OUT_NUM,
	
	.n_activations = STAI_NETWORK_ACTIVATIONS_NUM,
	.n_states = STAI_NETWORK_STATES_NUM,
	.n_weights = STAI_NETWORK_WEIGHTS_NUM,
	.weights_addr = stai_network_weights_addr,
	
	.details = &g_network_details,
	},
};

struct network_instance {
     const stai_network_entry_t *entry;
     stai_ptr handle;
};

/* Number of instance is aligned on the number of network */
static struct network_instance gnetworks[STAI_MNETWORK_NUMBER] = {0};

static bool _stai_mnetwork_is_valid(const char* name,
    const stai_network_entry_t *entry)
{
  if (name && (strlen(entry->name) == strlen(name)) &&
      (strncmp(entry->name, name, strlen(entry->name)) == 0))
    return true;
  return false;
}

static struct network_instance *_stai_mnetwork_handle(struct network_instance *inst)
{
  for (int i=0; i<STAI_MNETWORK_NUMBER; i++) {
    if ((inst) && (&gnetworks[i] == inst))
      return inst;
    else if ((!inst) && (gnetworks[i].entry == NULL))
      return &gnetworks[i];
  }
  return NULL;
}

static void _stai_mnetwork_release_handle(struct network_instance *inst)
{
  for (int i=0; i<STAI_MNETWORK_NUMBER; i++) {
    if ((inst) && (&gnetworks[i] == inst)) {
      gnetworks[i].entry = NULL;
      return;
    }
  }
}

STAI_API_ENTRY
stai_ptr stai_mnetwork_get_internal_handle(const stai_ptr network)
{
    struct network_instance *inn;
    inn =  _stai_mnetwork_handle((struct network_instance *)network);
    if (inn)
      return inn->handle;
    return NULL;
}

STAI_API_ENTRY
const char* stai_mnetwork_find(const char *name, int idx)
{
  const stai_network_entry_t *entry;

  for (int i=0; i<STAI_MNETWORK_NUMBER; i++) {
    entry = &networks[i];
    if (_stai_mnetwork_is_valid(name, entry))
      return entry->name;
    else {
      if (!idx--)
        return entry->name;
    }
  }
  return NULL;
}

STAI_API_ENTRY
stai_return_code stai_mnetwork_init(const char *name, stai_ptr* network)
{
  const stai_network_entry_t *entry;
  const stai_network_entry_t *found = NULL;
  stai_return_code stai_err;

  struct network_instance *inst = _stai_mnetwork_handle(NULL);

  if (!inst || !network) {
    return STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE;
  }

  *network = NULL;

  for (int i=0; i<STAI_MNETWORK_NUMBER; i++) {
    entry = &networks[i];
    if (_stai_mnetwork_is_valid(name, entry)) {
      found = entry;
      break;
    }
  }

  if (!found) {
    return STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE;
  }

  stai_err = found->init(found->context);
  if (stai_err == STAI_SUCCESS) {
    stai_size tmp_;
    inst->entry = found;
    inst->handle = found->context;
    found->get_weights(found->context, (stai_ptr *)found->weights_addr, &tmp_);
    *network = (stai_ptr)inst;
    return STAI_SUCCESS;
  }
  return stai_err;
}

STAI_API_ENTRY
stai_return_code stai_mnetwork_set_activations(stai_ptr network, const stai_ptr* activations)
{
  struct network_instance *inn;
  stai_return_code stai_err = STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE;

  inn =  _stai_mnetwork_handle((struct network_instance *)network);
  if (inn) {
    stai_err = inn->entry->set_activations(inn->handle, activations, inn->entry->n_activations);
  }
  return stai_err;
}

STAI_API_ENTRY
stai_return_code stai_mnetwork_set_states(stai_ptr network, const stai_ptr* states)
{
  struct network_instance *inn;
  stai_return_code stai_err = STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE;

  inn =  _stai_mnetwork_handle((struct network_instance *)network);
  if (inn) {
    stai_err = inn->entry->set_states(inn->handle, states, inn->entry->n_states);
  }
  return stai_err;
}

STAI_API_ENTRY
stai_return_code stai_mnetwork_set_weights(stai_ptr network, const stai_ptr* weights)
{
  struct network_instance *inn;
  stai_return_code stai_err = STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE;

  inn =  _stai_mnetwork_handle((struct network_instance *)network);
  if (inn) {
    stai_size tmp_;
    stai_err = inn->entry->set_weights(inn->handle, weights, inn->entry->n_weights);
    inn->entry->get_weights(inn->handle, (stai_ptr *)inn->entry->weights_addr, &tmp_);
  }
  return stai_err;
}

STAI_API_ENTRY
stai_return_code stai_mnetwork_set_inputs(stai_ptr network, const stai_ptr* inputs)
{
  struct network_instance *inn;
  stai_return_code stai_err = STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE;

  inn = _stai_mnetwork_handle((struct network_instance *)network);
  if (inn) {
    stai_err = inn->entry->set_inputs(inn->handle, inputs, inn->entry->n_inputs);
  }
  return stai_err;
}

STAI_API_ENTRY
stai_return_code stai_mnetwork_set_outputs(stai_ptr network, const stai_ptr* outputs)
{
  struct network_instance *inn;
  stai_return_code stai_err = STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE;

  inn = _stai_mnetwork_handle((struct network_instance *)network);
  if (inn) {
    stai_err = inn->entry->set_outputs(inn->handle, outputs, inn->entry->n_outputs);
  }
  return stai_err;
}

STAI_API_ENTRY
stai_return_code stai_mnetwork_set_callback(stai_ptr network, const stai_event_cb cb, void* cb_cookie)
{
  struct network_instance *inn;
  stai_return_code stai_err = STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE;

  inn = _stai_mnetwork_handle((struct network_instance *)network);
  if (inn) {
    stai_err = inn->entry->set_callback(inn->handle, cb, cb_cookie);
  }
  return stai_err;
}

STAI_API_ENTRY
const stai_network_details *stai_mnetwork_get_details(stai_ptr network)
{
  struct network_instance *inn;

  inn = _stai_mnetwork_handle((struct network_instance *)network);
  if (inn) {
    return inn->entry->details;
  }
  return NULL;
}

STAI_API_ENTRY
stai_return_code stai_mnetwork_get_activations(stai_ptr network, stai_ptr* activations)
{
  struct network_instance *inn;
  stai_return_code stai_err = STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE;

  inn = _stai_mnetwork_handle((struct network_instance *)network);
  if (inn) {
    stai_size tmp_;
    stai_err = inn->entry->get_activations(inn->handle, activations, &tmp_);
  }
  return stai_err;
}

STAI_API_ENTRY
stai_return_code stai_mnetwork_get_states(stai_ptr network, stai_ptr* states)
{
  struct network_instance *inn;
  stai_return_code stai_err = STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE;

  inn = _stai_mnetwork_handle((struct network_instance *)network);
  if (inn) {
    stai_size tmp_;
    stai_err = inn->entry->get_states(inn->handle, states, &tmp_);
  }
  return stai_err;
}

STAI_API_ENTRY
stai_return_code stai_mnetwork_get_weights(stai_ptr network, stai_ptr* weights)
{
  struct network_instance *inn;
  stai_return_code stai_err = STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE;

  inn = _stai_mnetwork_handle((struct network_instance *)network);
  if (inn) {
    stai_size tmp_;
    stai_err = inn->entry->get_weights(inn->handle, weights, &tmp_);
  }
  return stai_err;
}

STAI_API_ENTRY
stai_ptr *stai_mnetwork_get_weights_ext(stai_ptr network)
{
  struct network_instance *inn;

  inn = _stai_mnetwork_handle((struct network_instance *)network);
  if (inn) {
    return (stai_ptr *)inn->entry->weights_addr;
  }
  return NULL;
}

STAI_API_ENTRY
stai_return_code stai_mnetwork_get_inputs(stai_ptr network, stai_ptr* inputs)
{
  struct network_instance *inn;
  stai_return_code stai_err = STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE;

  inn = _stai_mnetwork_handle((struct network_instance *)network);
  if (inn) {
    stai_size tmp_;
    stai_err = inn->entry->get_inputs(inn->handle, inputs, &tmp_);
  }
  return stai_err;
}

STAI_API_ENTRY
stai_return_code stai_mnetwork_get_outputs(stai_ptr network, stai_ptr* outputs)
{
  struct network_instance *inn;
  stai_return_code stai_err = STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE;

  inn = _stai_mnetwork_handle((struct network_instance *)network);
  if (inn) {
    stai_size tmp_;
    stai_err = inn->entry->get_outputs(inn->handle, outputs, &tmp_);
  }
  return stai_err;
}


STAI_API_ENTRY
stai_return_code stai_mnetwork_deinit(stai_ptr network)
{
  struct network_instance *inn;
  stai_return_code stai_err = STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE;

  inn = _stai_mnetwork_handle((struct network_instance *)network);
  if (inn) {
    stai_err = inn->entry->deinit(inn->handle);
    if (stai_err == STAI_SUCCESS) {
      _stai_mnetwork_release_handle(inn);
    }
  }
  return stai_err;
}

STAI_API_ENTRY
stai_return_code stai_mnetwork_get_info(stai_ptr network, stai_network_info* report)
{
  struct network_instance *inn;
  stai_return_code stai_err = STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE;

  inn = _stai_mnetwork_handle((struct network_instance *)network);
  if (inn) {
    return inn->entry->get_info(inn->handle, report);
  }
  return stai_err;
}

STAI_API_ENTRY
stai_return_code stai_mnetwork_get_error(stai_ptr network)
{
  struct network_instance *inn;
  stai_return_code stai_err = STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE;

  inn = _stai_mnetwork_handle((struct network_instance *)network);
  if (inn) {
    return inn->entry->get_error(inn->handle);
  }
  return stai_err;
}

STAI_API_ENTRY
stai_return_code stai_mnetwork_run(stai_ptr network)
{
  struct network_instance* inn;
  stai_return_code stai_err = STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE;

  inn = _stai_mnetwork_handle((struct network_instance *)network);
  if (inn) {
    return inn->entry->run(inn->handle, STAI_MODE_SYNC);
  }
  return stai_err;
}

#ifdef __cplusplus
}
#endif


/** @} */
