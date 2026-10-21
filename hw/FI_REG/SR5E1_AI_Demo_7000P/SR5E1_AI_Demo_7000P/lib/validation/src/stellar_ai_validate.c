/****************************************************************************
*
* Copyright (c) 2023 STMicroelectronics - All Rights Reserved
*
* License terms: STMicroelectronics Proprietary in accordance with licensing
* terms SLA0098 at www.st.com.
*
* THIS SOFTWARE IS DISTRIBUTED "AS IS," AND ALL WARRANTIES ARE DISCLAIMED,
* INCLUDING MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
*
*****************************************************************************/
/**
 * @file    stellar_ai_validate.c
 * @brief   STELLAR AI Validate Wrapper Layer.
 *
 * @addtogroup STELLAR_AI_VALIDATE
 * @{
 */

#include "stellar_ai_validate.h"
#include "clock.h"
#include "io.h"
#include "irq.h"

/*===========================================================================*/
/* Module local definitions.                                                 */
/*===========================================================================*/

#if defined(SR5E1)
static uint8_t sr5e1_rxbuf[SR5E1_UART_RX_BUF_SIZE];
#endif

/*===========================================================================*/
/* Module exported variables.                                                */
/*===========================================================================*/

/*===========================================================================*/
/* Module local types.                                                       */
/*===========================================================================*/

/*===========================================================================*/
/* Module local variables.                                                   */
/*===========================================================================*/

/*===========================================================================*/
/* Module local functions.                                                   */
/*===========================================================================*/

/*===========================================================================*/
/* Module exported functions.                                                */
/*===========================================================================*/

/**
 * @brief   Initializes UART used during AI Validate
 *
 * @api
 */
#if defined(SR5E1)
void SR5E1_UART_Init(void) {

  uart_driver_t *io_up = NULL;
  /* If the RuntimeIO component is used with the same serial driver used by the
     AI Component, the serial driver must be stopped before re-starting it with
     a new specific configuration suitable for the host to target communication
     on which the validate procedure is based on. */

  /* Get the IO uart driver pointer.*/
  io_up = (uart_driver_t *)io_getptr();

  if ((io_up == NULL) || (io_up != &SR5E1_AI_VALIDATE_SERIAL_DRIVER)) {
    /* RuntimeIO not initialized or initilized with a different UART pointer.*/
  }
  else { /* the RuntimeIO component serial is the same of the AI component.*/
    /* Stop Runtime IO module.*/
    io_stop();
    /* De-initialize Runtime IO module.*/
    io_deinit();
    /* Stop the serial driver instance.*/
    uart_stop(&SR5E1_AI_VALIDATE_SERIAL_DRIVER);
    /* De-initialize the serial driver instance.*/
    uart_deinit(&SR5E1_AI_VALIDATE_SERIAL_DRIVER);
  }
  /* Initialize UART driver instance used for validation.*/
  uart_init(&SR5E1_AI_VALIDATE_SERIAL_DRIVER);
  /* Configure the AI UART driver instance.*/
  (void)uart_set_prio(&SR5E1_AI_VALIDATE_SERIAL_DRIVER, IRQ_PRIORITY_5);
  (void)uart_set_rx_drv_mode(&SR5E1_AI_VALIDATE_SERIAL_DRIVER, UART_RX_DRV_MODE_INT_BUFF_SYNC);
  (void)uart_set_tx_drv_mode(&SR5E1_AI_VALIDATE_SERIAL_DRIVER, UART_TX_DRV_MODE_INT_SYNC);
  (void)uart_set_baud(&SR5E1_AI_VALIDATE_SERIAL_DRIVER, UART_BAUDRATE_115200);
  (void)uart_set_presc(&SR5E1_AI_VALIDATE_SERIAL_DRIVER, UART_PRESCALER_DIV1);
  (void)uart_set_parity(&SR5E1_AI_VALIDATE_SERIAL_DRIVER, UART_PARITY_NONE);
  (void)uart_set_over(&SR5E1_AI_VALIDATE_SERIAL_DRIVER, UART_OVERSAMPLING_16);
  (void)uart_set_sbit(&SR5E1_AI_VALIDATE_SERIAL_DRIVER, UART_STOPBIT_1);
  (void)uart_set_sbrx_buf(&SR5E1_AI_VALIDATE_SERIAL_DRIVER, sr5e1_rxbuf, SR5E1_UART_RX_BUF_SIZE);
  /* Starts the AI UART instance.*/
  uart_start(&SR5E1_AI_VALIDATE_SERIAL_DRIVER);
}
#elif defined(SR6X)

extern UARTDriver UartHandle;

void SR6X_UART_Init(void) {
  /* Stop Runtime IO module.*/
  io_stop();
  /* Initialize UART driver instance used for validation.*/
  uart_init(&UartHandle, SR6X_AI_VALIDATE_SERIAL_DRIVER);
  /* Starts the AI UART instance.*/
  uart_start(&UartHandle);
}
#endif

/** @} */
