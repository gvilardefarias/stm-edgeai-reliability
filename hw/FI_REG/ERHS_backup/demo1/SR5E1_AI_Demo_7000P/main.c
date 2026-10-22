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

#include <test_env.h>
#if 1
#include <uart.h>
#include <io.h>
#endif
#include <irq.h>
#include <stdio.h>

// Custom Includes
#include "fault_injection.h"

#include "stellar_ai_cfg.h"

#if (defined EVBE7000P) && (EVBE7000P == 1)
#define UART1_RX gpio_iopack(GPIO_PORT_A, GPIO_PIN_0)
#define UART1_RX_CFG                             \
    (GPIO_MODE_MODER_ALTERNATE |                 \
     GPIO_MODE_OTYPER_PUSHPULL |                 \
     GPIO_MODE_OSPEEDR_LOW |                     \
     GPIO_MODE_PUPDR_FLOATING |                  \
     GPIO_MODE_AFR(7U) |                         \
     GPIO_MODE_IHYSTR_CMOS |                     \
     GPIO_MODE_TRIGENR_IN_ENABLED |              \
     GPIO_MODE_SAFESELR_DISABLED |               \
     GPIO_MODE_SAFEVALR_LOW)

#define UART1_TX gpio_iopack(GPIO_PORT_A, GPIO_PIN_1)
#define UART1_TX_CFG                             \
    (GPIO_MODE_MODER_ALTERNATE |                 \
     GPIO_MODE_OTYPER_PUSHPULL |                 \
     GPIO_MODE_OSPEEDR_LOW |                     \
     GPIO_MODE_PUPDR_FLOATING |                  \
     GPIO_MODE_AFR(7U) |                         \
     GPIO_MODE_IHYSTR_CMOS |                     \
    GPIO_MODE_TRIGENR_IN_DISABLED |             \
     GPIO_MODE_SAFESELR_DISABLED |               \
     GPIO_MODE_SAFEVALR_LOW)
#endif

#if (STELLAR_AI_VALIDATE == TRUE)
#include "stellar_ai.h"
#endif /* #if (STELLAR_AI_VALIDATE == TRUE) */

/* 
 * Example of system initialization function
 */
void SystemInit(void)
{
  /* Enable interrupts.*/
  osal_sys_unlock();

  test_env_init((TestInit_t)
                (TEST_INIT_CLOCK    |
                 TEST_INIT_GPIO     |
                 TEST_INIT_BOARD    |
                 TEST_INIT_IRQ      |
                 TEST_INIT_OSAL));

  gpio_set_pin_mode(UART1_RX, UART1_RX_CFG);
  gpio_set_pin_mode(UART1_TX, UART1_TX_CFG);

  /* Initialize UART driver instance used for IO redirection.*/
  uart_init(&DRV_UART1);

  /* Configure UART driver instance used for IO redirection.*/
  (void)uart_set_prio(&DRV_UART1, IRQ_PRIORITY_5);
  (void)uart_set_rx_drv_mode(&DRV_UART1, UART_RX_DRV_MODE_INT_SYNC);
  (void)uart_set_tx_drv_mode(&DRV_UART1, UART_TX_DRV_MODE_INT_SYNC);
  (void)uart_set_baud(&DRV_UART1, UART_BAUDRATE_115200);
  (void)uart_set_presc(&DRV_UART1, UART_PRESCALER_DIV1);
  (void)uart_set_parity(&DRV_UART1, UART_PARITY_NONE);
  (void)uart_set_over(&DRV_UART1, UART_OVERSAMPLING_16);
  (void)uart_set_sbit(&DRV_UART1, UART_STOPBIT_1);
    
  /* Initialize Runtime IO module.*/
  io_init(&DRV_UART1);

  /* Start Runtime IO module.*/
  io_start();

  /* Enabling the Data Cache.*/
  SCB_EnableDCache();
}

#if (STELLAR_AI_VALIDATE == TRUE)
/*
 * Application entry point for validation process
 */
int main(void) {

  SystemInit();

  FI_Init();
  FI_Start();

  aiValidateStart();

  /* never here...*/
  while (true) {
  }
}
#else
/*
 * Application entry point for inference run
 */
int main(void) {

  SystemInit();

  printf("#### Minimal application to run inference.\r\n");

  /* give a look at the main_loop() function inside the ws\build_network\stedgeai_app_main.c generated file on how to initialize and run the inference. */
  /* main_loop(); */
  
  /* never here...*/
  while (true) {
  }
}
#endif /* #if (STELLAR_AI_VALIDATE == TRUE) */
