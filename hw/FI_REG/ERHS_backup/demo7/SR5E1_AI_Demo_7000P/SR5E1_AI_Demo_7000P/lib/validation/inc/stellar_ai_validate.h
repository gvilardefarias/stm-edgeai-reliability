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
 * @file    stellar_ai_validate.h
 * @brief   STELLAR AI Validate Wrapper Layer.
 *
 * @addtogroup STELLAR_AI_VALIDATE
 * @{
 */

#ifndef _STELLAR_AI_VALIDATE_H_
#define _STELLAR_AI_VALIDATE_H_

#include "stellar_ai_cfg.h"
#include "app_stellar-studio-ai.h"
#include "osal.h"
#include "version.h"

#if defined (SR5E1)
#include "uart.h"
#include "dwt.h"
#include "regs.h"
#include "SR5E1_SYSCFG.h"
#elif defined (SR6X)
#include "agt.h"
#include "uart.h"
#include "siul2.h"
#include "clock.h"
#endif

/*===========================================================================*/
/* Module constants.                                                         */
/*===========================================================================*/

#if defined (SR5E1)
#define SR5E1_UART_RX_BUF_SIZE              64U
#endif

/**
 * @name    Host to target serial driver constants.
 * @{
 */
#define NONE                                0U
#if defined (SR5E1)

#define SR5E1_UART1                         1U
#define SR5E1_UART2                         2U
#define SR5E1_UART3                         3U

#elif defined (SR6X)

#define SR6X_UART0                          1U
#define SR6X_UART1                          2U
#define SR6X_UART2                          3U
#define SR6X_UART3                          4U
#define SR6X_UART4                          5U
#define SR6X_UART5                          6U
#define SR6X_UART6                          7U
#define SR6X_UART7                          8U
#define SR6X_UART8                          9U
#define SR6X_UART9                         10U
#define SR6X_UART10                        11U
#define SR6X_UART11                        12U
#define SR6X_UART12                        13U
#define SR6X_UART13                        14U
#define SR6X_UART14                        15U
#define SR6X_UART15                        16U
#define SR6X_UART16                        17U
#define SR6X_UART17                        18U
#define SR6X_UART18                        19U
#define SR6X_UART19                        20U
#define SR6X_UART20                        21U
#define SR6X_UART21                        22U
#define SR6X_UART22                        23U
#define SR6X_UART23                        24U
#define SR6X_UART24                        25U
#define SR6X_UART25                        26U
#define SR6X_UART26                        27U
#define SR6X_UART27                        28U
#endif

/**
 * @name    Get the Device ID.
 * @{
 */
/* MCU part numbers */
#if defined (SR5E1)
#define SR5E1_GetDEVID()                    (REG_READ_FIELD(SYSCFG->MIDR1, SYSCFG_MIDR1_DEVICE_INFO))
#elif defined (SR6X)
#define SR6X_GetDEVID()                     (siul2_getMIDR1()>>16U)
#endif
/** @} */

/**
 * @name    Get the Revision ID.
 * @{
 */
/* Major and Minor mask revisions */
  /* [7:4] MAJOR_MASK: Major mask revision */
  /* [3:0] MINOR_MASK: Minor mask revision */
#if defined (SR5E1)
#define SR5E1_GetREVID()                    (REG_READ(SYSCFG->MIDR1)&(0x000000FFU))
#elif defined (SR6X)
#define SR6X_GetREVID()                     (siul2_getMIDR1()&(0x000000FFU))
#endif
/** @} */

/**
 * @name    Get the Stellar E clock frequencies.
 * @{
 */
#if defined (SR5E1)
#define SR5E1_GetCpuClockFreq()             CLOCK_SYSCLK
#endif
/** @} */

/**
 * @name    Get the Stellar P/G clock frequencies.
 * @{
 */
#if defined (SR6X)
#define SR6X_GetCpuClockFreq()              SSDK_PLL1_D0_PHI_HP_CLK
#define SR6X_GetTbu0TimeStampBusClock()     (SSDK_TIME_BASE_CLK * (1000000000UL / SSDK_TIME_BASE_CLK))
#if defined(SR6X_WITH_NPU)
#define SR6X_GetNpuClockFreq()              SSDK_NPU_CLK
#endif
#endif

/** @} */

/*===========================================================================*/
/* Module pre-compile time settings.                                         */
/*===========================================================================*/

/*===========================================================================*/
/* Derived constants and error checks.                                       */
/*===========================================================================*/

/**
 * @name    Host to target serial driver error checks.
 * @{
 */
#if (STELLAR_AI_VALIDATE == TRUE)
#if (STELLAR_AI_VALIDATE_SERIAL == NONE)
#error "For validate on target a Validate Serial must be selected!"
#endif
#endif
/** @} */

/*===========================================================================*/
/* Module data structures and types.                                         */
/*===========================================================================*/

/*===========================================================================*/
/* Module macros.                                                            */
/*===========================================================================*/

/**
 * @name    Host to target serial driver macros.
 * @{
 */
#if (STELLAR_AI_VALIDATE == TRUE)

#if defined (SR5E1)
#if (STELLAR_AI_VALIDATE_SERIAL == SR5E1_UART1)
#define SR5E1_AI_VALIDATE_SERIAL_DRIVER     DRV_UART1
#elif (STELLAR_AI_VALIDATE_SERIAL == SR5E1_UART2)
#define SR5E1_AI_VALIDATE_SERIAL_DRIVER     DRV_UART2
#elif (STELLAR_AI_VALIDATE_SERIAL == SR5E1_UART3)
#define SR5E1_AI_VALIDATE_SERIAL_DRIVER     DRV_UART3
#else
#define SR5E1_AI_VALIDATE_SERIAL_DRIVER     NULL /* to remove warning.*/
#endif
#endif

#if defined (SR6X)
#if (STELLAR_AI_VALIDATE_SERIAL == SR6X_UART0)
#define SR6X_AI_VALIDATE_SERIAL_DRIVER     UART_0_DEVICE_ID
#elif (STELLAR_AI_VALIDATE_SERIAL == SR6X_UART1)
#define SR6X_AI_VALIDATE_SERIAL_DRIVER     UART_1_DEVICE_ID
#elif (STELLAR_AI_VALIDATE_SERIAL == SR6X_UART2)
#define SR6X_AI_VALIDATE_SERIAL_DRIVER     UART_2_DEVICE_ID
#elif (STELLAR_AI_VALIDATE_SERIAL == SR6X_UART3)
#define SR6X_AI_VALIDATE_SERIAL_DRIVER     UART_3_DEVICE_ID
#elif (STELLAR_AI_VALIDATE_SERIAL == SR6X_UART4)
#define SR6X_AI_VALIDATE_SERIAL_DRIVER     UART_4_DEVICE_ID
#elif (STELLAR_AI_VALIDATE_SERIAL == SR6X_UART5)
#define SR6X_AI_VALIDATE_SERIAL_DRIVER     UART_5_DEVICE_ID
#elif (STELLAR_AI_VALIDATE_SERIAL == SR6X_UART6)
#define SR6X_AI_VALIDATE_SERIAL_DRIVER     UART_6_DEVICE_ID
#elif (STELLAR_AI_VALIDATE_SERIAL == SR6X_UART7)
#define SR6X_AI_VALIDATE_SERIAL_DRIVER     UART_7_DEVICE_ID
#elif (STELLAR_AI_VALIDATE_SERIAL == SR6X_UART8)
#define SR6X_AI_VALIDATE_SERIAL_DRIVER     UART_8_DEVICE_ID
#elif (STELLAR_AI_VALIDATE_SERIAL == SR6X_UART9)
#define SR6X_AI_VALIDATE_SERIAL_DRIVER     UART_9_DEVICE_ID
#elif (STELLAR_AI_VALIDATE_SERIAL == SR6X_UART10)
#define SR6X_AI_VALIDATE_SERIAL_DRIVER     UART_10_DEVICE_ID
#elif (STELLAR_AI_VALIDATE_SERIAL == SR6X_UART11)
#define SR6X_AI_VALIDATE_SERIAL_DRIVER     UART_11_DEVICE_ID
#elif (STELLAR_AI_VALIDATE_SERIAL == SR6X_UART12)
#define SR6X_AI_VALIDATE_SERIAL_DRIVER     UART_12_DEVICE_ID
#elif (STELLAR_AI_VALIDATE_SERIAL == SR6X_UART13)
#define SR6X_AI_VALIDATE_SERIAL_DRIVER     UART_13_DEVICE_ID
#elif (STELLAR_AI_VALIDATE_SERIAL == SR6X_UART14)
#define SR6X_AI_VALIDATE_SERIAL_DRIVER     UART_14_DEVICE_ID
#elif (STELLAR_AI_VALIDATE_SERIAL == SR6X_UART15)
#define SR6X_AI_VALIDATE_SERIAL_DRIVER     UART_15_DEVICE_ID
#elif (STELLAR_AI_VALIDATE_SERIAL == SR6X_UART16)
#define SR6X_AI_VALIDATE_SERIAL_DRIVER     UART_16_DEVICE_ID
#elif (STELLAR_AI_VALIDATE_SERIAL == SR6X_UART17)
#define SR6X_AI_VALIDATE_SERIAL_DRIVER     UART_17_DEVICE_ID
#elif (STELLAR_AI_VALIDATE_SERIAL == SR6X_UART18) 
#define SR6X_AI_VALIDATE_SERIAL_DRIVER     UART_18_DEVICE_ID
#elif (STELLAR_AI_VALIDATE_SERIAL == SR6X_UART19)
#define SR6X_AI_VALIDATE_SERIAL_DRIVER     UART_19_DEVICE_ID
#elif (STELLAR_AI_VALIDATE_SERIAL == SR6X_UART20)
#define SR6X_AI_VALIDATE_SERIAL_DRIVER     UART_20_DEVICE_ID
#elif (STELLAR_AI_VALIDATE_SERIAL == SR6X_UART21)
#define SR6X_AI_VALIDATE_SERIAL_DRIVER     UART_21_DEVICE_ID
#elif (STELLAR_AI_VALIDATE_SERIAL == SR6X_UART22)
#define SR6X_AI_VALIDATE_SERIAL_DRIVER     UART_22_DEVICE_ID
#elif (STELLAR_AI_VALIDATE_SERIAL == SR6X_UART23)
#define SR6X_AI_VALIDATE_SERIAL_DRIVER     UART_23_DEVICE_ID
#elif (STELLAR_AI_VALIDATE_SERIAL == SR6X_UART24)
#define SR6X_AI_VALIDATE_SERIAL_DRIVER     UART_24_DEVICE_ID
#elif (STELLAR_AI_VALIDATE_SERIAL == SR6X_UART25)
#define SR6X_AI_VALIDATE_SERIAL_DRIVER     UART_25_DEVICE_ID
#elif (STELLAR_AI_VALIDATE_SERIAL == SR6X_UART26)
#define SR6X_AI_VALIDATE_SERIAL_DRIVER     UART_26_DEVICE_ID
#elif (STELLAR_AI_VALIDATE_SERIAL == SR6X_UART27)
#define SR6X_AI_VALIDATE_SERIAL_DRIVER     UART_27_DEVICE_ID
#else
#define SR6X_AI_VALIDATE_SERIAL_DRIVER     NULL /* to remove warning.*/
#endif
#endif
#endif
/** @} */

/*===========================================================================*/
/* External declarations.                                                    */
/*===========================================================================*/

#ifdef __cplusplus
extern "C" {
#endif

#if defined(SR5E1)
void SR5E1_UART_Init(void);
#elif defined (SR6X)
void SR6X_UART_Init(void);
#endif

#ifdef __cplusplus
}
#endif

#endif /* _SR5E1_AI_VALIDATE_H_ */

/** @} */
