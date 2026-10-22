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
 * @file    stellar_ai.h
 * @brief   STELLAR AI module header.
 *
 * @addtogroup STELLAR_AI
 * @{
 */

#ifndef _STELLAR_AI_H_
#define _STELLAR_AI_H_

#include "stellar_ai_cfg.h"

/*===========================================================================*/
/* Module constants.                                                         */
/*===========================================================================*/

/**
 * @name    Common constants
 * @{
 */
#if !defined(FALSE) || defined(__DOXYGEN__)
#define FALSE                               0
#endif

#if !defined(TRUE) || defined(__DOXYGEN__)
#define TRUE                                1
#endif
/** @} */

/*===========================================================================*/
/* Module pre-compile time settings.                                         */
/*===========================================================================*/

/**
 * @brief   AI Validate procedure enable switch.
 * @note    The default is @p FALSE (Validate procedure disabled).
 */
#if !defined(STELLAR_AI_VALIDATE) || defined(__DOXYGEN__)
#define STELLAR_AI_VALIDATE                 FALSE
#endif /* !defined(STELLAR_AI_VALIDATE) */

/*===========================================================================*/
/* Module macros.                                                            */
/*===========================================================================*/

/*===========================================================================*/
/* IRQ Handlers                                                              */
/*===========================================================================*/

/*===========================================================================*/
/* External declarations.                                                    */
/*===========================================================================*/
#ifdef __cplusplus
extern "C" {
#endif

#if (STELLAR_AI_VALIDATE == TRUE) || defined(__DOXYGEN__)
void aiValidateStart(void);
#endif /* (STELLAR_AI_VALIDATE == TRUE) */

#ifdef __cplusplus
}
#endif

#endif /* _STELLAR_AI_H_ */

/** @} */
