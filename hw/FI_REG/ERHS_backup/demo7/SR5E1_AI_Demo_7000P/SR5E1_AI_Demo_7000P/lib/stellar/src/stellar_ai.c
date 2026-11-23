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
 * @file    stellar_ai.c
 * @brief   Stellar AI module code.
 *
 * @addtogroup STELLAR_AI
 * @{
 */

#include "stellar_ai.h"
#if (STELLAR_AI_VALIDATE == TRUE) || defined(__DOXYGEN__)
#include "aiValidation.h"
#endif

/*===========================================================================*/
/* Module local variables.                                                   */
/*===========================================================================*/

/*===========================================================================*/
/* Module local functions.                                                   */
/*===========================================================================*/

/*===========================================================================*/
/* Module interrupt handlers.                                                */
/*===========================================================================*/

/*===========================================================================*/
/* Module exported functions.                                                */
/*===========================================================================*/

#if (STELLAR_AI_VALIDATE == TRUE) || defined(__DOXYGEN__)
/**
 * @brief   Stellar AI Validate start.
 * @details This function starts the validate on target procedure.
 *
 * @api
 */
void aiValidateStart(void) {

  /* Init Validate.*/
  aiValidationInit();

  /* Run Validate.*/
  for ( ; ; ) {
    aiValidationProcess();
  }
}
#endif /* STELLAR_AI_VALIDATE */

/** @} */
