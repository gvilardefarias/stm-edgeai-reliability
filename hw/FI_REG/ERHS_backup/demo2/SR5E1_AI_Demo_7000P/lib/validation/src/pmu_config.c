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

#include <pmu_config.h>

/*===========================================================================*/
/* Private constants.                                                         */
/*===========================================================================*/
#define PMCR_RESET_VALUE         0x41134000
#define PMCR_ENABLE_ALL          0x1  /* PMCR.E */
#define PMCR_DISALBE_CYCNT       0x20 /* PMCR.DP */
#define PMCR_RESET_CYCNT         0x4  /* PMCR.C */
#define IMP_BPCTLR_DISABLE_BP    0x7  /* IMP_BPCTLR.DBPEL0DIS/ IMP_BPCTLR.DBPEL1DIS/ IMP_BPCTLR.DBPEL2DIS */

/*===========================================================================*/
/* Private macros.                                                           */
/*===========================================================================*/

/*===========================================================================*/
/* Private types.                                                            */
/*===========================================================================*/
typedef union {
  struct {
    uint32_t  low;
    uint32_t  high;
  } u32;
    uint64_t  u64;
} cycleStamp_t;

/*===========================================================================*/
/* Global variables.                                                         */
/*===========================================================================*/
static volatile cycleStamp_t cycles;
static volatile uint32_t PMCR;
static volatile uint32_t start;

/*===========================================================================*/
/* Local variables.                                                          */
/*===========================================================================*/

/*===========================================================================*/
/* Local functions declaration.                                              */
/*===========================================================================*/

/*===========================================================================*/
/* Global functions.                                                         */
/*===========================================================================*/

void Pmu_counterInit(void) {

  cycles.u64=0;

  /* Enable CYC counter. */
  asm volatile(" mrc p15, 0, %0, c9, c12, 1;" : "=r" (start)); /* needed to make compiler avoid errors */
  start = 0x80000000;
  asm volatile(" mcr p15, 0, %0, c9, c12, 1;" : : "r" (start));

  /* Reset counter and read back register*/
  asm volatile(" mrc p15, 0, %0, c9, c12, 0;" : "=r" (PMCR));
  PMCR |= PMCR_RESET_CYCNT;
  asm volatile(" mcr p15, 0, %0, c9, c12, 0;" : : "r" (PMCR));

  /* Start counter */
  asm volatile(" mrc p15, 0, %0, c9, c12, 0;" : "=r" (PMCR));
  PMCR |= (PMCR_ENABLE_ALL);
  asm volatile(" mcr p15, 0, %0, c9, c12, 0;" : : "r" (PMCR));
}

void Pmu_counterReset(void) {

  /* Reset counter and read back register*/
  asm volatile(" mrc p15, 0, %0, c9, c12, 0;" : "=r" (PMCR));
  PMCR |= PMCR_RESET_CYCNT;
  asm volatile(" mcr p15, 0, %0, c9, c12, 0;" : : "r" (PMCR));
}

uint64_t Pmu_counterGet(void) {

  /* Read counter register value */
  asm volatile(" mrrc p15, 0, %0, %1, c9;"
                : "=r"  (cycles.u32.low),"=r" (cycles.u32.high));
  asm volatile("dsb");
  asm volatile("isb");

  /* cycles values */
  return cycles.u64;
}
