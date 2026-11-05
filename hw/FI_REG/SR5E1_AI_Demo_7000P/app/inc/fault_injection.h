#ifndef FAULT_INJECTION_H
#define FAULT_INJECTION_H

#include <stdint.h>

#define FI_TIMEOUT_US      30000U		/* WatchDog Timer for long Stalls between inferences: 30 ms */
#define FI_N_NODES         5U
#define FI_NODE_IDS        { 3U, 4U, 5U, 7U, 8U }   /* conv2d, conv2d_1, global_max_pooling2d_pool, dense, activation
													(softmax) */
#define FI_INPUT_NODE_ID   3U    					/* only layer that reads the input*/

#define FI_ABORT_NONE      0U
#define FI_ABORT_CRASH     1U
#define FI_ABORT_HANG      2U

/* Time period for layer computation per inference (us). (1.483-1.494 ms)
 * The flip instant is uniformely extracted in [1, FI_LAYERS_US]. */
#define FI_LAYERS_US       1483U

/* Saved Hardware frame at Interrupt beginning. */
typedef struct {
    uint32_t r0, r1, r2, r3, r12, lr, pc, xpsr;
    uint32_t s[16];					/* S0..S15 (EXC_RETURN bit 4 = 0) */
    uint32_t fpscr;
    uint32_t reserved;
} fi_exc_frame_t;

/* Injection result (Debugger). */
typedef struct {
    volatile uint32_t done;          /* 1 = Flip executed in this inference */
    volatile uint32_t reg;           /* 0..12 = R0..R12, 13..44 = S0..S31 */
    volatile uint32_t bit;           /* 0..31                          */
    volatile uint32_t value_before;
    volatile uint32_t value_after;
    volatile uint32_t stacked_pc;    /* Interrupted istruction           */
    volatile uint32_t exc_return;
    volatile uint32_t delay_ticks;   /* extracted time instant, in TIM2 ticks (1 tick = 1/150 MHz) */
    volatile uint32_t sample;        /* Inference index */
    volatile uint32_t node;          /* node id: 3 = conv2d, 4 = conv2d_1, 5 = global_max_pooling2d_pool,
                                        7 = dense, 8 = activation (softmax) */
    volatile uint32_t abort_cause;   /* FI_ABORT_NONE / FI_ABORT_CRASH / FI_ABORT_HANG */
    volatile uint32_t fault_pc;      /* istruction that caused the hardfault */

} fi_record_t;

typedef struct {
    volatile uint8_t dummy;
    volatile uint8_t done;          /* 1 = Flip executed in this inference */
    volatile uint8_t reg;           /* 0..12 = R0..R12, 13..44 = S0..S31 */
    volatile uint8_t bit;           /* 0..31                          */
    volatile uint8_t node;          /* node id: 3 = conv2d, 4 = conv2d_1, 5 = global_max_pooling2d_pool,
                                        7 = dense, 8 = activation (softmax) */
    volatile uint8_t abort_cause;   /* FI_ABORT_NONE / FI_ABORT_CRASH / FI_ABORT_HANG */
    volatile uint32_t delay_ticks;   /* Extracted time instant, in TIM2 ticks ( from 1 to 222 450 = 1483 µs × 150 tick/µs ) */
} fi_log_t;

extern fi_log_t fi_log;			        // struct with important info on the injection
extern fi_record_t fi_record;			// struct with all info on the injection
extern volatile uint32_t fi_n_runs;		// number of inferences
extern volatile uint32_t fi_n_injected; // number of fault injections
extern volatile uint32_t fi_n_crash;	// number of crashes
extern volatile uint32_t fi_n_hang;

void FI_Init(void);   /* TIM2 and MPU configuration, starts and put the timer in pause */

#endif /* FAULT_INJECTION_H */
