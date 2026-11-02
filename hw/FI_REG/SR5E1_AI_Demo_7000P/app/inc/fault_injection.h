#ifndef FAULT_INJECTION_H
#define FAULT_INJECTION_H

#include <stdint.h>

#define FI_TIMEOUT_US      30000U		// WatchDog Timer for long Stalls between inferences
#define FI_N_NODES         5U
#define FI_ABORT_NONE      0U
#define FI_ABORT_CRASH     1U
#define FI_ABORT_HANG      2U

/* Durata del calcolo dei layer per inferenza (us), dal report di validazione
 * (2.999-3.036 ms): l'istante del flip è estratto uniforme in [1, FI_LAYERS_US]. */
#define FI_LAYERS_US       2999U

/* Saved Hardware frame at Interrupt beginning. */
typedef struct {
    uint32_t r0, r1, r2, r3, r12, lr, pc, xpsr;
    uint32_t s[16];					/* S0..S15 (EXC_RETURN bit 4 = 0) */
    uint32_t fpscr;
    uint32_t reserved;
} fi_exc_frame_t;

/* Esito dell'iniezione (da leggere col debugger). */
typedef struct {
    volatile uint32_t done;          /* 1 = Flip executed in this inference */
    volatile uint32_t reg;           /* 0..12 = R0..R12, 13..44 = S0..S31 */
    volatile uint32_t bit;           /* 0..31                          */
    volatile uint32_t value_before;
    volatile uint32_t value_after;
    volatile uint32_t stacked_pc;    /* istruzione interrotta          */
    volatile uint32_t exc_return;
    volatile uint32_t delay_ticks;   /* istante estratto, in tick di TIM2 (1 tick = 1/150 MHz) */
    volatile uint32_t sample;        /* indice dell'inferenza (0..304) */
    volatile uint32_t node;          /* 1 = conv1d_1, 2 = activation_1, 3 = max_pooling1d_1,
                                        4 = lstm_1, 5 = dense_1 */
    volatile uint32_t abort_cause;   /* FI_ABORT_NONE / FI_ABORT_CRASH / FI_ABORT_HANG */
    volatile uint32_t fault_pc;      /* istruzione che ha causato il fault */

} fi_record_t;

typedef struct {
    volatile uint8_t dummy;
    volatile uint8_t done;          /* 1 = flip eseguito in questa inferenza */
    volatile uint8_t reg;           /* 0..12 = R0..R12, 13..44 = S0..S31 */
    volatile uint8_t bit;           /* 0..31                          */
    volatile uint8_t node;          /* 1 = conv1d_1, 2 = activation_1, 3 = max_pooling1d_1,
                                        4 = lstm_1, 5 = dense_1 */
    volatile uint8_t abort_cause;   /* FI_ABORT_NONE / FI_ABORT_CRASH / FI_ABORT_HANG */
    volatile uint32_t delay_ticks;   /* Extracted time instant, in TIM2 ticks ( from 0 to 449 850 = 2999 µs × 150 tick/µs ) */
} fi_log_t;

extern fi_log_t fi_log;			        // struct with important info on the injection
extern fi_record_t fi_record;			// struct with all info on the injection
extern volatile uint32_t fi_n_runs;		// number of inferences
extern volatile uint32_t fi_n_injected; // number of fault injections
extern volatile uint32_t fi_n_crash;	// number of crashes
extern volatile uint32_t fi_n_hang;

void FI_Init(void);   /* configura TIM2, non lo avvia */
void FI_Start(void);  /* azzera il conteggio e avvia TIM2 */

#endif /* FAULT_INJECTION_H */
