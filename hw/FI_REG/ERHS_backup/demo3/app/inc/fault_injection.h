#ifndef FAULT_INJECTION_H
#define FAULT_INJECTION_H

#include <stdint.h>

/* Durata del calcolo dei layer per inferenza (us), dal report di validazione
 * (2.999-3.036 ms): l'istante del flip è estratto uniforme in [1, FI_LAYERS_US]. */
#define FI_LAYERS_US       2999U

/* Frame salvato dall'hardware all'ingresso dell'interrupt quando il codice
 * interrotto usa la FPU (EXC_RETURN bit 4 = 0): registri interi + S0-S15 + FPSCR. */
typedef struct {
    uint32_t r0, r1, r2, r3, r12, lr, pc, xpsr;
    uint32_t s[16];                  /* S0..S15 */
    uint32_t fpscr;
    uint32_t reserved;
} fi_exc_frame_t;

/* Esito dell'iniezione (da leggere col debugger). */
typedef struct {
    volatile uint32_t done;          /* 1 = flip eseguito in questa inferenza */
    volatile uint32_t reg;           /* 0..12 = R0..R12                */
    volatile uint32_t bit;           /* 0..31                          */
    volatile uint32_t value_before;
    volatile uint32_t value_after;
    volatile uint32_t stacked_pc;    /* istruzione interrotta          */
    volatile uint32_t delay_ticks;   /* istante estratto, in tick di TIM2 (1 tick = 1/150 MHz) */
    volatile uint32_t sample;        /* indice dell'inferenza (0..304) */
    volatile uint32_t node;          /* 1 = conv1d_1, 2 = activation_1, 3 = max_pooling1d_1,
                                        4 = lstm_1, 5 = dense_1 */
} fi_record_t;

extern fi_record_t fi_record;			// struct with all info on the injection
extern volatile uint32_t fi_n_runs;		// number of inferences
extern volatile uint32_t fi_n_injected; // number of fault injections

void FI_Init(void);   /* configura TIM2, non lo avvia */
void FI_Start(void);  /* azzera il conteggio e avvia TIM2 */

#endif /* FAULT_INJECTION_H */
