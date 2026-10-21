#include "fault_injection.h"
#include "tim.h"
#include <platform.h>          /* TIM2, TIM_CR1_CEN, TIM_EGR_UG */
#include "core_cm7.h"
#include "stai.h"
#include "stai_events.h"

// Variables
__attribute__((section(".dtcm"))) fi_record_t fi_record;
__attribute__((section(".dtcm"))) volatile uint32_t fi_n_injected = 0U;
__attribute__((section(".dtcm"))) static volatile uint32_t fi_armed = 0U;
static uint32_t xs_state = 0U;
__attribute__((section(".dtcm"))) volatile uint32_t fi_in_inference = 0U;  /* 1 durante stai_mnetwork_run */
__attribute__((section(".dtcm"))) volatile uint32_t fi_node = 0U;          /* node id del layer in corso */
__attribute__((section(".dtcm"))) volatile uint32_t fi_n_runs = 0U;        /* inferenze completate */
static uint32_t fi_layers_ticks = 0U;     /* FI_LAYERS_US convertito in tick di TIM2 */

// Callbacks
static stai_event_cb fi_user_cb = NULL;    /* callback registrata dalla validazione */

/* xorshift32, seed da SysTick alla prima chiamata. */
static uint32_t xorshift32(void) {
    uint32_t x = xs_state;
    if (x == 0U) { x = SysTick->VAL | 1U; }
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    xs_state = x;
    return x;
}
void FI_Init(void) {
	tim_init(&DRV_TIM2);

	/* tick = 1 ciclo di TIMCLK2 (150 MHz) */
	fi_layers_ticks = (tim_get_clock(&DRV_TIM2) / 1000000U) * FI_LAYERS_US;

    tim_set_prio(&DRV_TIM2, 10);
    tim_set_presc(&DRV_TIM2, TIM_DIVIDE_BY(1U));		/* PSC = 0: nessuna divisione */
    tim_start(&DRV_TIM2);
    tim_set_interval(&DRV_TIM2, fi_layers_ticks);        /*Duration of a single inference*/
    tim_enable_update_callback(&DRV_TIM2);
}

/* TIM2 come "cronometro dei layer": conta solo tra NODE_START e NODE_STOP. */
static inline void fi_timer_pause(void)  { TIM2->CR1 &= ~TIM_CR1_CEN; }
static inline void fi_timer_resume(void) { TIM2->CR1 |=  TIM_CR1_CEN; }

/* Azzera il contatore e imposta l'update dopo 't' tick, timer fermo. */
static void fi_timer_arm(uint32_t t) {
    fi_timer_pause();
    TIM2->ARR = t - 1U;
    TIM2->EGR = TIM_EGR_UG;      /* CNT = 0, carica ARR/PSC (con URS=1 nessun IRQ) */
    TIM2->SR  = 0U;
}
void FI_Start(void) {
    fi_armed = 0U;
    tim_enable(&DRV_TIM2);               /* imposta ARPE/URS e abilita l'IRQ di update */
    fi_timer_pause();                    /* parte solo dentro i layer */
}

/* Callback sugli eventi di layer: tiene aggiornato fi_node e inoltra
 * l'evento alla callback della validazione (observer), se presente. */
static void fi_event_cb(void *cookie, const stai_event_type type, const void *payload) {
    const stai_event_node_start_stop *ev = (const stai_event_node_start_stop *)payload;

    if (type == STAI_EVENT_NODE_STOP) {
        fi_timer_pause();                        /* prima: l'observer non conta */
        fi_node = 0U;
    }
    if (fi_user_cb != NULL) {
        fi_user_cb(cookie, type, payload);
    }
    if (type == STAI_EVENT_NODE_START) {
        fi_node = (uint32_t)ev->node_id;
        if (fi_armed != 0U) {
            fi_timer_resume();                   /* dopo: conta solo il layer */
        }
    }
}

extern stai_return_code __real_stai_mnetwork_set_callback(stai_ptr network, const stai_event_cb cb, void *cb_cookie);
stai_return_code __wrap_stai_mnetwork_set_callback(stai_ptr network,
                                                   const stai_event_cb cb, void *cb_cookie) {
    fi_user_cb = cb;                             /* NULL se l'observer è disattivo */
    return __real_stai_mnetwork_set_callback(network, fi_event_cb, cb_cookie);
}

extern stai_return_code __real_stai_mnetwork_run(stai_ptr network);
stai_return_code __wrap_stai_mnetwork_run(stai_ptr network) {
    stai_return_code ret;

    fi_node = 0U;

	uint32_t t = 1U + (xorshift32() % fi_layers_ticks);   /* 1..fi_layers_ticks */
	fi_record.done     = 0U;
	fi_record.sample   = fi_n_runs;
	fi_record.delay_ticks = t;
	fi_timer_arm(t);
    fi_armed = 1U;

    fi_in_inference = 1U;
    ret = __real_stai_mnetwork_run(network);
    fi_in_inference = 0U;
    fi_armed = 0U;
    fi_timer_pause();
    fi_n_runs++;

    return ret;
}

/* Chiamata dal trampolino.
 * frame : frame HW (R0-R3, R12, LR, PC, xPSR) del contesto interrotto
 * r4_r11: copia di R4..R11 salvata dal trampolino, ricaricata con POP all'uscita
 * exc_ret: valore EXC_RETURN (LR all'ingresso dell'interrupt) */
void fi_dispatch(fi_exc_frame_t *frame, uint32_t *r4_r11, uint32_t exc_ret) {
    extern void __tim_serve_interrupt(tim_driver_t *tdp);
    __tim_serve_interrupt(&DRV_TIM2);            /* ack dell'update */

    if ((fi_armed == 0U) || (fi_node == 0U)) {

        return;
    }
    fi_armed = 0U;
    fi_timer_pause();                            /* un solo flip per inferenza */

    uint32_t reg = xorshift32() % 13U;           /* R0..R12 */
    uint32_t bit = xorshift32() & 0x1FU;         /* 0..31   */

    volatile uint32_t *target;
    if (reg <= 3U) {
        target = &((volatile uint32_t *)frame)[reg];   /* R0..R3 */
    } else if (reg == 12U) {
        target = &frame->r12;                          /* R12    */
    } else {
        target = &r4_r11[reg - 4U];                    /* R4..R11 */
    }

    fi_record.reg          = reg;
    fi_record.bit          = bit;
    fi_record.value_before = *target;
    *target ^= (1U << bit);
    fi_record.value_after  = *target;
    fi_record.stacked_pc   = frame->pc;
    fi_record.exc_return   = exc_ret;
    fi_record.done         = 1U;
    fi_record.node         = fi_node;
    fi_n_injected++;
}

/* TIM2 IRQ: salva tutto, chiama il dispatcher, ripristina tutto
 * (con l'eventuale bit flippato). */
__attribute__((naked)) void IRQ_TIM2_HANDLER(void) {
    __asm volatile (
        "TST    LR, #4          \n"   /* thread interrotto su MSP o PSP?  */
        "ITE    EQ              \n"
        "MRSEQ  R0, MSP         \n"   /* R0 = frame HW                    */
        "MRSNE  R0, PSP         \n"
        "PUSH   {R4-R11}        \n"   /* copia di R4..R11                 */
        "MOV    R1, SP          \n"   /* R1 = indirizzo della copia       */
        "MOV    R2, LR          \n"   /* R2 = EXC_RETURN                  */
        "PUSH   {R4, LR}        \n"   /* salva EXC_RETURN (R4 serve solo a mantenere SP allineato a 8 byte)  */
        "BL     fi_dispatch     \n"	  /* fi_dispatch(frame, r4_r11, exc_ret)*/
        "POP    {R4, LR}        \n"
        "POP    {R4-R11}        \n"   /* ricarica R4..R11 DALLA COPIA, eventualmente modificata */
        "BX     LR              \n"   /* l'HW ricarica R0-R3, R12 dal frame */
    );
}









