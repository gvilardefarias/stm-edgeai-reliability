#include "fault_injection.h"
#include "tim.h"
#include <platform.h>          /* TIM2, TIM_CR1_CEN, TIM_EGR_UG */
#include "core_cm7.h"
#include "stai.h"
#include "stai_events.h"
#include "app_stellar-studio-ai.h" 		// stai_mnetwork_get_outputs, data_activations)
#include <setjmp.h>
#include <math.h>

/* Stato del meccanismo per l'inferenza in corso. */
#define FI_IDLE            0U        /* nessuna inferenza armata          */
#define FI_WAIT_FLIP       1U        /* TIM2 conta fino all'istante del flip */
#define FI_WATCHDOG        2U        /* flip fatto: TIM2 conta fino al timeout */

// Variables
__attribute__((section(".dtcm"))) fi_record_t fi_record;
__attribute__((section(".dtcm"))) volatile uint32_t fi_n_injected = 0U;
__attribute__((section(".dtcm"))) volatile uint32_t fi_n_crash = 0U;
__attribute__((section(".dtcm"))) volatile uint32_t fi_n_hang = 0U;
__attribute__((section(".dtcm"))) static volatile uint32_t fi_phase = FI_IDLE;
static uint32_t xs_state = 0U;
__attribute__((section(".dtcm"))) volatile uint32_t fi_in_inference = 0U;  /* 1 durante stai_mnetwork_run */
__attribute__((section(".dtcm"))) volatile uint32_t fi_node = 0U;          /* node id del layer in corso */
__attribute__((section(".dtcm"))) volatile uint32_t fi_n_runs = 0U;        /* inferenze completate */
static uint32_t fi_layers_ticks = 0U;     /* FI_LAYERS_US convertito in tick di TIM2 */
static uint32_t fi_timeout_ticks = 0U;    /* FI_TIMEOUT_US convertito in tick di TIM2 */

/* Recupero da crash/stallo: punto di ritorno nel wrapper. */
static jmp_buf fi_jmp;
static volatile uint32_t fi_jmp_valid = 0U;   /* 1 mentre fi_jmp è utilizzabile */
static volatile uint32_t fi_stops_done = 0U;  /* NODE_STOP già inoltrati all'observer */

// Callbacks
static stai_event_cb fi_user_cb = NULL;    /* callback registrata dalla validazione */
static void *fi_user_cookie = NULL;

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
	uint32_t ticks_per_us;

	tim_init(&DRV_TIM2);

	/* tick = 1 ciclo di TIMCLK2 (150 MHz) */
	ticks_per_us     = tim_get_clock(&DRV_TIM2) / 1000000U;
	fi_layers_ticks  = ticks_per_us * FI_LAYERS_US;
	fi_timeout_ticks = ticks_per_us * FI_TIMEOUT_US;

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
    fi_phase = FI_IDLE;
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
    if (type == STAI_EVENT_NODE_STOP) {
        fi_stops_done++;                         /* layer chiuso anche per l'observer */
    }
    if (type == STAI_EVENT_NODE_START) {
        fi_node = (uint32_t)ev->node_id;
        if (fi_phase != FI_IDLE) {
            fi_timer_resume();                   /* conta anche durante il watchdog */
        }
    }
}

extern stai_return_code __real_stai_mnetwork_set_callback(stai_ptr network, const stai_event_cb cb, void *cb_cookie);
stai_return_code __wrap_stai_mnetwork_set_callback(stai_ptr network,
                                                   const stai_event_cb cb, void *cb_cookie) {
    fi_user_cb = cb;                             /* NULL se l'observer è disattivo */
    fi_user_cookie = cb_cookie;
    return __real_stai_mnetwork_set_callback(network, fi_event_cb, cb_cookie);
}

/* Inferenza interrotta: manda all'observer gli eventi dei layer non completati,
 * così l'host riceve la stessa sequenza di messaggi di un'inferenza normale. */
static void fi_send_event(stai_event_type type, uint32_t id) {
    const stai_ptr bufs[1] = { data_activations[0] };   /* RAM valida, dati non significativi */
    const stai_event_node_start_stop ev = {
        .node_id = (int32_t)id,
        .buffers = { .size = 1, .data = bufs }
    };
    if (fi_user_cb != NULL) {
        fi_user_cb(fi_user_cookie, type, &ev);
    }
}
static void fi_complete_events(uint32_t node) {
    uint32_t id = fi_stops_done;                 /* layer 1..id già chiusi */

    if (node != 0U) {                            /* interrotta dentro un layer */
        fi_send_event(STAI_EVENT_NODE_STOP, node);
        id = node;
    }
    for (id = id + 1U; id <= FI_N_NODES; id++) {
        fi_send_event(STAI_EVENT_NODE_START, id);
        fi_send_event(STAI_EVENT_NODE_STOP, id);
    }
}

extern stai_return_code __real_stai_mnetwork_run(stai_ptr network);
stai_return_code __wrap_stai_mnetwork_run(stai_ptr network) {
    stai_return_code ret;
    uint32_t t = 1U + (xorshift32() % fi_layers_ticks);   /* 1..fi_layers_ticks */

    fi_node = 0U;
    fi_stops_done = 0U;
    fi_record.done        = 0U;
    fi_record.sample      = fi_n_runs;
    fi_record.delay_ticks = t;
    fi_record.abort_cause = FI_ABORT_NONE;
    fi_timer_arm(t);
    fi_phase = FI_WAIT_FLIP;

    if (setjmp(fi_jmp) == 0) {
        fi_jmp_valid = 1U;
        fi_in_inference = 1U;
        ret = __real_stai_mnetwork_run(network);
    } else {
        /* Qui si arriva solo da fi_abort(): crash o stallo dopo il flip. */
        stai_ptr outs[1];
        uint32_t node = fi_node;

        fi_jmp_valid = 0U;
        fi_in_inference = 0U;
        fi_phase = FI_IDLE;
        fi_timer_pause();
        fi_node = 0U;

        fi_complete_events(node);                /* sequenza completa per l'observer */
        if (stai_mnetwork_get_outputs(network, outs) == STAI_SUCCESS) {
            ((float *)outs[0])[0] = NAN;         /* sentinella: inferenza non valida */
        }
        ret = STAI_SUCCESS;                      /* la validazione prosegue */
    }
    fi_jmp_valid = 0U;
    fi_in_inference = 0U;
    fi_phase = FI_IDLE;
    fi_timer_pause();
    fi_n_runs++;

    return ret;
}

/* Eseguita in thread mode al ritorno dall'eccezione (vedi fi_abort). */
__attribute__((noinline, noreturn)) static void fi_abort_to_wrapper(void) {
    longjmp(fi_jmp, 1);
}

/* Fa riprendere il codice interrotto da fi_abort_to_wrapper invece che
 * dall'istruzione interrotta. Ritorna 0 se non è possibile. */
static uint32_t fi_abort(fi_exc_frame_t *frame, uint32_t exc_ret) {
    fi_timer_pause();
    fi_phase = FI_IDLE;
    if (((exc_ret & 0x8U) == 0U) || (fi_jmp_valid == 0U)) {
        return 0U;                               /* non in thread mode o fuori dall'inferenza */
    }
    frame->pc   = (uint32_t)&fi_abort_to_wrapper & ~1U;
    frame->xpsr = (frame->xpsr & (1U << 9)) | (1U << 24);   /* solo bit Thumb (+ allineamento) */
    return 1U;
}

/* Chiamata dal trampolino.
 * frame : frame HW (R0-R3, R12, LR, PC, xPSR) del contesto interrotto
 * r4_r11: copia di R4..R11 salvata dal trampolino, ricaricata con POP all'uscita
 * exc_ret: valore EXC_RETURN (LR all'ingresso dell'interrupt) */
void fi_dispatch(fi_exc_frame_t *frame, uint32_t *r4_r11, uint32_t exc_ret) {
    __tim_serve_interrupt(&DRV_TIM2);            /* ack dell'update */

    if (fi_phase == FI_WATCHDOG) {               /* FI_TIMEOUT_US di calcolo dopo il flip */
        fi_record.abort_cause = FI_ABORT_HANG;
        fi_record.fault_pc    = frame->pc;
        fi_n_hang++;
        (void)fi_abort(frame, exc_ret);
        return;
    }
    if ((fi_phase != FI_WAIT_FLIP) || (fi_node == 0U)) {
        return;                                  /* fuori da un layer: nessun flip */
    }

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

    /* Da qui TIM2 fa da watchdog: conta il calcolo dei layer dopo il flip. */
    fi_phase = FI_WATCHDOG;
    fi_timer_arm(fi_timeout_ticks);
    fi_timer_resume();                           /* siamo dentro un layer */
}

/* Fault (BusFault/UsageFault/MemManage escalati a HardFault) durante
 * un'inferenza con flip: registra la causa e interrompe l'inferenza. */
void fi_fault_dispatch(fi_exc_frame_t *frame, uint32_t exc_ret) {
    uint32_t cfsr = SCB->CFSR;

    /* Errori durante il salvataggio/ripristino del frame: frame non affidabile. */
    const uint32_t stack_err = (1U << 3) | (1U << 4) | (1U << 5) |      /* MUNSTKERR, MSTKERR, MLSPERR */
                               (1U << 11) | (1U << 12) | (1U << 13);    /* UNSTKERR, STKERR, LSPERR */

    if ((fi_in_inference != 0U) && ((cfsr & stack_err) == 0U)) {
        fi_record.abort_cause = FI_ABORT_CRASH;
        fi_record.fault_pc    = frame->pc;
        SCB->CFSR = cfsr;                        /* azzera i flag per il prossimo fault */
        SCB->HFSR = SCB->HFSR;
        fi_n_crash++;
        if (fi_abort(frame, exc_ret) != 0U) {
            return;                              /* ritorno dall'eccezione -> fi_abort_to_wrapper */
        }
    }
    for (;;) {                                   /* fault non recuperabile: come prima */
    }
}

__attribute__((naked)) void HardFault_Handler(void) {
    __asm volatile (
        "TST    LR, #4          \n"   /* fault in codice su MSP o PSP?    */
        "ITE    EQ              \n"
        "MRSEQ  R0, MSP         \n"   /* R0 = frame HW                    */
        "MRSNE  R0, PSP         \n"
        "MOV    R1, LR          \n"   /* R1 = EXC_RETURN                  */
        "B      fi_fault_dispatch \n" /* il suo return fa da ritorno dall'eccezione */
    );
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









