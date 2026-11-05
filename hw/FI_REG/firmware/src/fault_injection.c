#include "fault_injection.h"
#include "tim.h"
#include <platform.h>          /* TIM2, TIM_CR1_CEN, TIM_EGR_UG */
#include "core_cm7.h"
#include "stai.h"
#include "stai_events.h"
#include "app_stellar-studio-ai.h" 		/* stai_mnetwork_get_outputs, data_activations */
#include "network.h"                    /* STAI_NETWORK_* (dimentions activations, input, output) */
#include <setjmp.h>
#include <math.h>
#include <errno.h>

/* Possible states during the inference process */
#define FI_IDLE            0U        /* No Armed Inference state          */
#define FI_WAIT_FLIP       1U        /* TIM2 counts until bit flip instant */
#define FI_WATCHDOG        2U        /* flip done: TIM2 counts until timeout */

/* Activations (fi_act substitute activations_1, 8 KB aligned) */
#define FI_ACT_SIZE        STAI_NETWORK_ACTIVATION_1_SIZE_BYTES          	/* 4352 B */
#define FI_ACT_SUB         1024U                                          	/* subregion */
#define FI_ACT_N_SUB       ((FI_ACT_SIZE + FI_ACT_SUB - 1U) / FI_ACT_SUB) 	/* 5 */
#define FI_ACT_ALIGN       (8U * FI_ACT_SUB)                              	/* 8 KB */
_Static_assert(FI_ACT_N_SUB <= 8U, "activations too large for a 8 KB region");

/* Input (network.c: _inputs[0] = activations + 4032, 288 B)
 *   6: fi_act + 0xFC0, 64 B                                    -> [4032, 4096)
 *   7: fi_act + 0x1000, 256 B, subregions of 32 B, 7 excluded  -> [4096, 4320) */
#define FI_IN_OFFSET       4032U
_Static_assert((FI_IN_OFFSET == 0xFC0U) && (STAI_NETWORK_IN_1_SIZE_BYTES == 288U),
               "Input moved: computes MPU regions 6 e 7");
#define FI_MPU_IN_A_RASR   ARM_MPU_RASR(0U, ARM_MPU_AP_RO, 1U, 0U, 1U, 1U, 0x00U, ARM_MPU_REGION_SIZE_64B)
#define FI_MPU_IN_B_RASR   ARM_MPU_RASR(0U, ARM_MPU_AP_RO, 1U, 0U, 1U, 1U, 0x80U, ARM_MPU_REGION_SIZE_256B)

extern uint8_t __main_stack_base__[];     /* simboli del linker script */
extern uint8_t __process_stack_end__[];

__attribute__((aligned(FI_ACT_ALIGN))) static uint8_t fi_act[FI_ACT_N_SUB * FI_ACT_SUB];

static const uint8_t fi_node_ids[FI_N_NODES] = FI_NODE_IDS;
_Static_assert(FI_N_NODES == STAI_NETWORK_NODES_NUM, "numero di layer cambiato: aggiornare FI_NODE_IDS");

// Variables
__attribute__((section(".dtcm"))) fi_record_t fi_record;
__attribute__((section(".dtcm"))) fi_log_t fi_log;
__attribute__((section(".dtcm"))) volatile uint32_t fi_n_injected = 0U;
__attribute__((section(".dtcm"))) volatile uint32_t fi_n_crash = 0U;
__attribute__((section(".dtcm"))) volatile uint32_t fi_n_hang = 0U;
__attribute__((section(".dtcm"))) static volatile uint32_t fi_phase = FI_IDLE;
__attribute__((section(".dtcm"))) static uint32_t xs_state = 0U;   				/* written in ISR with MPU active */
__attribute__((section(".dtcm"))) static uint32_t fi_sp_top = 0U;  				/* PSP at network beginning, up-rounded to 32 B */
__attribute__((section(".dtcm"))) volatile uint32_t fi_in_inference = 0U;  		/* 1 during stai_mnetwork_run */
__attribute__((section(".dtcm"))) static volatile uint32_t fi_node_idx = 0U; 	/* posizione 1..FI_N_NODES (0 = tra layer) */
__attribute__((section(".dtcm"))) volatile uint32_t fi_n_runs = 0U;        		/* inferenze completate */
static uint32_t fi_layers_ticks = 0U;     /* FI_LAYERS_US convertito in tick di TIM2 */
static uint32_t fi_timeout_ticks = 0U;    /* FI_TIMEOUT_US convertito in tick di TIM2 */

/* Recupero da crash/stallo: punto di ritorno nel wrapper. */
static jmp_buf fi_jmp;
static volatile uint32_t fi_stops_done = 0U;  /* NODE_STOP già inoltrati all'observer */

// Callbacks
static stai_event_cb fi_user_cb = NULL;    /* callback registrata dalla validazione */
static void *fi_user_cookie = NULL;

/* xorshift32, seed from SysTick at first call. */
static uint32_t xorshift32(void) {
    uint32_t x = xs_state;
    if (x == 0U) { x = SysTick->VAL | 1U; }
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    xs_state = x;
    return x;
}

/* Regione 'rnr' su [a, b): 8 sottoregioni da g byte (g >= 32), la più fine possibile.
 * outward = 1: copre tutto [a,b) (fino a g byte in più ai bordi)
 * outward = 0: niente sotto 'a' (i primi < g byte possono restare scoperti) */
static uint32_t fi_mpu_cover(uint32_t rnr, uint32_t a, uint32_t b, uint32_t ap, uint32_t outward) {
    uint32_t g;
    for (g = 32U; g <= 0x08000000U; g <<= 1) {
        uint32_t size = 8U * g;
        uint32_t base = a & ~(size - 1U);
        uint32_t lo   = (a - base) / g;
        uint32_t hi   = (b - base + g - 1U) / g;
        if ((outward == 0U) && (((a - base) % g) != 0U)) { lo++; }
        if ((hi <= 8U) && (lo < hi)) {
            uint32_t srd = ~(((1U << hi) - 1U) & ~((1U << lo) - 1U)) & 0xFFU;
            uint32_t size_field = (31U - __CLZ(size)) - 1U;   /* log2(size) - 1 */
            ARM_MPU_SetRegion(ARM_MPU_RBAR(rnr, base),
                ARM_MPU_RASR(0U, ap, 1U, 0U, 1U, 1U, srd, size_field));
            if (outward != 0U) { return (a - (base + lo * g)) + ((base + hi * g) - b); }
            return (base + lo * g) - a;
        }
    }
    ARM_MPU_ClrRegion(rnr);
    return 0xFFFFFFFFU;
}

/* [a, b) in scrittura con 2 regioni, divise sul confine più allineato (stack non allineati). */
static uint32_t fi_mpu_cover_rw2(uint32_t rnr_lo, uint32_t rnr_hi, uint32_t a, uint32_t b) {
    int32_t k;
    for (k = 31; k >= 5; k--) {
        uint32_t m = b & ~((1U << (uint32_t)k) - 1U);
        if ((m > a) && (m < b)) {
            return fi_mpu_cover(rnr_lo, a, m, ARM_MPU_AP_FULL, 1U) +
                   fi_mpu_cover(rnr_hi, m, b, ARM_MPU_AP_FULL, 1U);
        }
    }
    ARM_MPU_ClrRegion(rnr_hi);
    return fi_mpu_cover(rnr_lo, a, b, ARM_MPU_AP_FULL, 1U);
}

/* Regioni MPU attive solo durante i layer (vince il numero più alto).
 * Attributi = default SRAM: Normal, write-back write-allocate (TEX=1,C=1,B=1), non shareable. */
static void fi_mpu_init(void) {
    const uint32_t act = (uint32_t)fi_act;

    MPU->CTRL = 0U;
    __DSB(); __ISB();

    ARM_MPU_SetRegion(ARM_MPU_RBAR(0U, 0x24000000U),
        ARM_MPU_RASR(0U, ARM_MPU_AP_RO,   1U, 0U, 1U, 1U, 0x00U, ARM_MPU_REGION_SIZE_256KB));
    ARM_MPU_SetRegion(ARM_MPU_RBAR(1U, act),
        ARM_MPU_RASR(0U, ARM_MPU_AP_FULL, 1U, 0U, 1U, 1U,
                     (uint8_t)(0xFFU << FI_ACT_N_SUB), ARM_MPU_REGION_SIZE_8KB));
    (void) fi_mpu_cover_rw2(2U, 3U, (uint32_t)__main_stack_base__, (uint32_t)__process_stack_end__);
    ARM_MPU_ClrRegion(4U);
    (void)fi_mpu_cover(5U, (uint32_t)&errno, (uint32_t)&errno + sizeof(errno), ARM_MPU_AP_FULL, 1U);
    ARM_MPU_SetRegion(ARM_MPU_RBAR(6U, act + FI_IN_OFFSET), 0U);         /* [4032, 4096) */
    ARM_MPU_SetRegion(ARM_MPU_RBAR(7U, act + FI_IN_OFFSET + 64U), 0U);   /* [4096, 4320) */
}
/* Accende la protezione: solo lo stack sotto fi_sp_top resta scrivibile,
 * SysTick (4) e UART (5) mascherati, TIM2 (2) no.
 * MEMFAULTENA resta 0: il fault escala a HardFault -> fi_fault_dispatch. */
static void fi_mpu_on(void) {
	const uint32_t in = ((fi_node_idx != 0U) && (fi_node_ids[fi_node_idx - 1U] == FI_INPUT_NODE_ID)) ? 1U : 0U;

    MPU->RNR  = 6U;
    MPU->RASR = (in != 0U) ? FI_MPU_IN_A_RASR : 0U;
    MPU->RNR  = 7U;
    MPU->RASR = (in != 0U) ? FI_MPU_IN_B_RASR : 0U;

    __set_BASEPRI(4U << (8U - __NVIC_PRIO_BITS));
    MPU->CTRL = MPU_CTRL_PRIVDEFENA_Msk | MPU_CTRL_ENABLE_Msk;
    __DSB(); __ISB();
}
static void fi_mpu_off(void) {
    MPU->CTRL = 0U;
    __DSB(); __ISB();
    __set_BASEPRI(0U);
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

void FI_Init(void) {
	uint32_t ticks_per_us;

	tim_init(&DRV_TIM2);

	/* tick = 1 ciclo di TIMCLK2 (150 MHz) */
	ticks_per_us     = tim_get_clock(&DRV_TIM2) / 1000000U;
	fi_layers_ticks  = ticks_per_us * FI_LAYERS_US;
	fi_timeout_ticks = ticks_per_us * FI_TIMEOUT_US;

    tim_set_prio(&DRV_TIM2, 2);
    tim_set_presc(&DRV_TIM2, TIM_DIVIDE_BY(1U));		/* PSC = 0: nessuna divisione */
    tim_start(&DRV_TIM2);
    tim_enable_update_callback(&DRV_TIM2);

    data_activations[0] = fi_act;
    fi_mpu_init();

    tim_enable(&DRV_TIM2);               /* imposta ARPE/URS e abilita l'IRQ di update */
    fi_timer_pause();                    /* parte solo dentro i layer */
}

/* Callback on layer events: It updates fi_node_idx and send
 * the event to validation callback (observer), if present. */
static void fi_event_cb(void *cookie, const stai_event_type type, const void *payload) {
    const stai_event_node_start_stop *ev = (const stai_event_node_start_stop *)payload;

    if (type == STAI_EVENT_NODE_STOP) {
    	fi_mpu_off();						/* prima di tutto: l'observer deve poter scrivere */

        fi_timer_pause();                        /* prima: l'observer non conta */
        fi_node_idx = 0U;
    }
    if (fi_user_cb != NULL) {
        fi_user_cb(cookie, type, payload);
    }
    if (type == STAI_EVENT_NODE_STOP) {
        fi_stops_done++;                         /* layer chiuso anche per l'observer */
    }
    if (type == STAI_EVENT_NODE_START) {
        uint32_t i;

        fi_node_idx = 0U;
        for (i = 0U; i < FI_N_NODES; i++) {      /* node idx -> posizione 1..FI_N_NODES */
            if ((uint32_t)fi_node_ids[i] == (uint32_t)ev->node_id) {
                fi_node_idx = i + 1U;
                break;
            }
        }

        if (fi_phase != FI_IDLE) {
            fi_timer_resume();                  /* conta anche durante il watchdog */
            fi_mpu_on();						/* ultima cosa: da qui solo il layer */
        }
    }
}

extern stai_return_code __real_stai_mnetwork_set_callback(stai_ptr network, const stai_event_cb cb, void *cb_cookie);
stai_return_code __wrap_stai_mnetwork_set_callback(stai_ptr network,
                                                   const stai_event_cb cb, void *cb_cookie) {
    if (fi_layers_ticks == 0U) {                 /* FI_Init non chiamata: nessuna interferenza */
        return __real_stai_mnetwork_set_callback(network, cb, cb_cookie);
    }
    fi_user_cb = cb;
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
static void fi_complete_events(uint32_t idx) {
    uint32_t k = fi_stops_done;                  /* layer in posizione 1..k già chiusi */
    if (idx != 0U) {
        fi_send_event(STAI_EVENT_NODE_STOP, fi_node_ids[idx - 1U]);
        k = idx;
    }
    for (k = k + 1U; k <= FI_N_NODES; k++) {
        fi_send_event(STAI_EVENT_NODE_START, fi_node_ids[k - 1U]);
        fi_send_event(STAI_EVENT_NODE_STOP,  fi_node_ids[k - 1U]);
    }
}

extern stai_return_code __real_stai_mnetwork_run(stai_ptr network);
stai_return_code __wrap_stai_mnetwork_run(stai_ptr network) {
    stai_return_code ret;
    uint32_t t;

    if (fi_layers_ticks == 0U) {                 /* FI_Init non chiamata: inferenza normale */
        return __real_stai_mnetwork_run(network);
    }
    t = 1U + (xorshift32() % fi_layers_ticks);   /* 1..fi_layers_ticks */

    fi_node_idx = 0U;
    fi_stops_done = 0U;
    fi_record.done        = 0U;
    fi_record.sample      = fi_n_runs;
    fi_record.delay_ticks = t;
    fi_record.abort_cause = FI_ABORT_NONE;

    fi_log.done        = 0U;
    fi_log.delay_ticks = (uint32_t) t;
    fi_log.abort_cause = FI_ABORT_NONE;

    fi_timer_arm(t);
    fi_phase = FI_WAIT_FLIP;

    if (setjmp(fi_jmp) == 0) {
        fi_in_inference = 1U;
        fi_sp_top = (__get_PSP() + 31U) & ~31U;  /* sotto qui: solo frame della rete */
        /* Regione 4: frame dei chiamanti (sopra fi_sp_top) in sola lettura. MPU spenta qui. */
        (void) fi_mpu_cover(4U, fi_sp_top, (uint32_t)__process_stack_end__, ARM_MPU_AP_RO, 0U);

        ret = __real_stai_mnetwork_run(network);
    } else {
        /* Qui si arriva solo da fi_abort(): crash o stallo dopo il flip. */
        stai_ptr outs[1];
        uint32_t idx = fi_node_idx;
        uint32_t i;

        fi_in_inference = 0U;
        fi_phase = FI_IDLE;
        fi_timer_pause();
        fi_node_idx = 0U;

        fi_complete_events(idx);                 /* sequenza completa per l'observer */
        if (stai_mnetwork_get_outputs(network, outs) == STAI_SUCCESS) {
            for (i = 0U; i < STAI_NETWORK_OUT_1_SIZE; i++) {
                ((float *)outs[0])[i] = NAN;     /* sentinella su tutte le classi: inferenza non valida */
            }
        }
        ret = STAI_SUCCESS;                      /* la validazione prosegue */
    }

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
	fi_mpu_off();                                /* il wrapper riparte con SP sopra fi_sp_top */
	fi_timer_pause();
    fi_phase = FI_IDLE;
    if (((exc_ret & 0x8U) == 0U) || (fi_in_inference == 0U)) {
        return 0U;                               /* non in thread mode o fuori dall'inferenza */
    }
    frame->pc   = (uint32_t)&fi_abort_to_wrapper & ~1U;
    frame->xpsr = (frame->xpsr & (1U << 9)) | (1U << 24);   /* solo bit Thumb (+ allineamento) */
    return 1U;
}

/* Dispatch call.
 * frame : frame HW (R0-R3, R12, LR, PC, xPSR, S0-S15, FPSCR) of interrupted context
 * r4_r11: copy of R4..R11
 * s16_s31: copy of S16-S31
 * exc_ret: EXC_RETURN value (LR at interrupt beginning) */
void fi_dispatch(fi_exc_frame_t *frame, uint32_t *r4_r11, uint32_t exc_ret, uint32_t *s16_s31) {
    __tim_serve_interrupt(&DRV_TIM2);            /* ack dell'update */

    if (fi_phase == FI_WATCHDOG) {               /* FI_TIMEOUT_US di calcolo dopo il flip */
        fi_record.abort_cause = FI_ABORT_HANG;
        fi_record.fault_pc    = frame->pc;
        fi_n_hang++;

        fi_log.abort_cause = FI_ABORT_HANG;

        (void)fi_abort(frame, exc_ret);
        return;
    }
    if ((fi_phase != FI_WAIT_FLIP) || (fi_node_idx == 0U)) {
        return;                                  /* fuori da un layer: nessun flip */
    }

    /* 0..12 = R0..R12, 13..44 = S0..S31. Without FPU context in the frame
     * if (EXC_RETURN bit 4 = 1) the choice is between R0..R12 only */
    uint32_t reg = xorshift32() % (((exc_ret & 0x10U) == 0U) ? 45U : 13U);
    uint32_t bit = xorshift32() & 0x1FU;         /* 0..31   */

    volatile uint32_t *target;
    if (reg <= 3U) {
        target = &((volatile uint32_t *)frame)[reg];   	/* R0..R3 */
    } else if (reg == 12U) {
        target = &frame->r12;                          	/* R12 */
    } else if (reg < 12U) {
        target = &r4_r11[reg - 4U];                    	/* R4..R11 */
    } else if (reg < 29U) {
    	target = &frame->s[reg - 13U];                	/* S0..S15   */
    } else {
    	target = &s16_s31[reg - 29U];					/* S16..S31 */
    }

    fi_record.reg          = reg;
    fi_record.bit          = bit;
    fi_record.value_before = *target;
    *target ^= (1U << bit);
    fi_record.value_after  = *target;
    fi_record.stacked_pc   = frame->pc;
    fi_record.exc_return   = exc_ret;
    fi_record.done         = 1U;
    fi_record.node         = fi_node_ids[fi_node_idx - 1U];
    fi_n_injected++;

    fi_log.done         = 1;
    fi_log.reg          = (uint8_t)reg;
    fi_log.bit          = (uint8_t)bit;
    fi_log.node         = (uint8_t)fi_node_ids[fi_node_idx - 1U];
    fi_log.abort_cause  = FI_ABORT_NONE;

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

        fi_log.abort_cause = FI_ABORT_CRASH;

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

/* TIM2 IRQ: save all registers, call the dispatcher, restore all registers
 * (one of them has the bit flip). */
__attribute__((naked)) void IRQ_TIM2_HANDLER(void) {
    __asm volatile (
        "TST    LR, #4          \n"   /* Interrupted thread on MSP or PSP?  */
        "ITE    EQ              \n"
        "MRSEQ  R0, MSP         \n"   /* R0 = frame HW                    */
        "MRSNE  R0, PSP         \n"
        "PUSH   {R4-R11}        \n"   /* R4..R11 copy */
        "MOV    R1, SP          \n"   /* R1 = address of R4..R11 copy */
    	"VPUSH  {S16-S31}       \n"   /* S16..S31 copy */
    	"MOV    R3, SP          \n"   /* R3 = address of S16..S31 copy */
        "MOV    R2, LR          \n"   /* R2 = EXC_RETURN                  */
        "PUSH   {R4, LR}        \n"   /* save EXC_RETURN (R4 is just for align SP to 8 byte)  */
        "BL     fi_dispatch     \n"	  /* fi_dispatch(frame, r4_r11, exc_ret, s16_31)*/
        "POP    {R4, LR}        \n"
    	"VPOP	{S16-S31}		\n"	  /* reload S16..S31 from copy */
        "POP    {R4-R11}        \n"   /* reload R4..R11 from copy */
        "BX     LR              \n"   /* the HW reload R0-R3, R12, S0-S15, FPSCR from frame */
    );
}



