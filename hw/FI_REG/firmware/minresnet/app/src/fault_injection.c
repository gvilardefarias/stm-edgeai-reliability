#include "fault_injection.h"
#include "tim.h"
#include <platform.h>          /* TIM2, TIM_CR1_CEN, TIM_EGR_UG */
#include "core_cm7.h"
#include "stai.h"
#include "stai_events.h"
#include "app_stellar-studio-ai.h" 		/* stai_mnetwork_*, data_activations */
#include "network.h"                    /* STAI_NETWORK_* (sizes of activations, input, output) */
#include <setjmp.h>
#include <math.h>
#include <errno.h>

#include <stdio.h>

/* Possible states during the inference process */
#define FI_IDLE            0U        /* No Armed Inference state          */
#define FI_WAIT_FLIP       1U        /* TIM2 counts until bit flip instant */
#define FI_WATCHDOG        2U        /* flip done: TIM2 counts until timeout */

/* Assumptions on the generated network (see app/ReadMe.txt). */
_Static_assert(STAI_NETWORK_ACTIVATIONS_NUM == 1, "FI supports a single activations buffer");
_Static_assert(STAI_NETWORK_IN_NUM == 1, "FI protects a single input");
_Static_assert(FI_LAYERS_US > FI_LAYERS_MIN_US, "FI_LAYERS_US must be greater than FI_LAYERS_MIN_US");

#define FI_SRAM_BASE       0x24000000U

extern uint8_t __main_stack_base__[];     /* linker script symbols */
extern uint8_t __process_stack_end__[];

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
__attribute__((section(".dtcm"))) static volatile uint32_t fi_node_idx = 0U; 	/* position of the running layer, 1..n (0 = between layers) */
__attribute__((section(".dtcm"))) static volatile uint32_t fi_node_id = 0U;  	/* node id of the running layer */
__attribute__((section(".dtcm"))) static volatile uint32_t fi_starts = 0U;   	/* NODE_START seen in this inference */
__attribute__((section(".dtcm"))) static volatile uint32_t fi_in_read = 0U;  	/* 1 if the running layer reads the network input */
__attribute__((section(".dtcm"))) static volatile uint32_t fi_in_alive = 0U; 	/* 1 until a layer that does not read the input */
__attribute__((section(".dtcm"))) volatile uint32_t fi_n_runs = 0U;        		/* completed inferences */
static uint32_t fi_layers_min_ticks = 0U; /* FI_LAYERS_MIN_US in TIM2 ticks */
static uint32_t fi_layers_ticks = 0U;     /* FI_LAYERS_US in TIM2 ticks */
static uint32_t fi_timeout_ticks = 0U;    /* FI_TIMEOUT_US in TIM2 ticks */

/* Network input [fi_in_a, fi_in_b) and MPU region 7 that protects it (read-only). */
static uint32_t fi_in_a = 0U;
static uint32_t fi_in_b = 0U;
static uint32_t fi_in_rasr = 0U;          /* 0 = input not protected */
static uint32_t fi_in_ready = 0U;         /* input range computed */

/* Layer list of the network (ids in execution order), for crash recovery. */
static const stai_network_details *fi_details = NULL;

/* Crash/hang recovery: return point in the wrapper. */
static jmp_buf fi_jmp;
static volatile uint32_t fi_stops_done = 0U;  /* NODE_STOP already forwarded to the observer */

// Callbacks
static stai_event_cb fi_user_cb = NULL;    /* callback registered by the validation */
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

static uint32_t fi_mpu_size_field(uint32_t size) {
    return (31U - __CLZ(size)) - 1U;          /* log2(size) - 1 */
}

/* Region 'rnr' on [a, b): 8 subregions of g bytes (g >= 32), as fine as possible.
 * outward = 1: covers all of [a,b) (up to g bytes more at the edges)
 * outward = 0: nothing below 'a' (the first < g bytes may stay uncovered) */
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
            ARM_MPU_SetRegion(ARM_MPU_RBAR(rnr, base),
                ARM_MPU_RASR(0U, ap, 1U, 0U, 1U, 1U, srd, fi_mpu_size_field(size)));
            if (outward != 0U) { return (a - (base + lo * g)) + ((base + hi * g) - b); }
            return (base + lo * g) - a;
        }
    }
    ARM_MPU_ClrRegion(rnr);
    return 0xFFFFFFFFU;
}

/* [a, b) read/write with 2 regions, split at the most aligned boundary inside (unaligned buffers). */
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

/* One region STRICTLY inside [a, b) (never outside on either edge), with the largest
 * coverage: among all blocks of 8 x g bytes that overlap [a, b), the subregions fully
 * inside are enabled. Returns the region attributes (RASR, 0 if nothing fits) and
 * leaves the region programmed but DISABLED: fi_mpu_on() enables it. */
static uint32_t fi_mpu_cover_in(uint32_t rnr, uint32_t a, uint32_t b, uint32_t ap) {
    uint32_t g, best = 0U, best_base = 0U, best_rasr = 0U;

    for (g = 32U; (8U * g) <= 0x40000U; g <<= 1) {
        uint32_t size = 8U * g;
        uint32_t base;
        for (base = a & ~(size - 1U); base < b; base += size) {
            uint32_t lo = (a > base) ? ((a - base + g - 1U) / g) : 0U;
            uint32_t hi = ((b - base) / g < 8U) ? ((b - base) / g) : 8U;
            if ((hi > lo) && (((hi - lo) * g) > best)) {
                uint32_t srd = ~(((1U << hi) - 1U) & ~((1U << lo) - 1U)) & 0xFFU;
                best      = (hi - lo) * g;
                best_base = base;
                best_rasr = ARM_MPU_RASR(0U, ap, 1U, 0U, 1U, 1U, srd, fi_mpu_size_field(size));
            }
        }
    }
    ARM_MPU_SetRegion(ARM_MPU_RBAR(rnr, best_base), 0U);   /* base only, disabled */
    return best_rasr;
}

/* MPU regions active only inside the layers (the highest number wins).
 * Attributes = default SRAM: Normal, write-back write-allocate (TEX=1,C=1,B=1), not shareable.
 *   0: whole SRAM, read-only
 *   1-2: activations buffer (generated, data_activations[0]), read/write
 *   3-4: MSP + PSP stacks, read/write
 *   5: PSP frames above fi_sp_top (callers of the network), read-only - set by the wrapper
 *   6: errno (written by math functions)
 *   7: network input, read-only only in the layers that read it - set by the wrapper */
static void fi_mpu_init(void) {
    const uint32_t act = (uint32_t)data_activations[0];

    MPU->CTRL = 0U;
    __DSB(); __ISB();

    ARM_MPU_SetRegion(ARM_MPU_RBAR(0U, FI_SRAM_BASE),
        ARM_MPU_RASR(0U, ARM_MPU_AP_RO,   1U, 0U, 1U, 1U, 0x00U, ARM_MPU_REGION_SIZE_256KB));
    (void) fi_mpu_cover_rw2(1U, 2U, act, act + STAI_NETWORK_ACTIVATION_1_SIZE_BYTES);
    (void) fi_mpu_cover_rw2(3U, 4U, (uint32_t)__main_stack_base__, (uint32_t)__process_stack_end__);
    ARM_MPU_ClrRegion(5U);
    (void) fi_mpu_cover(6U, (uint32_t)&errno, (uint32_t)&errno + sizeof(errno), ARM_MPU_AP_FULL, 1U);
    ARM_MPU_ClrRegion(7U);
}
/* Turns the protection on: only the stack below fi_sp_top stays writable,
 * SysTick (4) and UART (5) masked, TIM2 (2) not.
 * MEMFAULTENA stays 0: the fault escalates to HardFault -> fi_fault_dispatch. */
static void fi_mpu_on(void) {
    MPU->RNR  = 7U;
    MPU->RASR = (fi_in_read != 0U) ? fi_in_rasr : 0U;

    __set_BASEPRI(4U << (8U - __NVIC_PRIO_BITS));
    MPU->CTRL = MPU_CTRL_PRIVDEFENA_Msk | MPU_CTRL_ENABLE_Msk;
    __DSB(); __ISB();
}
static void fi_mpu_off(void) {
    MPU->CTRL = 0U;
    __DSB(); __ISB();
    __set_BASEPRI(0U);
}

/* TIM2 as "layer stopwatch": counts only between NODE_START and NODE_STOP. */
static inline void fi_timer_pause(void)  { TIM2->CR1 &= ~TIM_CR1_CEN; }
static inline void fi_timer_resume(void) { TIM2->CR1 |=  TIM_CR1_CEN; }

/* Clears the counter and sets the update after 't' ticks, timer stopped. */
static void fi_timer_arm(uint32_t t) {
    fi_timer_pause();
    TIM2->ARR = t - 1U;
    TIM2->EGR = TIM_EGR_UG;      /* CNT = 0, loads ARR/PSC (with URS=1 no IRQ) */
    TIM2->SR  = 0U;
}

void FI_Init(void) {
	uint32_t ticks_per_us;

	tim_init(&DRV_TIM2);

	/* tick = 1 cycle of TIMCLK2 (150 MHz) */
	ticks_per_us        = tim_get_clock(&DRV_TIM2) / 1000000U;
	fi_layers_min_ticks = ticks_per_us * FI_LAYERS_MIN_US;
	fi_layers_ticks     = ticks_per_us * FI_LAYERS_US;
	fi_timeout_ticks    = ticks_per_us * FI_TIMEOUT_US;

    tim_set_prio(&DRV_TIM2, 2);
    tim_set_presc(&DRV_TIM2, TIM_DIVIDE_BY(1U));		/* PSC = 0: no division */
    tim_start(&DRV_TIM2);
    tim_enable_update_callback(&DRV_TIM2);

    fi_mpu_init();

    tim_enable(&DRV_TIM2);               /* sets ARPE/URS and enables the update IRQ */
    fi_timer_pause();                    /* counts only inside the layers */
}

/* 1 if one of the buffers passed to NODE_START (= inputs of the layer) is the network input.
 * The input is protected only in the first consecutive layers that read it: after the
 * first layer that does not read it, its area can be reused by other buffers. */
static uint32_t fi_reads_input(const stai_event_node_start_stop *ev) {
    stai_size i;
    if ((fi_in_alive != 0U) && (fi_in_rasr != 0U)) {
        for (i = 0; i < ev->buffers.size; i++) {
            if ((uint32_t)ev->buffers.data[i] == fi_in_a) {
                return 1U;
            }
        }
    }
    fi_in_alive = 0U;
    return 0U;
}

/* Callback on layer events: It updates the running layer and sends
 * the event to validation callback (observer), if present. */
static void fi_event_cb(void *cookie, const stai_event_type type, const void *payload) {
    const stai_event_node_start_stop *ev = (const stai_event_node_start_stop *)payload;

    if (type == STAI_EVENT_NODE_STOP) {
        fi_node_idx = 0U;                        /* first: prevents any flip in fi_dispatch() */
        __DSB();
        fi_timer_pause();                        /* stops TIM2 while the MPU is still on */
        __DSB();
        fi_mpu_off();                            /* now the observer can write */
    }
    if (fi_user_cb != NULL) {
        fi_user_cb(cookie, type, payload);
    }
    if (type == STAI_EVENT_NODE_STOP) {
        fi_stops_done++;                         /* layer closed for the observer too */
    }
    if (type == STAI_EVENT_NODE_START) {
        fi_starts++;                             /* position in execution order (ids can repeat) */
        fi_node_id  = (uint32_t)ev->node_id;
        fi_in_read  = fi_reads_input(ev);
        fi_node_idx = fi_starts;

        if (fi_phase != FI_IDLE) {
            fi_mpu_on();                         /* first MPU on and IRQ masked */
            fi_timer_resume();                   /* then TIM2 counts (also in watchdog) */
        }
    }
}

extern stai_return_code __real_stai_mnetwork_set_callback(stai_ptr network, const stai_event_cb cb, void *cb_cookie);
stai_return_code __wrap_stai_mnetwork_set_callback(stai_ptr network,
                                                   const stai_event_cb cb, void *cb_cookie) {
    if (fi_layers_ticks == 0U) {                 /* FI_Init not called: no interference */
        return __real_stai_mnetwork_set_callback(network, cb, cb_cookie);
    }
    fi_user_cb = cb;
    fi_user_cookie = cb_cookie;
    fi_details = stai_mnetwork_get_details(network);   /* NULL without HAVE_NETWORK_INFO */
    return __real_stai_mnetwork_set_callback(network, fi_event_cb, cb_cookie);
}

/* Interrupted inference: sends to the observer the events of the layers not completed,
 * so the host receives the same message sequence as a normal inference. */
static void fi_send_event(stai_event_type type, uint32_t id) {
    const stai_ptr bufs[1] = { data_activations[0] };   /* valid RAM, meaningless data */
    const stai_event_node_start_stop ev = {
        .node_id = (int32_t)id,
        .buffers = { .size = 1, .data = bufs }
    };
    if (fi_user_cb != NULL) {
        fi_user_cb(fi_user_cookie, type, &ev);
    }
}
static void fi_complete_events(uint32_t idx) {
    uint32_t k = fi_stops_done;                  /* layers in position 1..k already closed */

    if ((fi_user_cb == NULL) || (fi_details == NULL)) {
        return;                                  /* no observer (campaign) or no layer list */
    }
    if (idx != 0U) {
        fi_send_event(STAI_EVENT_NODE_STOP, (uint32_t)fi_details->nodes[idx - 1U].id);
        k = idx;
    }
    for (k = k + 1U; k <= fi_details->n_nodes; k++) {
        fi_send_event(STAI_EVENT_NODE_START, (uint32_t)fi_details->nodes[k - 1U].id);
        fi_send_event(STAI_EVENT_NODE_STOP,  (uint32_t)fi_details->nodes[k - 1U].id);
    }
}

/* Network input range and its MPU region, computed once (MPU off here). */
static void fi_input_init(stai_ptr network) {
    stai_ptr ins[STAI_NETWORK_IN_NUM];

    fi_in_ready = 1U;
    if (stai_mnetwork_get_inputs(network, ins) != STAI_SUCCESS) {
        return;                                  /* input not protected */
    }
    fi_in_a    = (uint32_t)ins[0];
    fi_in_b    = fi_in_a + STAI_NETWORK_IN_1_SIZE_BYTES;
    fi_in_rasr = fi_mpu_cover_in(7U, fi_in_a, fi_in_b, ARM_MPU_AP_RO);
}

extern stai_return_code __real_stai_mnetwork_run(stai_ptr network);
stai_return_code __wrap_stai_mnetwork_run(stai_ptr network) {
    stai_return_code ret;
    uint32_t t;

    if (fi_layers_ticks == 0U) {                 /* FI_Init not called: normal inference */
        return __real_stai_mnetwork_run(network);
    }
    if (fi_in_ready == 0U) {
        fi_input_init(network);
    }
    t = fi_layers_min_ticks + 1U + (xorshift32() % (fi_layers_ticks - fi_layers_min_ticks)); /* (fi_layers_min_ticks .. fi_layers_ticks] */

    fi_node_idx = 0U;
    fi_starts = 0U;
    fi_in_alive = 1U;
    fi_stops_done = 0U;
    fi_record.done        = 0U;
    fi_record.sample      = fi_n_runs;
    fi_record.delay_ticks = t;
    fi_record.abort_cause = FI_ABORT_NONE;

    fi_log.done_and_bit = 0U;
    fi_log.delay_ticks = (uint32_t) t;
    fi_log.abort_cause = FI_ABORT_NONE;

    fi_timer_arm(t);
    fi_phase = FI_WAIT_FLIP;

    if (setjmp(fi_jmp) == 0) {
        fi_in_inference = 1U;
        fi_sp_top = (__get_PSP() + 31U) & ~31U;  /* below here: only frames of the network */
        /* Region 5: frames of the callers (above fi_sp_top) read-only. MPU off here. */
        (void) fi_mpu_cover(5U, fi_sp_top, (uint32_t)__process_stack_end__, ARM_MPU_AP_RO, 0U);

        ret = __real_stai_mnetwork_run(network);
    } else {
        /* Only from fi_abort(): crash or hang after the flip. */
        stai_ptr outs[1];
        uint32_t idx = fi_node_idx;
        uint32_t i;

        fi_in_inference = 0U;
        fi_phase = FI_IDLE;
        fi_timer_pause();
        fi_mpu_off();
        fi_node_idx = 0U;

        fi_complete_events(idx);                 /* complete sequence for the observer */
        if (stai_mnetwork_get_outputs(network, outs) == STAI_SUCCESS) {
            for (i = 0U; i < STAI_NETWORK_OUT_1_SIZE; i++) {
                ((float *)outs[0])[i] = NAN;     /* sentinel on all classes: invalid inference */
            }
        }
        ret = STAI_SUCCESS;                      /* the validation goes on */
    }

    fi_phase = FI_IDLE;
    fi_node_idx = 0U;
    fi_in_inference = 0U;
    __DSB();
    fi_timer_pause();
    TIM2->SR = 0U;
    fi_mpu_off();
    fi_n_runs++;

    return ret;
}

/* Runs in thread mode at exception return (see fi_abort). */
__attribute__((noinline, noreturn)) static void fi_abort_to_wrapper(void) {
    longjmp(fi_jmp, 1);
}

/* Makes the interrupted code resume in fi_abort_to_wrapper instead of the
 * interrupted instruction. Returns 0 if not possible. */
static uint32_t fi_abort(fi_exc_frame_t *frame, uint32_t exc_ret) {
	fi_mpu_off();                                /* the wrapper restarts with SP above fi_sp_top */
	fi_timer_pause();
    fi_phase = FI_IDLE;
    if (((exc_ret & 0x8U) == 0U) || (fi_in_inference == 0U)) {
        return 0U;                               /* not in thread mode or outside the inference */
    }
    frame->pc   = (uint32_t)&fi_abort_to_wrapper & ~1U;
    frame->xpsr = (frame->xpsr & (1U << 9)) | (1U << 24);   /* only Thumb bit (+ alignment) */
    return 1U;
}

/* Dispatch call.
 * frame : frame HW (R0-R3, R12, LR, PC, xPSR, S0-S15, FPSCR) of interrupted context
 * r4_r11: copy of R4..R11
 * s16_s31: copy of S16-S31
 * exc_ret: EXC_RETURN value (LR at interrupt beginning) */
void fi_dispatch(fi_exc_frame_t *frame, uint32_t *r4_r11, uint32_t exc_ret, uint32_t *s16_s31) {
    __tim_serve_interrupt(&DRV_TIM2);            /* update ack */

    if (fi_in_inference == 0U) {
        return;                                  /* inference finished: no flip nor abort */
    }
    if (fi_phase == FI_WATCHDOG) {               /* FI_TIMEOUT_US of computation after the flip */
        fi_record.abort_cause = FI_ABORT_HANG;
        fi_record.fault_pc    = frame->pc;
        fi_n_hang++;

        fi_log.abort_cause = FI_ABORT_HANG;

        (void)fi_abort(frame, exc_ret);
        return;
    }
    if ((fi_phase != FI_WAIT_FLIP) || (fi_node_idx == 0U)) {
        return;                                  /* outside a layer: no flip */
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
    fi_record.node         = fi_node_id;
    fi_record.node_pos     = fi_node_idx;
    fi_n_injected++;

    fi_log.done_and_bit = 1 << 7 | (uint8_t)bit;
    fi_log.reg          = (uint8_t)reg;
    fi_log.node         = (uint8_t)fi_node_idx;  /* position, unique even when ids repeat */
    fi_log.abort_cause  = FI_ABORT_NONE;

    /* From here TIM2 is a watchdog: it counts the layer computation after the flip. */
    fi_phase = FI_WATCHDOG;
    fi_timer_arm(fi_timeout_ticks);
    fi_timer_resume();                           /* we are inside a layer */
}

/* Fault (BusFault/UsageFault/MemManage escalated to HardFault) during
 * an inference with flip: records the cause and interrupts the inference. */
void fi_fault_dispatch(fi_exc_frame_t *frame, uint32_t exc_ret) {
    uint32_t cfsr = SCB->CFSR;

    /* Errors during frame stacking/unstacking: frame not reliable. */
    const uint32_t stack_err = (1U << 3) | (1U << 4) | (1U << 5) |      /* MUNSTKERR, MSTKERR, MLSPERR */
                               (1U << 11) | (1U << 12) | (1U << 13);    /* UNSTKERR, STKERR, LSPERR */

    if ((fi_in_inference != 0U) && ((cfsr & stack_err) == 0U)) {
        fi_record.abort_cause = FI_ABORT_CRASH;
        fi_record.fault_pc    = frame->pc;

        fi_log.abort_cause = FI_ABORT_CRASH;

        SCB->CFSR = cfsr;                        /* clears the flags for the next fault */
        SCB->HFSR = SCB->HFSR;
        fi_n_crash++;
        if (fi_abort(frame, exc_ret) != 0U) {
            return;                              /* exception return -> fi_abort_to_wrapper */
        }
    }
    for (;;) {                                   /* unrecoverable fault */
    }
}

__attribute__((naked)) void HardFault_Handler(void) {
    __asm volatile (
        "TST    LR, #4          \n"   /* fault in code on MSP or PSP?    */
        "ITE    EQ              \n"
        "MRSEQ  R0, MSP         \n"   /* R0 = frame HW                    */
        "MRSNE  R0, PSP         \n"
        "MOV    R1, LR          \n"   /* R1 = EXC_RETURN                  */
        "B      fi_fault_dispatch \n" /* its return is the exception return */
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
