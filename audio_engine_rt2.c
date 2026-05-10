/*
 * audio_engine_rt2.c
 *
 * Realtime JACK audio engine for the live EQ mixer. It receives EQ, gain,
 * and pan updates from Python over a Unix socket, applies biquad filters in
 * the JACK callback, mixes the tracks to stereo, and serves meter readings
 * back to the GUI.
 *
 * The important design rule is that the JACK callback must stay realtime-safe:
 * it does no socket I/O, memory allocation, printing, or blocking waits. Slower
 * work happens on helper threads, then the callback picks up finished updates
 * through small atomic flags.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <jack/jack.h>
#include <stdatomic.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <errno.h>
#include <stdint.h>
#include <signal.h>
#include <sys/types.h>

/* Audio and EQ limits shared with the Python control message format.
 * If these change, the Python struct layouts must change too. */
#define MAX_SECTIONS 16
#define MAX_ORDER 8
#define MAX_JACK_FRAME_SIZE 1024
#define NUM_CHANNELS 4
#define DEFAULT_CROSSFADE_SAMPLES 256
#define MAX_TRACKS 4
#define NUM_BANDS 3

/* Local Unix sockets used for Python <-> C control and metering.
 * /tmp is used so stale socket files are easy to clean on startup. */
#define CONTROL_SOCKET_PATH "/tmp/eq_control.sock"
#define METER_SOCKET_PATH   "/tmp/eq_meter.sock"

/* One direct-form biquad section plus its two-sample input/output history. */
typedef struct {
    float b0, b1, b2;
    float a1, a2;
    float x1, x2;
    float y1, y2;
} Biquad;

/* A channel EQ is a small cascade of biquad sections run in order. */
typedef struct {
    Biquad sections[MAX_SECTIONS];
    int num_sections;
} EQChain;

/* Filter type ids must match FILTER_MAP in audio_engine7_live.py. */
typedef enum {
    FILTER_PEAK,
    FILTER_LOWSHELF,
    FILTER_HIGHSHELF,
    FILTER_LOWPASS,
    FILTER_HIGHPASS
} FilterType;

/* Internal band description used when building an EQChain from GUI settings. */
typedef struct {
    int enabled;
    FilterType type;
    float freq;
    float q;
    float gain_db;
    float slope;
    int order;
} BandConfig;

/* -------------------------------------------------------------------------
 * Per-channel runtime state (owned exclusively by the RT callback thread).
 * The control thread never touches this directly; it writes to the staging
 * slot below and sets the atomic flag.
 * ------------------------------------------------------------------------- */
typedef struct {
    EQChain active;     /* currently processing chain, with live filter state  */
    EQChain fading_out; /* old chain kept alive during crossfade               */

    int  crossfade_total;
    int  crossfade_remaining;
    int  in_crossfade;  /* 1 while a fade is in progress                       */
} ChannelRT;

/* -------------------------------------------------------------------------
 * Staging slot: written by the control thread, consumed by RT thread.
 * Uses a 2-state atomic handshake so the control thread can always
 * overwrite a pending-but-unconsumed update without stalling.
 *
 *   IDLE  -> control thread fills the slot and sets READY
 *   READY -> RT thread sees it, copies it, sets IDLE
 *
 * If the control thread writes again before RT picks it up, it just
 * overwrites the slot (last-write-wins, acceptable for EQ param changes).
 * ------------------------------------------------------------------------- */
#define PENDING_IDLE  0
#define PENDING_READY 1

typedef struct {
    EQChain  chain;
    float    gain;
    _Atomic int state; /* PENDING_IDLE / PENDING_READY                        */
} PendingChannel;

/* MeterState stores both instant values and short holds so the GUI meter
 * looks stable instead of flickering every audio block. */
typedef struct {
    float rms_dbfs;
    float peak_dbfs;
    float peak_hold_dbfs;
    int   clipped;

    int clip_hold_count;
    int clip_hold_blocks;
    int peak_hold_count;
    int peak_hold_blocks;

    float peak_release;
    float clip_limit;
} MeterState;

typedef struct {
    int num_tracks;
    jack_client_t *client;
    jack_port_t *in[MAX_TRACKS];
    jack_port_t *outL;
    jack_port_t *outR;

    /* RT-thread-owned state. Never touched directly from other threads. */
    ChannelRT  ch[MAX_TRACKS];
    float      gain[MAX_TRACKS];
    _Atomic float pan[MAX_TRACKS];  /* Panning: -1.0 (left) to 1.0 (right) */

    /* Per-channel meters for GUI6 track displays. */
    MeterState track_meter[MAX_TRACKS];

    /* Staging slots: written by control thread, read by RT thread. */
    PendingChannel pending[MAX_TRACKS];

    /* Scratch buffers avoid heap allocation inside the RT callback. */
    float scratch_new[MAX_TRACKS][MAX_JACK_FRAME_SIZE];
    float scratch_old[MAX_TRACKS][MAX_JACK_FRAME_SIZE];

    MeterState meter;
} AppData;

/* Control payload from Python. Keep this field order in sync with CONTROL_STRUCT. */
typedef struct {
    int   enabled;
    int   type;
    float freq;
    float q;
    float gain_db;
    float slope;
    int   order;
} BandMsg;

typedef struct {
    /* Fixed-size arrays keep the socket ABI simple even with fewer tracks. */
    float   gains[NUM_CHANNELS];
    float   pans[NUM_CHANNELS];  /* Panning values for each track */
    BandMsg bands[NUM_CHANNELS][NUM_BANDS];
} ControlMessage;

/* Meter payload sent back to Python. Keep this in sync with METER_STRUCT. */
typedef struct {
    float rms_dbfs;
    float peak_dbfs;
    float peak_hold_dbfs;
    int   clipped;
} MeterMessage;

static AppData    app;
static const float fs = 48000.0f;

/* =========================================================================
 * Utility
 * ========================================================================= */

static float lin_to_dbfs(float x) {
    /* Clamp silence so log10 never sees zero. */
    if (x < 1e-9f) x = 1e-9f;
    return 20.0f * log10f(x);
}

static void print_jack_open_status(jack_status_t status) {
    /* Decode JACK bit flags into messages that are useful on the Pi. */
    if (status == 0) return;

    fprintf(stderr, "jack_client_open status: 0x%x\n", (unsigned int)status);
    if (status & JackFailure) {
        fprintf(stderr, "  - Overall operation failed.\n");
    }
    if (status & JackInvalidOption) {
        fprintf(stderr, "  - Invalid or unsupported JACK option.\n");
    }
    if (status & JackNameNotUnique) {
        fprintf(stderr, "  - Client name was not unique (JACK renamed it).\n");
    }
    if (status & JackServerStarted) {
        fprintf(stderr, "  - JACK server was started for this client.\n");
    }
    if (status & JackServerFailed) {
        fprintf(stderr, "  - Unable to connect to or start JACK server.\n");
    }
    if (status & JackServerError) {
        fprintf(stderr, "  - Communication error with JACK server.\n");
    }
    if (status & JackNoSuchClient) {
        fprintf(stderr, "  - Requested client does not exist.\n");
    }
    if (status & JackLoadFailure) {
        fprintf(stderr, "  - Unable to load internal JACK client.\n");
    }
    if (status & JackInitFailure) {
        fprintf(stderr, "  - JACK library initialization failed.\n");
    }
    if (status & JackShmFailure) {
        fprintf(stderr, "  - Shared memory access failed.\n");
    }
    if (status & JackVersionError) {
        fprintf(stderr, "  - JACK protocol version mismatch.\n");
    }
    if (status & JackBackendError) {
        fprintf(stderr, "  - Audio backend error in JACK server.\n");
    }
    if (status & JackClientZombie) {
        fprintf(stderr, "  - JACK reports this client as a zombie.\n");
    }
}

/* =========================================================================
 * Metering
 * ========================================================================= */

static void meter_init(MeterState *m, float sample_rate, int blocksize) {
    /* Hold counters are expressed in callback blocks, not seconds. */
    m->rms_dbfs      = -120.0f;
    m->peak_dbfs     = -120.0f;
    m->peak_hold_dbfs = -120.0f;
    m->clipped       = 0;

    m->clip_hold_blocks = (int)(0.25f * sample_rate / blocksize);
    if (m->clip_hold_blocks < 1) m->clip_hold_blocks = 1;
    m->clip_hold_count = 0;

    m->peak_hold_blocks = (int)(0.45f * sample_rate / blocksize);
    if (m->peak_hold_blocks < 1) m->peak_hold_blocks = 1;
    m->peak_hold_count = 0;

    m->peak_release = 0.35f;
    m->clip_limit   = 0.98f;
}

static void update_mono_meter(MeterState *m, const float *samples, float gain, int n) {
    if (n <= 0) return;

    /* RMS uses the gain-adjusted signal; peak is smoothed for the GUI. */
    double sumsq = 0.0;
    float peak = 0.0f;

    for (int i = 0; i < n; i++) {
        float v = gain * samples[i];
        float a = fabsf(v);
        if (a > peak) peak = a;
        sumsq += (double)v * v;
    }

    float rms = sqrtf((float)(sumsq / (double)n));
    float inst_peak_db = lin_to_dbfs(peak);

    /* Fast attack, slower release gives a readable meter without hiding peaks. */
    float prev_peak_lin  = powf(10.0f, m->peak_dbfs / 20.0f);
    float smooth_peak_lin = (peak >= prev_peak_lin)
        ? peak
        : prev_peak_lin + m->peak_release * (peak - prev_peak_lin);

    m->rms_dbfs  = lin_to_dbfs(rms);
    m->peak_dbfs = lin_to_dbfs(smooth_peak_lin);

    if (inst_peak_db >= m->peak_hold_dbfs) {
        /* Hold the highest recent peak briefly, then let it decay. */
        m->peak_hold_dbfs  = inst_peak_db;
        m->peak_hold_count = m->peak_hold_blocks;
    } else if (m->peak_hold_count > 0) {
        m->peak_hold_count--;
    } else {
        m->peak_hold_dbfs -= 1.5f;
        if (m->peak_hold_dbfs < m->peak_dbfs) m->peak_hold_dbfs = m->peak_dbfs;
    }

    if (peak >= m->clip_limit) {
        /* Keep the clip light on long enough for the user to notice it. */
        m->clip_hold_count = m->clip_hold_blocks;
    } else if (m->clip_hold_count > 0) {
        m->clip_hold_count--;
    }

    m->clipped = (m->clip_hold_count > 0);
}

static int send_all(int fd, const void *buf, size_t len) {
    /* send() may write only part of the packet, so loop until complete. */
    const uint8_t *ptr = (const uint8_t *)buf;
    size_t total = 0;

    while (total < len) {
        long sent = (long)send(fd, ptr + total, len - total, 0);
        if (sent <= 0) return -1;
        total += (size_t)sent;
    }

    return 0;
}

static void update_output_meter(MeterState *m, const float *outL, const float *outR, int n) {
    /* Master meter combines left and right channels into one display value. */
    double sumsq = 0.0;
    float  peak  = 0.0f;

    for (int i = 0; i < n; i++) {
        float al = fabsf(outL[i]);
        float ar = fabsf(outR[i]);
        if (al > peak) peak = al;
        if (ar > peak) peak = ar;
        sumsq += (double)outL[i] * outL[i];
        sumsq += (double)outR[i] * outR[i];
    }

    float rms          = sqrtf((float)(sumsq / (2.0 * n)));
    float inst_peak_db = lin_to_dbfs(peak);

    /* Same smoothing/hold behavior as the per-track meters. */
    float prev_peak_lin  = powf(10.0f, m->peak_dbfs / 20.0f);
    float smooth_peak_lin = (peak >= prev_peak_lin)
        ? peak
        : prev_peak_lin + m->peak_release * (peak - prev_peak_lin);

    m->rms_dbfs  = lin_to_dbfs(rms);
    m->peak_dbfs = lin_to_dbfs(smooth_peak_lin);

    if (inst_peak_db >= m->peak_hold_dbfs) {
        m->peak_hold_dbfs  = inst_peak_db;
        m->peak_hold_count = m->peak_hold_blocks;
    } else if (m->peak_hold_count > 0) {
        m->peak_hold_count--;
    } else {
        m->peak_hold_dbfs -= 1.5f;
        if (m->peak_hold_dbfs < m->peak_dbfs) m->peak_hold_dbfs = m->peak_dbfs;
    }

    if (peak >= m->clip_limit) {
        m->clip_hold_count = m->clip_hold_blocks;
    } else if (m->clip_hold_count > 0) {
        m->clip_hold_count--;
    }

    m->clipped = (m->clip_hold_count > 0);
}

static void get_meter_snapshot(const MeterState *src, MeterMessage *m) {
    /* Copy only plain values so the socket thread never shares pointers. */
    m->rms_dbfs      = src->rms_dbfs;
    m->peak_dbfs     = src->peak_dbfs;
    m->peak_hold_dbfs = src->peak_hold_dbfs;
    m->clipped       = src->clipped;
}

/* =========================================================================
 * Biquad DSP
 * ========================================================================= */

static void process_block(EQChain *chain, const float *input, float *output, int n) {
    /* Run each sample through every biquad while preserving section history. */
    for (int i = 0; i < n; i++) {
        float x = input[i];
        for (int j = 0; j < chain->num_sections; j++) {
            Biquad *bq = &chain->sections[j];
            float y = bq->b0 * x
                    + bq->b1 * bq->x1 + bq->b2 * bq->x2
                    - bq->a1 * bq->y1 - bq->a2 * bq->y2;
            bq->x2 = bq->x1; bq->x1 = x;
            bq->y2 = bq->y1; bq->y1 = y;
            x = y;
        }
        output[i] = x;
    }
}

static Biquad make_lowpass(float sample_rate, float f0, float q) {
    /* RBJ cookbook low-pass coefficients. */
    Biquad f;
    float w0     = 2.0f * M_PI * f0 / sample_rate;
    float alpha  = sinf(w0) / (2.0f * q);
    float cos_w0 = cosf(w0);
    float a0     = 1.0f + alpha;
    f.b0 = (1.0f - cos_w0) * 0.5f / a0;
    f.b1 = (1.0f - cos_w0)        / a0;
    f.b2 = (1.0f - cos_w0) * 0.5f / a0;
    f.a1 = (-2.0f * cos_w0)       / a0;
    f.a2 = (1.0f - alpha)         / a0;
    f.x1 = f.x2 = f.y1 = f.y2 = 0.0f;
    return f;
}

static Biquad make_highpass(float sample_rate, float f0, float q) {
    /* RBJ cookbook high-pass coefficients. */
    Biquad f;
    float w0     = 2.0f * M_PI * f0 / sample_rate;
    float alpha  = sinf(w0) / (2.0f * q);
    float cos_w0 = cosf(w0);
    float a0     = 1.0f + alpha;
    f.b0 =  (1.0f + cos_w0) * 0.5f / a0;
    f.b1 = -(1.0f + cos_w0)        / a0;
    f.b2 =  (1.0f + cos_w0) * 0.5f / a0;
    f.a1 = (-2.0f * cos_w0)        / a0;
    f.a2 = (1.0f - alpha)          / a0;
    f.x1 = f.x2 = f.y1 = f.y2 = 0.0f;
    return f;
}

static Biquad make_peak(float sample_rate, float f0, float q, float gain_db) {
    /* Peaking EQ uses A = sqrt(linear gain). */
    Biquad f;
    float A      = powf(10.0f, gain_db / 40.0f);
    float w0     = 2.0f * M_PI * f0 / sample_rate;
    float alpha  = sinf(w0) / (2.0f * q);
    float cos_w0 = cosf(w0);
    float a0     = 1.0f + alpha / A;
    f.b0 = (1.0f + alpha * A) / a0;
    f.b1 = (-2.0f * cos_w0)   / a0;
    f.b2 = (1.0f - alpha * A) / a0;
    f.a1 = (-2.0f * cos_w0)   / a0;
    f.a2 = (1.0f - alpha / A) / a0;
    f.x1 = f.x2 = f.y1 = f.y2 = 0.0f;
    return f;
}

static Biquad make_lowshelf(float sample_rate, float f0, float gain_db, float slope) {
    /* Low shelf coefficient formula from the RBJ audio EQ cookbook. */
    Biquad f;
    float A        = powf(10.0f, gain_db / 40.0f);
    float w0       = 2.0f * M_PI * f0 / sample_rate;
    float cos_w0   = cosf(w0);
    float alpha    = sinf(w0) * 0.5f * sqrtf((A + 1.0f / A) * (1.0f / slope - 1.0f) + 2.0f);
    float two_sqA  = 2.0f * sqrtf(A) * alpha;
    float a0       = (A + 1.0f) + (A - 1.0f) * cos_w0 + two_sqA;
    f.b0 =  A * ((A + 1.0f) - (A - 1.0f) * cos_w0 + two_sqA) / a0;
    f.b1 =  2.0f * A * ((A - 1.0f) - (A + 1.0f) * cos_w0)    / a0;
    f.b2 =  A * ((A + 1.0f) - (A - 1.0f) * cos_w0 - two_sqA) / a0;
    f.a1 = -2.0f * ((A - 1.0f) + (A + 1.0f) * cos_w0)        / a0;
    f.a2 = ((A + 1.0f) + (A - 1.0f) * cos_w0 - two_sqA)      / a0;
    f.x1 = f.x2 = f.y1 = f.y2 = 0.0f;
    return f;
}

static Biquad make_highshelf(float sample_rate, float f0, float gain_db, float slope) {
    /* High shelf coefficient formula from the RBJ audio EQ cookbook. */
    Biquad f;
    float A        = powf(10.0f, gain_db / 40.0f);
    float w0       = 2.0f * M_PI * f0 / sample_rate;
    float cos_w0   = cosf(w0);
    float alpha    = sinf(w0) * 0.5f * sqrtf((A + 1.0f / A) * (1.0f / slope - 1.0f) + 2.0f);
    float two_sqA  = 2.0f * sqrtf(A) * alpha;
    float a0       = (A + 1.0f) - (A - 1.0f) * cos_w0 + two_sqA;
    f.b0 =  A * ((A + 1.0f) + (A - 1.0f) * cos_w0 + two_sqA) / a0;
    f.b1 = -2.0f * A * ((A - 1.0f) + (A + 1.0f) * cos_w0)    / a0;
    f.b2 =  A * ((A + 1.0f) + (A - 1.0f) * cos_w0 - two_sqA) / a0;
    f.a1 =  2.0f * ((A - 1.0f) - (A + 1.0f) * cos_w0)        / a0;
    f.a2 = ((A + 1.0f) - (A - 1.0f) * cos_w0 - two_sqA)      / a0;
    f.x1 = f.x2 = f.y1 = f.y2 = 0.0f;
    return f;
}

static int setup_butterworth(int order, float *qs, float q_override) {
    /* Even filter orders are built as cascaded second-order sections. */
    if (order < 2 || order > MAX_ORDER) order = 2;
    if (order % 2 != 0) order += 1;
    int num_sections = order / 2;
    if (num_sections > MAX_SECTIONS) num_sections = MAX_SECTIONS;
    for (int k = 1; k <= num_sections; k++) {
        /* q_override lets the GUI force resonance; otherwise use Butterworth Qs. */
        qs[k - 1] = (q_override > 0.0f)
            ? q_override
            : 1.0f / (2.0f * sinf(((2.0f * k) - 1.0f) * M_PI / (2.0f * order)));
    }
    return num_sections;
}

static int build_eq_chain(EQChain *chain, const BandConfig *bands, float sample_rate, int num_bands) {
    /* Rebuild a full channel chain from the latest GUI band settings. */
    chain->num_sections = 0;
    for (int i = 0; i < num_bands && chain->num_sections < MAX_SECTIONS; i++) {
        const BandConfig *band = &bands[i];
        if (!band->enabled) continue;
        switch (band->type) {
            case FILTER_PEAK:
                /* Bell filter: one biquad centered at band->freq. */
                chain->sections[chain->num_sections++] =
                    make_peak(sample_rate, band->freq, band->q, band->gain_db);
                break;
            case FILTER_LOWSHELF:
                /* Shelves boost/cut everything below or above the corner. */
                chain->sections[chain->num_sections++] =
                    make_lowshelf(sample_rate, band->freq, band->gain_db, band->slope);
                break;
            case FILTER_HIGHSHELF:
                chain->sections[chain->num_sections++] =
                    make_highshelf(sample_rate, band->freq, band->gain_db, band->slope);
                break;
            case FILTER_LOWPASS: {
                /* Higher-order filters are several 2-pole sections in series. */
                float qs[MAX_SECTIONS];
                int ns = setup_butterworth(band->order, qs, band->q);
                for (int s = 0; s < ns && chain->num_sections < MAX_SECTIONS; s++)
                    chain->sections[chain->num_sections++] =
                        make_lowpass(sample_rate, band->freq, qs[s]);
                break;
            }
            case FILTER_HIGHPASS: {
                /* High-pass uses the same order/Q cascade as low-pass. */
                float qs[MAX_SECTIONS];
                int ns = setup_butterworth(band->order, qs, band->q);
                for (int s = 0; s < ns && chain->num_sections < MAX_SECTIONS; s++)
                    chain->sections[chain->num_sections++] =
                        make_highpass(sample_rate, band->freq, qs[s]);
                break;
            }
            default:
                break;
        }
    }
    return chain->num_sections > 0;
}

/* =========================================================================
 * Per-channel crossfade processing (RT thread only)
 *
 * Design:
 *  - Equal-power (sine/cosine) crossfade avoids the ~3 dB loudness dip
 *    that a linear blend produces at the midpoint.
 *  - When a new pending chain arrives we snapshot the old active chain
 *    into fading_out (preserving its live filter state) and install the
 *    new chain as active with zeroed filter state.  The crossfade then
 *    runs old->fading_out down and new->active up.
 *  - Because the new chain starts from a cold (zeroed) state there will
 *    be a brief settling transient, but this is masked by the fade and is
 *    the correct behaviour; the alternative (copying filter state from a
 *    different topology) produces worse artefacts.
 * ========================================================================= */

/* Called at the top of every RT callback to pull in any waiting update. */
static void channel_rt_check_pending(ChannelRT *ch, PendingChannel *pending, float *gain_out) {
    /* Acquire load pairs with the control thread's release store so the copied
       chain/gain values are visible before the RT thread installs them. */
    if (atomic_load_explicit(&pending->state, memory_order_acquire) != PENDING_READY)
        return;

    /* Consume the pending chain. Gain is not crossfaded; only EQ topology is. */
    *gain_out = pending->gain;

    if (ch->in_crossfade) {
        /* Previous crossfade was interrupted.  The current active chain
           becomes the new fading_out; blend position resets to full old. */
        ch->fading_out = ch->active;
    } else {
        ch->fading_out = ch->active;
    }

    /* Install the new chain with zeroed filter state (already zeroed by
       build_eq_chain via make_* helpers). */
    ch->active             = pending->chain;
    ch->crossfade_total    = DEFAULT_CROSSFADE_SAMPLES;
    ch->crossfade_remaining = DEFAULT_CROSSFADE_SAMPLES;
    ch->in_crossfade       = 1;

    atomic_store_explicit(&pending->state, PENDING_IDLE, memory_order_release);
}

static void process_channel(ChannelRT *ch, const float *input, float *output,
                             float *scratch_old, float *scratch_new, int n) {
    if (!ch->in_crossfade) {
        process_block(&ch->active, input, output, n);
        return;
    }

    int rem   = ch->crossfade_remaining;
    int total = ch->crossfade_total;

    /* Number of samples still in the fade window this block. */
    int fade_n = (n < rem) ? n : rem;

    /* Run both chains over the fade portion. */
    process_block(&ch->fading_out, input, scratch_old, fade_n);
    process_block(&ch->active,     input, scratch_new, fade_n);

    /* Equal-power blend over the fade window. */
    float pos_start = (float)(total - rem)         / (float)total; /* 0 -> 1 */
    float pos_end   = (float)(total - rem + fade_n) / (float)total;

    for (int i = 0; i < fade_n; i++) {
        float t     = pos_start + (pos_end - pos_start) * ((float)i / (float)fade_n);
        float angle = t * (M_PI * 0.5f);   /* 0 .. pi/2 */
        float g_old = cosf(angle);          /* 1 -> 0   */
        float g_new = sinf(angle);          /* 0 -> 1   */
        output[i]   = scratch_old[i] * g_old + scratch_new[i] * g_new;
    }

    ch->crossfade_remaining -= fade_n;

    /* Process the post-fade tail (new chain only). */
    if (fade_n < n) {
        process_block(&ch->active, input + fade_n, output + fade_n, n - fade_n);
    }

    if (ch->crossfade_remaining <= 0) {
        ch->in_crossfade        = 0;
        ch->crossfade_remaining = 0;
        /* fading_out is simply abandoned; its filter state is discarded. */
    }
}

/* =========================================================================
 * JACK callback (RT thread)
 * ========================================================================= */

static int callback(jack_nframes_t nframes, void *arg) {
    AppData *app = (AppData *)arg;

    /* Ignore unexpectedly large buffers instead of overflowing scratch arrays. */
    if (nframes > MAX_JACK_FRAME_SIZE)
        return 0;

    /* Pull any pending updates and process each channel.
     * The processed audio is stored in scratch_new[ch] for the mix stage. */
    for (int ch = 0; ch < app->num_tracks; ch++) {
        channel_rt_check_pending(&app->ch[ch], &app->pending[ch], &app->gain[ch]);

        jack_default_audio_sample_t *in =
            (jack_default_audio_sample_t *)jack_port_get_buffer(app->in[ch], nframes);

        process_channel(&app->ch[ch], in,
                        app->scratch_new[ch], /* reuse as output scratch */
                        app->scratch_old[ch],
                        app->scratch_new[ch], /* NOTE: process_channel writes output here */
                        (int)nframes);

        update_mono_meter(&app->track_meter[ch], app->scratch_new[ch], app->gain[ch], (int)nframes);
    }

    /* Precompute per-track effective gains including panning for the mix loop.
     * This keeps per-sample math small inside the nested mixing loop. */
    float eff_gain_L[app->num_tracks];
    float eff_gain_R[app->num_tracks];

    for (int ch = 0; ch < app->num_tracks; ch++) {
        float pan = atomic_load_explicit(&app->pan[ch], memory_order_acquire); /* -1.0 (left) to 1.0 (right) */
        if (pan < -1.0f) pan = -1.0f;
        if (pan > 1.0f)  pan = 1.0f;
        float gain = app->gain[ch]; /* Track fader gain */
        
        /* Equal-power panning keeps center position from sounding too quiet. */
        float panL = sqrtf((1.0f - pan) * 0.5f);
        float panR = sqrtf((1.0f + pan) * 0.5f);
        
        /* Combine track gain and pan into a single multiplier */
        eff_gain_L[ch] = gain * panL;
        eff_gain_R[ch] = gain * panR;
    }

    /* Fetch JACK output buffers only after all input channels are processed. */
    jack_default_audio_sample_t *outL =
        (jack_default_audio_sample_t *)jack_port_get_buffer(app->outL, nframes);
    jack_default_audio_sample_t *outR =
        (jack_default_audio_sample_t *)jack_port_get_buffer(app->outR, nframes);

    /* Mix loop with precomputed effective gains for better performance. */
    for (jack_nframes_t i = 0; i < nframes; i++) {
        float mixL = 0.0f;
        float mixR = 0.0f;

        for (int ch = 0; ch < app->num_tracks; ch++) {
            float sample = app->gain[ch] * app->scratch_new[ch][i];
            
            
            mixL += sample * eff_gain_L[ch];
            mixR += sample * eff_gain_R[ch];
        }
        
        outL[i] = mixL;
        outR[i] = mixR;
    }

    /* Master meter is based on the final stereo output sent to JACK. */
    update_output_meter(&app->meter, outL, outR, (int)nframes);
    return 0;
}

/* =========================================================================
 * Initialisation
 * ========================================================================= */

static void init_app(AppData *a, float sample_rate, int num_tracks) {
    /* Start with bypassed EQ, unity gain, and centered pan on every track. */
    memset(a, 0, sizeof(AppData));
    a->num_tracks = num_tracks;

    /* A zeroed bypass array means every band starts disabled. */
    BandConfig bypass[NUM_BANDS];
    memset(bypass, 0, sizeof(bypass));

    for (int i = 0; i < num_tracks; i++) {
        a->gain[i] = 1.0f;
        atomic_store_explicit(&a->pan[i], 0.0f, memory_order_release);  /* Center panning */
        build_eq_chain(&a->ch[i].active, bypass, sample_rate, NUM_BANDS);
        atomic_init(&a->pending[i].state, PENDING_IDLE);
    }
}

/* =========================================================================
 * Control message handling (control thread)
 * ========================================================================= */

static void apply_control_message(AppData *a, const ControlMessage *msg, float sample_rate) {
    /* This runs outside the JACK callback, so it can rebuild filter chains
     * without risking audio dropouts. The finished chains are then published
     * to the RT thread through PendingChannel slots. */
    for (int ch = 0; ch < a->num_tracks; ch++) {
        BandConfig bands[NUM_BANDS];
        for (int b = 0; b < NUM_BANDS; b++) {
            /* Copy from the socket ABI into the internal BandConfig format. */
            bands[b].enabled  = msg->bands[ch][b].enabled;
            bands[b].type     = (FilterType)msg->bands[ch][b].type;
            bands[b].freq     = msg->bands[ch][b].freq;
            bands[b].q        = msg->bands[ch][b].q;
            bands[b].gain_db  = msg->bands[ch][b].gain_db;
            bands[b].slope    = msg->bands[ch][b].slope;
            bands[b].order    = msg->bands[ch][b].order;
        }

        /* Build the new chain into the staging slot for this track. */
        PendingChannel *p = &a->pending[ch];

        /* The RT thread only installs this slot after it sees READY. */
        build_eq_chain(&p->chain, bands, sample_rate, NUM_BANDS);
        p->gain = msg->gains[ch];

        /* Update panning value for the RT callback thread. */
        float pan = msg->pans[ch];
        if (pan < -1.0f) pan = -1.0f;
        if (pan > 1.0f)  pan = 1.0f;
        atomic_store_explicit(&a->pan[ch], pan, memory_order_release);

        /* Publish to the RT thread (release store pairs with RT acquire load). */
        atomic_store_explicit(&p->state, PENDING_READY, memory_order_release);
    }
}

/* =========================================================================
 * Control socket thread
 * ========================================================================= */

static void *control_thread_fn(void *arg) {
    (void)arg;

    /* This thread blocks on control packets so the JACK callback never has to. */
    int server_fd = -1, client_fd = -1;
    struct sockaddr_un addr;

    /* Remove a leftover socket file from a previous crash or forced exit. */
    unlink(CONTROL_SOCKET_PATH);

    server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd < 0) return NULL;

    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, CONTROL_SOCKET_PATH, sizeof(addr.sun_path) - 1);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(server_fd); return NULL;
    }
    if (listen(server_fd, 1) < 0) {
        close(server_fd); return NULL;
    }

    for (;;) {
        /* Python may reconnect for each update; accept each client in turn. */
        client_fd = accept(server_fd, NULL, NULL);
        if (client_fd < 0) continue;

        for (;;) {
            ControlMessage msg;
            /* MSG_WAITALL waits for one full control packet before applying it. */
            ssize_t got = recv(client_fd, &msg, sizeof(msg), MSG_WAITALL);
            if (got <= 0) break;
            if ((size_t)got == sizeof(msg))
                apply_control_message(&app, &msg, fs);
        }

        close(client_fd);
        client_fd = -1;
    }

    close(server_fd);
    unlink(CONTROL_SOCKET_PATH);
    return NULL;
}

/* =========================================================================
 * Meter socket thread
 * ========================================================================= */

static void *meter_thread_fn(void *arg) {
    (void)arg;

    /* Meter requests are served on demand by the Python polling thread. */
    int server_fd = -1, client_fd = -1;
    struct sockaddr_un addr;

    /* Recreate the socket file each time this engine starts. */
    unlink(METER_SOCKET_PATH);

    server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd < 0) return NULL;

    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, METER_SOCKET_PATH, sizeof(addr.sun_path) - 1);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(server_fd); return NULL;
    }
    if (listen(server_fd, 1) < 0) {
        close(server_fd); return NULL;
    }

    for (;;) {
        /* Python polls this socket repeatedly while the GUI is open. */
        client_fd = accept(server_fd, NULL, NULL);
        if (client_fd < 0) continue;

        for (;;) {
            uint32_t request;
            /* Request value 1 means "send the latest meter packet." */
            ssize_t got = recv(client_fd, &request, sizeof(request), MSG_WAITALL);
            if (got <= 0) break;
            if ((size_t)got == sizeof(request) && request == 1) {
                struct {
                    MeterMessage master;
                    MeterMessage tracks[MAX_TRACKS];
                } packet;

                get_meter_snapshot(&app.meter, &packet.master);
                for (int ch = 0; ch < MAX_TRACKS; ch++) {
                    if (ch < app.num_tracks) {
                        get_meter_snapshot(&app.track_meter[ch], &packet.tracks[ch]);
                    } else {
                        /* Keep packet size fixed; inactive track meters read silent. */
                        packet.tracks[ch].rms_dbfs = -120.0f;
                        packet.tracks[ch].peak_dbfs = -120.0f;
                        packet.tracks[ch].peak_hold_dbfs = -120.0f;
                        packet.tracks[ch].clipped = 0;
                    }
                }

                if (send_all(client_fd, &packet, sizeof(packet)) < 0) break;
            }
        }

        close(client_fd);
        client_fd = -1;
    }

    close(server_fd);
    unlink(METER_SOCKET_PATH);
    return NULL;
}
static volatile sig_atomic_t keep_running = 1;
static void handle_sigint(int sig) {
    /* Signal handlers can only safely flip simple atomic-style flags. */
    (void)sig;
    keep_running = 0;
}

/* =========================================================================
 * Entry point
 * ========================================================================= */
int main(int argc, char **argv) {
    /* Optional argv[1] lets Python choose how many JACK inputs are active. */
    int num_tracks = 4;
    if (argc >= 2) {
        num_tracks = atoi(argv[1]);
        if (num_tracks < 1) num_tracks = 1;
        if (num_tracks > MAX_TRACKS) num_tracks = MAX_TRACKS;
    }

    /* Must run before jack_client_open because init_app zeroes AppData. */
    init_app(&app, fs, num_tracks);

    jack_status_t status = 0;
    app.client = jack_client_open("usb_eq_mixer", JackNullOption, &status);
    if (!app.client) {
        fprintf(stderr, "Failed to open JACK client 'usb_eq_mixer'.\n");
        print_jack_open_status(status);
        return 1;
    }

    if (status & JackNameNotUnique) {
        fprintf(stderr, "JACK renamed client to avoid a name collision.\n");
    }

    /* Meter hold times depend on JACK block size, so read it after open. */
    int jack_blocksize = (int)jack_get_buffer_size(app.client);
    if (jack_blocksize <= 0) jack_blocksize = DEFAULT_CROSSFADE_SAMPLES;
    meter_init(&app.meter, fs, jack_blocksize);
    for (int i = 0; i < app.num_tracks; i++) {
        meter_init(&app.track_meter[i], fs, jack_blocksize);
    }

    if (jack_set_process_callback(app.client, callback, &app) != 0) {
        fprintf(stderr, "Failed to set JACK process callback.\n");
        jack_client_close(app.client);
        return 1;
    }

    /* Register one mono input port per active track. */
    for (int i = 0; i < app.num_tracks; i++) {
        char name[16];
        snprintf(name, sizeof(name), "in%d", i + 1);
        app.in[i] = jack_port_register(app.client, name,
                                        JACK_DEFAULT_AUDIO_TYPE, JackPortIsInput, 0);
        if (!app.in[i]) {
            fprintf(stderr, "Failed to register JACK input port in%d.\n", i + 1);
            jack_client_close(app.client);
            return 1;
        }
    }

    /* The engine always outputs a stereo mix. */
    app.outL = jack_port_register(app.client, "outL", JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput, 0);
    app.outR = jack_port_register(app.client, "outR", JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput, 0);
    if (!app.outL || !app.outR) {
        fprintf(stderr, "Failed to register JACK output ports.\n");
        jack_client_close(app.client);
        return 1;
    }

    /* Socket threads must be running before Python waits for the sockets. */
    pthread_t control_thread, meter_thread;
    if (pthread_create(&control_thread, NULL, control_thread_fn, NULL) != 0) {
        fprintf(stderr, "Failed to start control socket thread.\n");
        jack_client_close(app.client);
        return 1;
    }
    if (pthread_create(&meter_thread, NULL, meter_thread_fn, NULL) != 0) {
        fprintf(stderr, "Failed to start meter socket thread.\n");
        jack_client_close(app.client);
        return 1;
    }

    /* Activating JACK starts calling callback() on the realtime audio thread. */
    if (jack_activate(app.client) != 0) {
        fprintf(stderr, "Failed to activate JACK client.\n");
        jack_client_close(app.client);
        return 1;
    }

    printf("Running.\n");
    printf("Control socket: %s\n", CONTROL_SOCKET_PATH);
    printf("Meter socket:   %s\n", METER_SOCKET_PATH);
    printf("Audio engine running. Kill python process to quit.\n");
    
    signal(SIGINT, handle_sigint);
    signal(SIGTERM, handle_sigint);
    /* Keep the process alive while JACK and socket threads do the real work. */
    while (keep_running) {
        usleep(500000); /* 0.5 seconds */
    }

    /* JACK close stops the callback; unlink removes sockets for the next run. */
    jack_client_close(app.client);
    unlink(CONTROL_SOCKET_PATH);
    unlink(METER_SOCKET_PATH);
    return 0;
}
