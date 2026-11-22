// Timed, software-only memory-traffic OOK sender for the local SDR experiment.
// Build: cc -O3 -std=c11 -pthread -o work/fast_sender work/fast_sender.c
#define _DARWIN_C_SOURCE 1
#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <mach/mach_time.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { CHUNK_LINES = 256, MAX_WORKERS = 12 };
typedef enum { IDLE_PARK, IDLE_SPIN, IDLE_CACHE } IdleMode;
typedef enum { ENCODING_NRZ, ENCODING_MANCHESTER } Encoding;

static atomic_bool active = false;
static atomic_bool finished = false;
static atomic_int ready_workers = 0;
static atomic_uint transition_index = 0;
static pthread_mutex_t gate_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t gate_cv = PTHREAD_COND_INITIALIZER;
static IdleMode idle_mode = IDLE_PARK;
static Encoding encoding = ENCODING_NRZ;
static bool busy_clock = false;
static bool worker_qos_initiated = false;
static bool clock_qos_interactive = false;
static uint32_t spin_margin_us = 500;
static mach_timebase_info_data_t timebase;

typedef struct {
    uint64_t *buffer;
    size_t words;
    uint64_t lines_touched;
    uint64_t checksum;
    uint64_t *cache_buffer;
    uint64_t *seen_on;
    uint64_t *seen_off;
    int qos_set_rc;
    int qos_get_rc;
    qos_class_t qos_actual_class;
    int qos_actual_relative;
    pthread_t thread;
} Worker;

static const char *qos_class_name(qos_class_t qos) {
    switch (qos) {
        case QOS_CLASS_USER_INTERACTIVE: return "interactive";
        case QOS_CLASS_USER_INITIATED: return "initiated";
        case QOS_CLASS_DEFAULT: return "default";
        case QOS_CLASS_UTILITY: return "utility";
        case QOS_CLASS_BACKGROUND: return "background";
        default: return "unspecified";
    }
}

static uint64_t ticks_for_ns(uint64_t ns) {
    return (uint64_t)(((__uint128_t)ns * timebase.denom + timebase.numer - 1) / timebase.numer);
}

static uint64_t symbol_offset_ns(uint32_t index, uint32_t rate) {
    // Divide after multiplication so fractional nanoseconds do not accumulate
    // into a clock drift at rates such as 5120 symbols/s.
    return ((uint64_t)index * UINT64_C(1000000000)) / rate;
}

static uint64_t monotonic_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

static void wait_until(uint64_t deadline) {
    uint64_t spin_margin = ticks_for_ns((uint64_t)spin_margin_us * 1000);
    uint64_t now = mach_absolute_time();
    // A long single sleep can return several milliseconds late on macOS.
    // Wake 10 ms early before the shorter final wait.
    uint64_t coarse_margin = ticks_for_ns(10000000);
    if (!busy_clock && deadline > now + coarse_margin) mach_wait_until(deadline - coarse_margin);
    now = mach_absolute_time();
    if (!busy_clock && deadline > now + spin_margin) mach_wait_until(deadline - spin_margin);
    while (mach_absolute_time() < deadline) {
#if defined(__aarch64__)
        __asm__ volatile("yield");
#endif
    }
}

static void *memory_worker(void *arg) {
    Worker *w = arg;
    if (worker_qos_initiated) {
        w->qos_set_rc = pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
    }
    w->qos_get_rc = pthread_get_qos_class_np(
        pthread_self(), &w->qos_actual_class, &w->qos_actual_relative);
    size_t offset = 0;
    uint64_t checksum = 0;
    uint64_t touched = 0;
    bool was_active = false;
    atomic_fetch_add_explicit(&ready_workers, 1, memory_order_release);
    while (!atomic_load_explicit(&finished, memory_order_acquire)) {
        bool is_active = atomic_load_explicit(&active, memory_order_acquire);
        if (is_active != was_active) {
            unsigned index = atomic_load_explicit(&transition_index, memory_order_acquire);
            if (is_active) w->seen_on[index] = monotonic_ns();
            else w->seen_off[index] = monotonic_ns();
            was_active = is_active;
        }
        if (!is_active) {
            if (idle_mode == IDLE_SPIN) {
#if defined(__aarch64__)
                __asm__ volatile("yield");
#endif
            } else if (idle_mode == IDLE_CACHE) {
                // Match the active arithmetic while accesses hit a 16 KiB local buffer.
                for (size_t i = 0; i < CHUNK_LINES; ++i) {
                    size_t index = i * 8;
                    uint64_t value = w->cache_buffer[index] + index + 1;
                    w->cache_buffer[index] = value;
                    checksum += value;
                }
            } else {
                pthread_mutex_lock(&gate_mutex);
                while (!atomic_load_explicit(&active, memory_order_acquire) &&
                       !atomic_load_explicit(&finished, memory_order_acquire)) {
                    pthread_cond_wait(&gate_cv, &gate_mutex);
                }
                pthread_mutex_unlock(&gate_mutex);
            }
            continue;
        }
        if (offset + CHUNK_LINES * 8 > w->words) offset = 0;
        for (size_t i = 0; i < CHUNK_LINES; ++i) {
            size_t index = offset + i * 8; // one 64-byte cache line
            uint64_t value = w->buffer[index] + index + 1;
            w->buffer[index] = value;
            checksum += value;
        }
        offset += CHUNK_LINES * 8;
        touched += CHUNK_LINES;
    }
    w->checksum = checksum;
    w->lines_touched = touched;
    return NULL;
}

static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static void usage(const char *name) {
    fprintf(stderr,
            "Usage: %s [--rate BPS] [--payload-bits N] [--workers N] "
            "[--buffer-mib N] [--lead-ms N] [--seed N] [--mode park|spin|cache] "
            "[--encoding nrz|manchester] [--payload-file PATH] "
            "[--clock hybrid|busy] [--spin-us N] "
            "[--worker-qos default|initiated] [--clock-qos default|interactive] "
            "[--log PATH]\n",
            name);
}

int main(int argc, char **argv) {
    uint32_t rate = 1000, payload_bits = 256, worker_count = 4;
    uint32_t buffer_mib = 128, lead_ms = 1000, seed = 0x93c5a771;
    const char *log_path = NULL;
    const char *payload_file_path = NULL;
    bool payload_bits_explicit = false;
    static const struct option options[] = {
        {"rate", required_argument, NULL, 'r'},
        {"payload-bits", required_argument, NULL, 'b'},
        {"workers", required_argument, NULL, 'w'},
        {"buffer-mib", required_argument, NULL, 'm'},
        {"lead-ms", required_argument, NULL, 'l'},
        {"seed", required_argument, NULL, 's'},
        {"mode", required_argument, NULL, 'o'},
        {"encoding", required_argument, NULL, 'e'},
        {"payload-file", required_argument, NULL, 'p'},
        {"clock", required_argument, NULL, 'c'},
        {"spin-us", required_argument, NULL, 'u'},
        {"worker-qos", required_argument, NULL, 1000},
        {"clock-qos", required_argument, NULL, 1001},
        {"log", required_argument, NULL, 'f'},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0},
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "r:b:w:m:l:s:o:e:p:c:u:f:h", options, NULL)) != -1) {
        switch (opt) {
            case 'r': rate = (uint32_t)strtoul(optarg, NULL, 0); break;
            case 'b': payload_bits = (uint32_t)strtoul(optarg, NULL, 0); payload_bits_explicit = true; break;
            case 'w': worker_count = (uint32_t)strtoul(optarg, NULL, 0); break;
            case 'm': buffer_mib = (uint32_t)strtoul(optarg, NULL, 0); break;
            case 'l': lead_ms = (uint32_t)strtoul(optarg, NULL, 0); break;
            case 's': seed = (uint32_t)strtoul(optarg, NULL, 0); break;
            case 'o':
                if (!strcmp(optarg, "spin")) idle_mode = IDLE_SPIN;
                else if (!strcmp(optarg, "park")) idle_mode = IDLE_PARK;
                else if (!strcmp(optarg, "cache")) idle_mode = IDLE_CACHE;
                else { usage(argv[0]); return 2; }
                break;
            case 'e':
                if (!strcmp(optarg, "nrz")) encoding = ENCODING_NRZ;
                else if (!strcmp(optarg, "manchester")) encoding = ENCODING_MANCHESTER;
                else { usage(argv[0]); return 2; }
                break;
            case 'p': payload_file_path = optarg; break;
            case 'c':
                if (!strcmp(optarg, "busy")) busy_clock = true;
                else if (!strcmp(optarg, "hybrid")) busy_clock = false;
                else { usage(argv[0]); return 2; }
                break;
            case 'u': spin_margin_us = (uint32_t)strtoul(optarg, NULL, 0); break;
            case 1000:
                if (!strcmp(optarg, "default")) worker_qos_initiated = false;
                else if (!strcmp(optarg, "initiated")) worker_qos_initiated = true;
                else { usage(argv[0]); return 2; }
                break;
            case 1001:
                if (!strcmp(optarg, "default")) clock_qos_interactive = false;
                else if (!strcmp(optarg, "interactive")) clock_qos_interactive = true;
                else { usage(argv[0]); return 2; }
                break;
            case 'f': log_path = optarg; break;
            case 'h': usage(argv[0]); return 0;
            default: usage(argv[0]); return 2;
        }
    }
    uint8_t *file_payload = NULL;
    if (payload_file_path) {
        FILE *file = fopen(payload_file_path, "rb");
        if (!file) { perror("fopen payload-file"); return 2; }
        file_payload = malloc(1000000);
        if (!file_payload) { perror("malloc payload-file"); fclose(file); return 1; }
        uint32_t count = 0;
        int ch;
        while ((ch = fgetc(file)) != EOF) {
            if (ch == '0' || ch == '1') {
                if (count == 1000000) {
                    fprintf(stderr, "payload-file exceeds 1000000 bits\n");
                    fclose(file);
                    free(file_payload);
                    return 2;
                }
                file_payload[count++] = ch == '1';
            } else if (ch != ' ' && ch != '\t' && ch != '\r' && ch != '\n') {
                fprintf(stderr, "payload-file contains non-bit character 0x%02x\n", ch);
                fclose(file);
                free(file_payload);
                return 2;
            }
        }
        if (ferror(file)) { perror("read payload-file"); fclose(file); free(file_payload); return 2; }
        fclose(file);
        if (!count || (payload_bits_explicit && payload_bits != count)) {
            fprintf(stderr, "payload-file has %u bits; expected %u\n", count,
                    payload_bits_explicit ? payload_bits : count);
            free(file_payload);
            return 2;
        }
        payload_bits = count;
    }
    if (!rate || rate > 50000 || !payload_bits || payload_bits > 1000000 ||
        !worker_count || worker_count > MAX_WORKERS || !buffer_mib || buffer_mib > 1024 ||
        !seed || lead_ms < 10 || spin_margin_us > 10000) {
        usage(argv[0]);
        return 2;
    }
    mach_timebase_info(&timebase);
    const uint32_t pilot_bits = 32, preamble_bits = 64, sync_bits = 16;
    const uint32_t logical_total = pilot_bits + preamble_bits + sync_bits + payload_bits;
    const uint32_t expansion = encoding == ENCODING_MANCHESTER ? 2 : 1;
    const uint32_t total = logical_total * expansion;
    uint8_t *logical_bits = calloc(logical_total, 1);
    uint8_t *bits = calloc(total, 1);
    uint64_t *actual = calloc(total, sizeof(*actual));
    if (!logical_bits || !bits || !actual) { perror("calloc"); return 1; }
    // First 16 zeros, then 16 ones, then alternating clock recovery, then sync.
    for (uint32_t i = 16; i < pilot_bits; ++i) logical_bits[i] = 1;
    for (uint32_t i = 0; i < preamble_bits; ++i) logical_bits[pilot_bits + i] = (i % 2) == 0;
    const uint16_t sync = 0xD391;
    for (uint32_t i = 0; i < sync_bits; ++i) {
        logical_bits[pilot_bits + preamble_bits + i] = (sync >> (15 - i)) & 1;
    }
    uint32_t prbs_state = seed;
    for (uint32_t i = pilot_bits + preamble_bits + sync_bits; i < logical_total; ++i) {
        logical_bits[i] = file_payload ? file_payload[i - pilot_bits - preamble_bits - sync_bits]
                                       : xorshift32(&prbs_state) & 1;
    }
    free(file_payload);
    for (uint32_t i = 0; i < logical_total; ++i) {
        if (encoding == ENCODING_MANCHESTER) {
            bits[2 * i] = logical_bits[i];
            bits[2 * i + 1] = !logical_bits[i];
        } else {
            bits[i] = logical_bits[i];
        }
    }
    uint32_t first_on_index = 0;
    while (first_on_index < total && bits[first_on_index] == 0) ++first_on_index;

    Worker *workers = calloc(worker_count, sizeof(*workers));
    if (!workers) { perror("calloc"); return 1; }
    const size_t bytes = (size_t)buffer_mib * 1024 * 1024;
    for (uint32_t i = 0; i < worker_count; ++i) {
        if (posix_memalign((void **)&workers[i].buffer, 64, bytes)) { perror("posix_memalign"); return 1; }
        if (posix_memalign((void **)&workers[i].cache_buffer, 64, 16384)) { perror("posix_memalign cache"); return 1; }
        workers[i].words = bytes / sizeof(uint64_t);
        workers[i].seen_on = calloc(total + 1, sizeof(uint64_t));
        workers[i].seen_off = calloc(total + 1, sizeof(uint64_t));
        if (!workers[i].seen_on || !workers[i].seen_off) { perror("calloc seen"); return 1; }
        memset(workers[i].buffer, 0, bytes);
        memset(workers[i].cache_buffer, 0, 16384);
        int error = pthread_create(&workers[i].thread, NULL, memory_worker, &workers[i]);
        if (error) { errno = error; perror("pthread_create"); return 1; }
    }
    while (atomic_load_explicit(&ready_workers, memory_order_acquire) != (int)worker_count) {
        struct timespec ts = {.tv_sec = 0, .tv_nsec = 1000000};
        nanosleep(&ts, NULL);
    }

    int clock_qos_set_rc = 0;
    if (clock_qos_interactive) {
        clock_qos_set_rc = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }
    qos_class_t clock_qos_actual_class = QOS_CLASS_UNSPECIFIED;
    int clock_qos_actual_relative = 0;
    int clock_qos_get_rc = pthread_get_qos_class_np(
        pthread_self(), &clock_qos_actual_class, &clock_qos_actual_relative);

    const uint64_t period_ns = UINT64_C(1000000000) / rate;
    const uint64_t now_ticks = mach_absolute_time();
    const uint64_t now_ns = monotonic_ns();
    const uint64_t start_ticks = now_ticks + ticks_for_ns((uint64_t)lead_ms * 1000000);
    const uint64_t start_ns = now_ns + (uint64_t)lead_ms * 1000000;
    const char *mode_name = idle_mode == IDLE_PARK ? "park" : idle_mode == IDLE_SPIN ? "spin" : "cache";
    const char *encoding_name = encoding == ENCODING_MANCHESTER ? "manchester" : "nrz";
    printf("READY start_monotonic_ns=%" PRIu64 " period_ns=%" PRIu64
           " physical_rate=%u logical_rate=%u physical_bits=%u logical_bits=%u "
           "encoding=%s mode=%s clock=%s\n",
           start_ns, period_ns, rate, rate / expansion, total, logical_total,
           encoding_name, mode_name,
           busy_clock ? "busy" : "hybrid");
    fflush(stdout);
    bool previous_bit = false;
    for (uint32_t i = 0; i < total; ++i) {
        wait_until(start_ticks + ticks_for_ns(symbol_offset_ns(i, rate)));
        actual[i] = monotonic_ns();
        if (bits[i] && !previous_bit) {
            atomic_store_explicit(&transition_index, i, memory_order_release);
            if (idle_mode == IDLE_PARK) {
                pthread_mutex_lock(&gate_mutex);
                atomic_store_explicit(&active, true, memory_order_release);
                pthread_cond_broadcast(&gate_cv);
                pthread_mutex_unlock(&gate_mutex);
            } else {
                atomic_store_explicit(&active, true, memory_order_release);
            }
        } else if (!bits[i] && previous_bit) {
            atomic_store_explicit(&transition_index, i, memory_order_release);
            atomic_store_explicit(&active, false, memory_order_release);
        }
        previous_bit = bits[i] != 0;
    }
    wait_until(start_ticks + ticks_for_ns(symbol_offset_ns(total, rate)));
    atomic_store_explicit(&transition_index, total, memory_order_release);
    atomic_store_explicit(&active, false, memory_order_release);
    atomic_store_explicit(&finished, true, memory_order_release);
    pthread_mutex_lock(&gate_mutex);
    pthread_cond_broadcast(&gate_cv);
    pthread_mutex_unlock(&gate_mutex);
    for (uint32_t i = 0; i < worker_count; ++i) pthread_join(workers[i].thread, NULL);

    FILE *log = log_path ? fopen(log_path, "w") : stdout;
    if (!log) { perror("fopen log"); return 1; }
    fprintf(log, "start_monotonic_ns=%" PRIu64 "\nperiod_ns=%" PRIu64
            "\nrate_bps=%u\nphysical_symbol_rate=%u\nlogical_bit_rate=%u\n"
            "physical_bit_count=%u\nlogical_bit_count=%u\nencoding=%s\nworkers=%u\n"
            "buffer_mib=%u\nmode=%s\nclock=%s\nspin_margin_us=%u\n"
            "worker_qos_requested=%s\nclock_qos_requested=%s\n"
            "clock_qos_set_rc=%d\nclock_qos_get_rc=%d\n"
            "clock_qos_actual=%s\nclock_qos_actual_relative=%d\n"
            "pilot_bits=%u\npreamble_bits=%u\nsync_bits=%u\npayload_bits=%u\n"
            "physical_preamble_symbols=%u\nphysical_payload_symbols=%u\n"
            "payload_source=%s\n"
            "seed=%u\n",
            start_ns, period_ns, rate, rate, rate / expansion,
            total, logical_total, encoding_name, worker_count, buffer_mib, mode_name,
            busy_clock ? "busy" : "hybrid", spin_margin_us,
            worker_qos_initiated ? "initiated" : "default",
            clock_qos_interactive ? "interactive" : "default",
            clock_qos_set_rc, clock_qos_get_rc,
            qos_class_name(clock_qos_actual_class), clock_qos_actual_relative,
            pilot_bits, preamble_bits, sync_bits, payload_bits,
            (pilot_bits + preamble_bits + sync_bits) * expansion,
            payload_bits * expansion, payload_file_path ? "file" : "prbs", seed);
    fputs("logical_bits=", log);
    for (uint32_t i = 0; i < logical_total; ++i) fputc(logical_bits[i] ? '1' : '0', log);
    fputc('\n', log);
    fputs("bits=", log);
    for (uint32_t i = 0; i < total; ++i) fputc(bits[i] ? '1' : '0', log);
    fputc('\n', log);
    fputs("transition_late_ns=", log);
    int64_t max_late = INT64_MIN, min_late = INT64_MAX;
    uint64_t late_abs_sum = 0;
    for (uint32_t i = 0; i < total; ++i) {
        int64_t late = (int64_t)actual[i] - (int64_t)(start_ns + symbol_offset_ns(i, rate));
        if (late > max_late) max_late = late;
        if (late < min_late) min_late = late;
        late_abs_sum += (uint64_t)(late < 0 ? -late : late);
        fprintf(log, "%s%" PRId64, i ? "," : "", late);
    }
    fputc('\n', log);
    fprintf(log, "timing_min_late_ns=%" PRId64 "\ntiming_max_late_ns=%" PRId64
            "\ntiming_mean_abs_late_ns=%" PRIu64 "\n",
            min_late, max_late, late_abs_sum / total);
    uint64_t touched = 0, checksum = 0;
    for (uint32_t i = 0; i < worker_count; ++i) {
        uint64_t observed = 0, missed = 0, delay_sum = 0, delay_max = 0;
        fprintf(log, "worker_%u_qos_set_rc=%d\nworker_%u_qos_get_rc=%d\n"
                "worker_%u_qos_actual=%s\nworker_%u_qos_actual_relative=%d\n",
                i, workers[i].qos_set_rc, i, workers[i].qos_get_rc,
                i, qos_class_name(workers[i].qos_actual_class),
                i, workers[i].qos_actual_relative);
        fprintf(log, "worker_%u_transition_delay_ns=", i);
        bool prior = false;
        for (uint32_t symbol = 0; symbol < total; ++symbol) {
            bool bit = bits[symbol] != 0;
            if (bit != prior) {
                uint64_t when = bit ? workers[i].seen_on[symbol] : workers[i].seen_off[symbol];
                if (when) {
                    uint64_t delay = when > actual[symbol] ? when - actual[symbol] : 0;
                    fprintf(log, "%s%" PRIu64, observed + missed ? "," : "", delay);
                    delay_sum += delay;
                    if (delay > delay_max) delay_max = delay;
                    observed++;
                } else {
                    fprintf(log, "%sNA", observed + missed ? "," : "");
                    missed++;
                }
            }
            prior = bit;
        }
        fputc('\n', log);
        fprintf(log, "worker_%u_observed=%" PRIu64 "\nworker_%u_missed=%" PRIu64
                "\nworker_%u_mean_delay_ns=%" PRIu64 "\nworker_%u_max_delay_ns=%" PRIu64
                "\nworker_%u_first_on_delay_ns=%" PRIu64 "\n",
                i, observed, i, missed, i, observed ? delay_sum / observed : 0,
                i, delay_max, i,
                workers[i].seen_on[first_on_index] > actual[first_on_index] ?
                workers[i].seen_on[first_on_index] - actual[first_on_index] : 0);
        touched += workers[i].lines_touched;
        checksum ^= workers[i].checksum;
        free(workers[i].buffer);
        free(workers[i].cache_buffer);
        free(workers[i].seen_on);
        free(workers[i].seen_off);
    }
    fprintf(log, "worker_bytes_touched=%" PRIu64 "\nchecksum=%" PRIu64 "\n",
            touched * 64, checksum);
    if (log != stdout) fclose(log);
    free(workers);
    free(logical_bits);
    free(bits);
    free(actual);
    return 0;
}
