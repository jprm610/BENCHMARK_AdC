/*
 * timing.h - Monotonic time helper for benchmark measurements.
 *
 * CLOCK_MONOTONIC is immune to wall-clock adjustments and offers
 * nanosecond resolution on modern Linux. Used everywhere we time a
 * kernel call.
 */

#ifndef TIMING_H
#define TIMING_H

#include <time.h>

static inline double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#endif /* TIMING_H */
