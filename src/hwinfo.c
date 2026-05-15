/*
 * hwinfo.c - Runtime hardware fingerprint for the matmul benchmark.
 *
 * Reads CPU model, core / thread counts, L1d / L2 / L3 sizes, RAM totals
 * and SIMD feature flags directly from the kernel interfaces of Linux:
 *
 *   /proc/cpuinfo      -> CPU model name (vendor string fallback).
 *   /proc/meminfo      -> MemTotal, MemAvailable.
 *   /sys topology      -> physical core / package id pairs (count distinct).
 *   /sys cache         -> level, type, size, ways_of_associativity, line.
 *   sysconf()          -> primary source for L1d / L2 / L3 sizes; the /sys
 *                         path is the fallback when sysconf reports zero.
 *   __get_cpuid(0/1)   -> vendor string and family for the BMI2 microcode
 *                         warning (AMD pre-Zen3 implements PDEP/PEXT in
 *                         microcode with ~18-cycle latency; Sesion 03 must
 *                         avoid them in hot paths).
 *   __builtin_cpu_supports -> avx2 / fma / bmi2 feature detection.
 *
 * Two output modes:
 *
 *   ./bin/hwinfo            human-readable block, warnings interleaved.
 *   ./bin/hwinfo --csv      header + single CSV row to stdout; warnings on
 *                           stderr so the CSV does not get contaminated.
 *
 * Warnings emitted when applicable:
 *
 *   - L3 reported < 8 MiB suggests the value is per-CCX, not the package
 *     total (relevant on AMD Renoir / Ryzen 5 4600H with 2 CCX of 4 MiB).
 *   - BMI2 supported but vendor=AMD and family < 0x19 (pre-Zen3).
 */

/* Required to expose sysconf, _SC_NPROCESSORS_ONLN and the
 * _SC_LEVEL{1,2,3}_*CACHE_{SIZE,ASSOC,LINESIZE} extensions in glibc.
 * _GNU_SOURCE already implies _POSIX_C_SOURCE 2008. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <ctype.h>
#include <cpuid.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/* Small filesystem helpers                                            */
/* ------------------------------------------------------------------ */

/* Read a single long integer from the start of a file.
 * Returns 1 on success, 0 on any failure. */
static int read_long_file(const char *path, long *out)
{
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    int ok = (fscanf(f, "%ld", out) == 1);
    fclose(f);
    return ok;
}

/* Read the first line of a file into buf (newline stripped).
 * Returns 1 on success, 0 on any failure. */
static int read_str_file(const char *path, char *buf, size_t bufsize)
{
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    if (!fgets(buf, (int)bufsize, f)) { fclose(f); return 0; }
    fclose(f);
    size_t n = strlen(buf);
    if (n > 0 && buf[n - 1] == '\n') buf[n - 1] = '\0';
    return 1;
}

/* Parse cache size strings such as "32K", "512K", "4096K", "8M".
 * Returns the size in KiB. Returns 0 on parse failure. */
static long parse_size_kib(const char *s)
{
    if (!s || !*s) return 0;
    char *endp = NULL;
    long v = strtol(s, &endp, 10);
    if (endp == s) return 0;
    while (*endp == ' ' || *endp == '\t') endp++;
    switch (*endp) {
        case 'K': case 'k': return v;
        case 'M': case 'm': return v * 1024L;
        case 'G': case 'g': return v * 1024L * 1024L;
        case '\0':          return v / 1024L;  /* bytes */
        default:            return v;          /* assume KiB */
    }
}

/* Wrapper for sysconf that clamps -1 / errors to 0. */
static long sysconf_safe(int name)
{
    long v = sysconf(name);
    return v > 0 ? v : 0;
}

/* ------------------------------------------------------------------ */
/* /sys cache fallback                                                 */
/* ------------------------------------------------------------------ */

/* Find the cache index under /sys/devices/system/cpu/cpu0/cache/index*
 * that matches the requested level and (optionally) type string.
 * Writes the index into *index_out on success. */
static int find_cache_index(int level, const char *required_type, int *index_out)
{
    char path[256];
    char type_buf[32];
    long lvl;
    for (int i = 0; i < 16; i++) {
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu0/cache/index%d/level", i);
        if (!read_long_file(path, &lvl)) continue;
        if (lvl != level) continue;
        if (required_type) {
            snprintf(path, sizeof(path),
                     "/sys/devices/system/cpu/cpu0/cache/index%d/type", i);
            if (!read_str_file(path, type_buf, sizeof(type_buf))) continue;
            if (strcmp(type_buf, required_type) != 0) continue;
        }
        *index_out = i;
        return 1;
    }
    return 0;
}

/* Fill *size_kib, *assoc, *line by reading /sys cache attributes for the
 * given level and (optionally) type. Leaves zeros if no matching entry. */
static void read_cache_level_sys(int level, const char *type,
                                 long *size_kib, long *assoc, long *line)
{
    int idx = -1;
    if (!find_cache_index(level, type, &idx)) return;

    char path[256];
    char buf[64];

    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu0/cache/index%d/size", idx);
    if (read_str_file(path, buf, sizeof(buf))) {
        *size_kib = parse_size_kib(buf);
    }

    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu0/cache/index%d/ways_of_associativity",
             idx);
    read_long_file(path, assoc);

    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu0/cache/index%d/coherency_line_size",
             idx);
    read_long_file(path, line);
}

/* ------------------------------------------------------------------ */
/* /proc/cpuinfo and /proc/meminfo                                     */
/* ------------------------------------------------------------------ */

/* Walk /proc/cpuinfo capturing the first "model name" line and counting
 * "processor" entries (SMT threads). Also counts distinct
 * (physical id, core id) pairs as a fallback for physical-core count.
 * Any field can be missing on WSL2 etc; callers must tolerate zeros. */
static void read_proc_cpuinfo(char *model_out, size_t model_size,
                              long *num_threads_out, long *num_cores_proc_out)
{
    model_out[0] = '\0';
    *num_threads_out = 0;
    *num_cores_proc_out = 0;

    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) return;

    enum { MAX_PAIRS = 1024 };
    int phys_ids[MAX_PAIRS];
    int core_ids[MAX_PAIRS];
    int n_pairs = 0;
    int current_phys = -1;
    int current_core = -1;
    int model_captured = 0;

    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        if (!model_captured && strncmp(line, "model name", 10) == 0) {
            char *colon = strchr(line, ':');
            if (colon) {
                colon++;
                while (*colon == ' ' || *colon == '\t') colon++;
                size_t len = strlen(colon);
                if (len > 0 && colon[len - 1] == '\n') colon[len - 1] = '\0';
                strncpy(model_out, colon, model_size - 1);
                model_out[model_size - 1] = '\0';
                model_captured = 1;
            }
        } else if (strncmp(line, "processor", 9) == 0) {
            (*num_threads_out)++;
            current_phys = -1;
            current_core = -1;
        } else if (strncmp(line, "physical id", 11) == 0) {
            char *colon = strchr(line, ':');
            if (colon) current_phys = atoi(colon + 1);
        } else if (strncmp(line, "core id", 7) == 0) {
            char *colon = strchr(line, ':');
            if (colon) current_core = atoi(colon + 1);
            if (current_phys >= 0 && current_core >= 0) {
                int found = 0;
                for (int i = 0; i < n_pairs; i++) {
                    if (phys_ids[i] == current_phys
                        && core_ids[i] == current_core) {
                        found = 1;
                        break;
                    }
                }
                if (!found && n_pairs < MAX_PAIRS) {
                    phys_ids[n_pairs] = current_phys;
                    core_ids[n_pairs] = current_core;
                    n_pairs++;
                }
                current_phys = -1;
                current_core = -1;
            }
        }
    }
    fclose(f);
    *num_cores_proc_out = n_pairs;
}

/* Count distinct physical cores via /sys topology. Returns 0 if /sys is
 * not populated (e.g. some WSL2 builds). */
static long count_physical_cores_sys(void)
{
    DIR *d = opendir("/sys/devices/system/cpu");
    if (!d) return 0;

    enum { MAX_PAIRS = 1024 };
    int pkg_ids[MAX_PAIRS];
    int core_ids[MAX_PAIRS];
    int n_pairs = 0;

    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (strncmp(e->d_name, "cpu", 3) != 0) continue;
        if (!isdigit((unsigned char)e->d_name[3])) continue;

        /* d_name can be up to NAME_MAX (255) bytes per POSIX, so size the
         * buffer to comfortably hold the longest possible path. */
        char path[512];
        long pkg = -1, core = -1;
        int ok_pkg, ok_core;
        int n;

        n = snprintf(path, sizeof(path),
                     "/sys/devices/system/cpu/%s/topology/physical_package_id",
                     e->d_name);
        if (n < 0 || (size_t)n >= sizeof(path)) continue;
        ok_pkg = read_long_file(path, &pkg);

        n = snprintf(path, sizeof(path),
                     "/sys/devices/system/cpu/%s/topology/core_id", e->d_name);
        if (n < 0 || (size_t)n >= sizeof(path)) continue;
        ok_core = read_long_file(path, &core);

        if (!ok_pkg || !ok_core) continue;

        int found = 0;
        for (int i = 0; i < n_pairs; i++) {
            if (pkg_ids[i] == (int)pkg && core_ids[i] == (int)core) {
                found = 1;
                break;
            }
        }
        if (!found && n_pairs < MAX_PAIRS) {
            pkg_ids[n_pairs] = (int)pkg;
            core_ids[n_pairs] = (int)core;
            n_pairs++;
        }
    }
    closedir(d);
    return n_pairs;
}

static void read_meminfo(long *total_kib, long *avail_kib)
{
    *total_kib = 0;
    *avail_kib = 0;
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "MemTotal:", 9) == 0) {
            sscanf(line + 9, "%ld", total_kib);
        } else if (strncmp(line, "MemAvailable:", 13) == 0) {
            sscanf(line + 13, "%ld", avail_kib);
        }
    }
    fclose(f);
}

/* ------------------------------------------------------------------ */
/* CPU identity (vendor + family) via CPUID                            */
/* ------------------------------------------------------------------ */

static void detect_cpu_identity(char vendor_out[16], unsigned int *family_out)
{
    vendor_out[0] = '\0';
    *family_out = 0;
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;

    if (__get_cpuid(0, &eax, &ebx, &ecx, &edx)) {
        memcpy(vendor_out + 0, &ebx, 4);
        memcpy(vendor_out + 4, &edx, 4);
        memcpy(vendor_out + 8, &ecx, 4);
        vendor_out[12] = '\0';
    }

    eax = ebx = ecx = edx = 0;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        unsigned int base_family = (eax >> 8) & 0xFu;
        unsigned int ext_family  = (eax >> 20) & 0xFFu;
        *family_out = (base_family == 0xF) ? (base_family + ext_family)
                                           : base_family;
    }
}

/* ------------------------------------------------------------------ */
/* Aggregated record + main                                            */
/* ------------------------------------------------------------------ */

typedef struct {
    char cpu_model[256];
    char vendor[16];
    unsigned int family;

    long num_cores;
    long num_threads;

    long l1d_kib, l1d_assoc, l1d_line;
    long l2_kib,  l2_assoc,  l2_line;
    long l3_kib,  l3_assoc,  l3_line;

    long mem_total_kib;
    long mem_avail_kib;

    int avx2_supported;
    int fma_supported;
    int bmi2_supported;
} hw_info_t;

static void collect(hw_info_t *hw)
{
    long num_threads_proc = 0;
    long num_cores_proc = 0;
    read_proc_cpuinfo(hw->cpu_model, sizeof(hw->cpu_model),
                      &num_threads_proc, &num_cores_proc);
    if (hw->cpu_model[0] == '\0') {
        strncpy(hw->cpu_model, "unknown", sizeof(hw->cpu_model) - 1);
        hw->cpu_model[sizeof(hw->cpu_model) - 1] = '\0';
    }

    hw->num_threads = sysconf_safe(_SC_NPROCESSORS_ONLN);
    if (hw->num_threads <= 0) hw->num_threads = num_threads_proc;

    long cores_topo = count_physical_cores_sys();
    if (cores_topo > 0) {
        hw->num_cores = cores_topo;
    } else if (num_cores_proc > 0) {
        hw->num_cores = num_cores_proc;
    } else {
        hw->num_cores = hw->num_threads;
    }

    /* sysconf first; /sys cache fallback when sysconf returns 0. */
    hw->l1d_kib   = sysconf_safe(_SC_LEVEL1_DCACHE_SIZE) / 1024;
    hw->l1d_assoc = sysconf_safe(_SC_LEVEL1_DCACHE_ASSOC);
    hw->l1d_line  = sysconf_safe(_SC_LEVEL1_DCACHE_LINESIZE);
    if (hw->l1d_kib == 0)
        read_cache_level_sys(1, "Data",
                             &hw->l1d_kib, &hw->l1d_assoc, &hw->l1d_line);

    hw->l2_kib   = sysconf_safe(_SC_LEVEL2_CACHE_SIZE) / 1024;
    hw->l2_assoc = sysconf_safe(_SC_LEVEL2_CACHE_ASSOC);
    hw->l2_line  = sysconf_safe(_SC_LEVEL2_CACHE_LINESIZE);
    if (hw->l2_kib == 0)
        read_cache_level_sys(2, NULL,
                             &hw->l2_kib, &hw->l2_assoc, &hw->l2_line);

    hw->l3_kib   = sysconf_safe(_SC_LEVEL3_CACHE_SIZE) / 1024;
    hw->l3_assoc = sysconf_safe(_SC_LEVEL3_CACHE_ASSOC);
    hw->l3_line  = sysconf_safe(_SC_LEVEL3_CACHE_LINESIZE);
    if (hw->l3_kib == 0)
        read_cache_level_sys(3, NULL,
                             &hw->l3_kib, &hw->l3_assoc, &hw->l3_line);

    read_meminfo(&hw->mem_total_kib, &hw->mem_avail_kib);

    __builtin_cpu_init();
    hw->avx2_supported = __builtin_cpu_supports("avx2") ? 1 : 0;
    hw->fma_supported  = __builtin_cpu_supports("fma")  ? 1 : 0;
    hw->bmi2_supported = __builtin_cpu_supports("bmi2") ? 1 : 0;

    detect_cpu_identity(hw->vendor, &hw->family);
}

/* True when the running CPU has BMI2 reported but is AMD pre-Zen3, where
 * PDEP/PEXT are microcoded (~18 cycles). Family 0x19 is Zen3+. */
static int bmi2_is_microcoded(const hw_info_t *hw)
{
    return hw->bmi2_supported
        && strcmp(hw->vendor, "AuthenticAMD") == 0
        && hw->family > 0
        && hw->family < 0x19u;
}

/* True when reported L3 is < 8 MiB, which on Zen 2 Renoir means the
 * kernel is reporting per-CCX L3 (4 MiB) rather than a global figure. */
static int l3_likely_per_ccx(const hw_info_t *hw)
{
    return hw->l3_kib > 0 && hw->l3_kib < 8L * 1024L;
}

static void print_human(const hw_info_t *hw)
{
    int microcoded = bmi2_is_microcoded(hw);
    int l3_small   = l3_likely_per_ccx(hw);

    printf("=== Hardware fingerprint ===\n");
    printf("cpu_model       : %s\n", hw->cpu_model);
    printf("cpu_vendor      : %s  (family=0x%x)\n",
           hw->vendor[0] ? hw->vendor : "unknown", hw->family);
    printf("num_cores       : %ld\n", hw->num_cores);
    printf("num_threads     : %ld\n", hw->num_threads);
    printf("l1d_kib         : %ld   (assoc=%ld, line=%ld)\n",
           hw->l1d_kib, hw->l1d_assoc, hw->l1d_line);
    printf("l2_kib          : %ld   (assoc=%ld, line=%ld)\n",
           hw->l2_kib, hw->l2_assoc, hw->l2_line);
    printf("l3_kib          : %ld   (assoc=%ld, line=%ld)\n",
           hw->l3_kib, hw->l3_assoc, hw->l3_line);
    printf("mem_total_kib   : %ld  (~%.2f GiB)\n",
           hw->mem_total_kib, hw->mem_total_kib / 1024.0 / 1024.0);
    printf("mem_avail_kib   : %ld  (~%.2f GiB)\n",
           hw->mem_avail_kib, hw->mem_avail_kib / 1024.0 / 1024.0);
    printf("avx2_supported  : %s\n", hw->avx2_supported ? "yes" : "no");
    printf("fma_supported   : %s\n", hw->fma_supported  ? "yes" : "no");
    printf("bmi2_supported  : %s%s\n",
           hw->bmi2_supported ? "yes" : "no",
           microcoded
               ? " (WARNING: pre-Zen3 AMD implementa PDEP/PEXT en microcodigo, evitar)"
               : "");

    if (l3_small) {
        printf("[WARNING]: L3 reportada (%ld KiB ~ %.0f MiB) es menor que el "
               "datasheet (8 MiB). Probable medicion por CCX. Para "
               "single-thread, el L3 efectivo es %ld KiB.\n",
               hw->l3_kib, hw->l3_kib / 1024.0, hw->l3_kib);
    }
}

static void print_csv(const hw_info_t *hw)
{
    int microcoded = bmi2_is_microcoded(hw);
    int l3_small   = l3_likely_per_ccx(hw);

    printf("cpu_model,cpu_vendor,cpu_family,num_cores,num_threads,"
           "l1d_kib,l1d_assoc,l1d_line,"
           "l2_kib,l2_assoc,l2_line,"
           "l3_kib,l3_assoc,l3_line,"
           "mem_total_kib,mem_avail_kib,"
           "avx2_supported,fma_supported,bmi2_supported,"
           "bmi2_microcoded_warning,l3_small_warning\n");
    printf("\"%s\",%s,0x%x,%ld,%ld,"
           "%ld,%ld,%ld,"
           "%ld,%ld,%ld,"
           "%ld,%ld,%ld,"
           "%ld,%ld,"
           "%d,%d,%d,"
           "%d,%d\n",
           hw->cpu_model,
           hw->vendor[0] ? hw->vendor : "unknown",
           hw->family,
           hw->num_cores, hw->num_threads,
           hw->l1d_kib, hw->l1d_assoc, hw->l1d_line,
           hw->l2_kib, hw->l2_assoc, hw->l2_line,
           hw->l3_kib, hw->l3_assoc, hw->l3_line,
           hw->mem_total_kib, hw->mem_avail_kib,
           hw->avx2_supported, hw->fma_supported, hw->bmi2_supported,
           microcoded, l3_small);

    /* Warnings go to stderr so the CSV stream stays clean. */
    if (microcoded) {
        fprintf(stderr,
                "[WARNING]: BMI2 disponible pero %s family 0x%x: "
                "PDEP/PEXT microcoded, do not use in hot loops.\n",
                hw->vendor[0] ? hw->vendor : "unknown", hw->family);
    }
    if (l3_small) {
        fprintf(stderr,
                "[WARNING]: L3 reportada (%ld KiB) es menor que el "
                "datasheet (8 MiB). Probable medicion por CCX. Para "
                "single-thread, el L3 efectivo es %ld KiB.\n",
                hw->l3_kib, hw->l3_kib);
    }
}

static void print_usage(const char *progname)
{
    fprintf(stderr,
            "Usage: %s [--csv] [--help]\n"
            "  --csv   Emit a single CSV header + data row to stdout.\n"
            "          Warnings (if any) are written to stderr.\n"
            "  --help  Print this message and exit.\n",
            progname);
}

int main(int argc, char **argv)
{
    int csv = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--csv") == 0) {
            csv = 1;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    hw_info_t hw;
    memset(&hw, 0, sizeof(hw));
    collect(&hw);

    if (csv) print_csv(&hw);
    else     print_human(&hw);

    return 0;
}
