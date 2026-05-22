#!/usr/bin/env bash
# audit_no_pdep.sh
# Verifies that source code does not use PDEP/PEXT intrinsics and that the
# assembler emitted for the main Morton modules does not contain pdep/pext.
# Fails with non-zero exit code if either check finds something.
#
# Rationale (AMD Zen 2 / Ryzen 5 4600H):
#   pdep/pext exist in the Zen 2 ISA but are microcoded with ~18 cycles of
#   latency, whereas on Zen 3 and Intel Haswell+ they take ~3 cycles. Using
#   them in the Morton index computation would make each encode slower than
#   the locality benefit from the Z-order layout, defeating the whole point
#   of Phase 6.

set -euo pipefail

echo "=== PDEP/PEXT audit ==="

echo
echo "[1/3] Searching for PDEP/PEXT intrinsics in source code..."
HITS_SOURCE=$(grep -rn -E "_pdep_u(32|64)|_pext_u(32|64)" src/ || true)
if [ -n "$HITS_SOURCE" ]; then
    echo "FAIL: PDEP/PEXT intrinsics found in source code:"
    echo "$HITS_SOURCE"
    exit 1
fi
echo "OK: no intrinsics in src/."

echo
echo "[2/3] Compiling target binaries with -S for inspection..."
mkdir -p build/audit
# Include paths must cover both the algorithm's own directory and core/
# (matrix_utils.h, morton.h) so each TU compiles standalone.
AUDIT_INCS="-Isrc/core -Isrc/algorithms/morton"
for src in src/algorithms/morton/matmul_morton.c src/core/morton.c; do
    base=$(basename "$src" .c)
    gcc -O3 -march=znver2 -mavx2 -mfma -mbmi -mbmi2 \
        -S -o "build/audit/${base}.s" \
        $AUDIT_INCS "$src" || { echo "FAIL: could not compile $src"; exit 1; }
done
echo "OK: assembler emitted under build/audit/"

echo
echo "[3/3] Searching for pdep/pext in the emitted assembler..."
HITS_ASM=$(grep -in -E "\b(pdep|pext)[a-z]?\b" build/audit/*.s || true)
if [ -n "$HITS_ASM" ]; then
    echo "FAIL: the compiler emitted pdep/pext in one of the audited modules:"
    echo "$HITS_ASM"
    exit 1
fi
echo "OK: the compiler did not emit pdep/pext in the audited modules."

echo
echo "=== PDEP/PEXT audit: PASS ==="
