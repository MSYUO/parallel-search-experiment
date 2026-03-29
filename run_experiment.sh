#!/usr/bin/env bash
# run_experiment.sh — 분산 병렬 탐색 실험 자동화
# Usage: bash run_experiment.sh [--difficulty D] [--k K] [--trials T] [--test]

set -uo pipefail

# ── 기본값 ──────────────────────────────────────────────────────
DIFFICULTY=5
K_TARGET=10
TRIALS=30
N_LIST=(1 2 4 8 16)
RESULTS_DIR="results"
CSV_FILE="$RESULTS_DIR/experiment_results.csv"
MAX_RETRY=3
TEST_MODE=false
BUILT=false

# ── 인수 파싱 ────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --difficulty) DIFFICULTY=$2; shift 2 ;;
        --k)          K_TARGET=$2;   shift 2 ;;
        --trials)     TRIALS=$2;     shift 2 ;;
        --test)       TEST_MODE=true; shift ;;
        -h|--help)
            echo "Usage: bash run_experiment.sh [--difficulty D] [--k K] [--trials T] [--test]"
            exit 0 ;;
        *) echo "[ERROR] 알 수 없는 인수: $1"; exit 1 ;;
    esac
done

# ── 테스트 모드 ──────────────────────────────────────────────────
if $TEST_MODE; then
    DIFFICULTY=4
    K_TARGET=3
    TRIALS=2
    N_LIST=(1)
fi

# ── 초기화 ──────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR" figures

# ── 헬퍼 함수 ────────────────────────────────────────────────────
fmt_time() {
    # 초 → "Xm Ys" (소수점 입력도 처리)
    local s; s=$(python3 -c "print(int($1))" 2>/dev/null || echo "0")
    printf "%dm %02ds" $((s/60)) $((s%60))
}

already_done() {
    local trial=$1 n=$2
    [ -f "$CSV_FILE" ] && grep -q "^${trial},${n}," "$CSV_FILE"
}

get_last_treal() {
    tail -1 "$CSV_FILE" | cut -d',' -f5
}

n_avg_treal() {
    # CSV에서 특정 N의 평균 T_real 계산
    local n=$1
    python3 - <<EOF
import csv, sys
try:
    with open("$CSV_FILE") as f:
        rows = [r for r in csv.DictReader(f) if int(r['n_workers']) == $n]
    if rows:
        avg = sum(float(r['T_real']) for r in rows) / len(rows)
        print(f"{avg:.4f}")
    else:
        print("N/A")
except Exception as e:
    print("N/A")
EOF
}

# ── 완료 카운트 초기화 ────────────────────────────────────────────
TOTAL_TRIALS=$(( ${#N_LIST[@]} * TRIALS ))
DONE_COUNT=0
if [ -f "$CSV_FILE" ]; then
    for n in "${N_LIST[@]}"; do
        for trial in $(seq 1 $TRIALS); do
            already_done "$trial" "$n" && DONE_COUNT=$((DONE_COUNT+1)) || true
        done
    done
fi

EXPERIMENT_START=$(date +%s)
FIRST_TRIAL_SECS=""

# ── 배너 ─────────────────────────────────────────────────────────
echo "======================================================="
$TEST_MODE && echo " ★ TEST MODE: N=1, trial=1~2, DIFFICULTY=4, K=3"
echo " DIFFICULTY=$DIFFICULTY  K=$K_TARGET  TRIALS=$TRIALS"
echo " N list: ${N_LIST[*]}"
echo " Total: $TOTAL_TRIALS trials  (이미 완료: $DONE_COUNT)"
echo "======================================================="
echo ""

# ── 메인 루프 ────────────────────────────────────────────────────
for N in "${N_LIST[@]}"; do
    N_START=$(date +%s)
    echo "━━━ N=$N workers ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for TRIAL in $(seq 1 $TRIALS); do

        # resume: 이미 CSV에 있으면 스킵
        if already_done "$TRIAL" "$N"; then
            echo "[N=$N] Trial $TRIAL/$TRIALS  SKIPPED (CSV에 이미 존재)"
            DONE_COUNT=$((DONE_COUNT+1))
            continue
        fi

        # 재시도 루프
        RETRY=0
        SUCCESS=false

        while [ $RETRY -lt $MAX_RETRY ] && [ "$SUCCESS" = "false" ]; do
            [ $RETRY -gt 0 ] && echo "[N=$N] Trial $TRIAL  재시도 $RETRY/$MAX_RETRY..."
            TRIAL_START=$(date +%s)

            # a) 이전 trial JSON 삭제 + worker ID claim 파일 초기화 (CSV는 보존)
            rm -f "$RESULTS_DIR"/result_w*_t${TRIAL}.json
            rm -f "$RESULTS_DIR"/.claimed_id_*

            # b) docker compose up
            #    최초 1회만 --build, 이후는 캐시 사용
            BUILD_FLAG=""
            if [ "$BUILT" = "false" ]; then
                BUILD_FLAG="--build"
                BUILT=true
            fi

            COMPOSE_OK=true
            N_WORKERS=$N \
            DIFFICULTY=$DIFFICULTY \
            K_TARGET=$K_TARGET \
            TRIAL_NUMBER=$TRIAL \
            docker compose up $BUILD_FLAG --scale worker="$N" \
                2>&1 | sed "s/^/  /" || COMPOSE_OK=false

            # c) docker compose down (성공/실패 무관하게 항상)
            docker compose down --remove-orphans 2>/dev/null || true

            if [ "$COMPOSE_OK" = "false" ]; then
                echo "[N=$N] Trial $TRIAL  compose 실패"
                RETRY=$((RETRY+1))
                continue
            fi

            # d) merge_trial.py 실행
            MERGE_OK=true
            python merge_trial.py \
                --trial       "$TRIAL" \
                --n-workers   "$N" \
                --k           "$K_TARGET" \
                --difficulty  "$DIFFICULTY" \
                --results-dir "$RESULTS_DIR" \
                --csv         "$CSV_FILE" || MERGE_OK=false

            if [ "$MERGE_OK" = "false" ]; then
                echo "[N=$N] Trial $TRIAL  merge 실패"
                RETRY=$((RETRY+1))
                continue
            fi

            # ── 성공 ─────────────────────────────────────────
            TRIAL_END=$(date +%s)
            TRIAL_ELAPSED=$((TRIAL_END - TRIAL_START))
            SUCCESS=true
            DONE_COUNT=$((DONE_COUNT+1))
            T_REAL=$(get_last_treal)

            # 전체 예상 소요시간 (N=1 첫 trial 기준으로 1회만)
            if [ -z "$FIRST_TRIAL_SECS" ] && [ "$N" -eq "${N_LIST[0]}" ] && [ "$TRIAL" -eq 1 ]; then
                FIRST_TRIAL_SECS=$TRIAL_ELAPSED
                REMAINING=$((TOTAL_TRIALS - DONE_COUNT))
                ETA_SECS=$((FIRST_TRIAL_SECS * REMAINING))
                echo "[INFO] 전체 예상 소요시간: ~$(fmt_time $ETA_SECS)  (${TRIAL_ELAPSED}s/trial × ${REMAINING}회 남음)"
            fi

            ELAPSED_TOTAL=$((TRIAL_END - EXPERIMENT_START))
            printf "[N=%2d] Trial %2d/%d 완료 | T_real=%-9s | %ds/trial | 누적 경과: %s\n" \
                "$N" "$TRIAL" "$TRIALS" "${T_REAL}s" "$TRIAL_ELAPSED" "$(fmt_time $ELAPSED_TOTAL)"

        done  # retry loop

        if [ "$SUCCESS" = "false" ]; then
            echo "[ERROR] N=$N Trial $TRIAL: ${MAX_RETRY}회 모두 실패. 스킵."
        fi

    done  # trial loop

    # e) N 구간 평균 출력
    AVG=$(n_avg_treal "$N")
    N_ELAPSED=$(($(date +%s) - N_START))
    echo "[N=$N] 구간 완료 | 평균 T_real=${AVG}s | 구간 소요: $(fmt_time $N_ELAPSED)"
    echo ""

done  # N loop

TOTAL_ELAPSED=$(($(date +%s) - EXPERIMENT_START))
echo "======================================================="
echo " 실험 완료!"
echo " 총 소요: $(fmt_time $TOTAL_ELAPSED)"
echo " 결과 파일: $CSV_FILE"
echo "======================================================="
