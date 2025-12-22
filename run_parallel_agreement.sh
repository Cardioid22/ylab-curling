#!/bin/bash
################################################################################
# Parallel Agreement Experiment Runner
#
# This script runs agreement experiments in parallel across multiple test cases.
# Uses nohup to keep processes running even after logout.
#
# Usage:
#   ./run_parallel_agreement.sh [OPTIONS]
#
# Options:
#   --cn N          Number of clusters (default: 4)
#   --d D           Search depth (default: 1)
#   --test-num T    Number of test patterns per type (default: 10)
#   --parallel P    Number of parallel jobs (default: 10)
#   --start-id S    Starting test ID (default: 0)
#   --end-id E      Ending test ID (default: auto-calculated from test-num)
#
# Example:
#   ./run_parallel_agreement.sh --cn 4 --d 1 --test-num 10 --parallel 10
################################################################################

set -e  # Exit on error

# Default parameters
CLUSTER_NUM=4
DEPTH=1
TEST_NUM=10
PARALLEL_JOBS=10
START_ID=0
END_ID=-1  # Will be calculated
SIM_COUNT=10

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cn)
            CLUSTER_NUM="$2"
            shift 2
            ;;
        --d)
            DEPTH="$2"
            shift 2
            ;;
        --test-num)
            TEST_NUM="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --start-id)
            START_ID="$2"
            shift 2
            ;;
        --end-id)
            END_ID="$2"
            shift 2
            ;;
        --sim)
            SIM_COUNT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--cn N] [--d D] [--test-num T] [--parallel P] [--start-id S] [--end-id E] [--sim S]"
            exit 1
            ;;
    esac
done

# Calculate total test cases
TOTAL_TESTS=$((TEST_NUM * 10))

# If END_ID not specified, calculate from test_num
if [ $END_ID -eq -1 ]; then
    END_ID=$((TOTAL_TESTS - 1))
fi

# Validate parameters
if [ $START_ID -gt $END_ID ]; then
    echo "Error: START_ID ($START_ID) cannot be greater than END_ID ($END_ID)"
    exit 1
fi

if [ $END_ID -ge $TOTAL_TESTS ]; then
    echo "Error: END_ID ($END_ID) must be less than TOTAL_TESTS ($TOTAL_TESTS)"
    exit 1
fi

# Calculate grid size (assuming 4x4 = 16)
GRID_SIZE=16

# Create timestamp
TIMESTAMP=$(date +%Y%m%d%H%M%S)

# Create result directory
RESULT_DIR="experiments/parallel_agreement_results/Grid${GRID_SIZE}_Depth${DEPTH}_Clusters${CLUSTER_NUM}_Tests${TOTAL_TESTS}_${TIMESTAMP}"
mkdir -p "$RESULT_DIR"

# Export result directory as environment variable
export AGREEMENT_RESULT_DIR="$RESULT_DIR"

# Create log directory
LOG_DIR="$RESULT_DIR/logs"
mkdir -p "$LOG_DIR"

# Save experiment configuration
CONFIG_FILE="$RESULT_DIR/experiment_config.txt"
cat > "$CONFIG_FILE" << EOF
Experiment Configuration
========================
Timestamp: $(date)
Cluster Number: $CLUSTER_NUM
Search Depth: $DEPTH
Test Patterns per Type: $TEST_NUM
Total Test Cases: $TOTAL_TESTS
Parallel Jobs: $PARALLEL_JOBS
Start Test ID: $START_ID
End Test ID: $END_ID
Simulations per Shot: $SIM_COUNT
Grid Size: $GRID_SIZE
Result Directory: $RESULT_DIR
EOF

echo "========================================"
echo "Parallel Agreement Experiment"
echo "========================================"
cat "$CONFIG_FILE"
echo "========================================"
echo ""

# Find executable
if [ -f "build/Release/ylab_client.exe" ]; then
    EXECUTABLE="build/Release/ylab_client.exe"
elif [ -f "build/Release/ylab_client" ]; then
    EXECUTABLE="build/Release/ylab_client"
elif [ -f "ylab_client.exe" ]; then
    EXECUTABLE="./ylab_client.exe"
elif [ -f "ylab_client" ]; then
    EXECUTABLE="./ylab_client"
else
    echo "Error: ylab_client executable not found!"
    echo "Please build the project first."
    exit 1
fi

echo "Using executable: $EXECUTABLE"
echo ""

# Progress tracking file
PROGRESS_FILE="$RESULT_DIR/progress.txt"
echo "0/$((END_ID - START_ID + 1))" > "$PROGRESS_FILE"

# Function to run a single test
run_test() {
    local test_id=$1
    local log_file="$LOG_DIR/test_$(printf '%03d' $test_id).log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting test $test_id" >> "$log_file"

    # Run the test with nohup
    nohup "$EXECUTABLE" --agreement-experiment \
        --test-id "$test_id" \
        --cn "$CLUSTER_NUM" \
        --d "$DEPTH" \
        --test-num "$TEST_NUM" \
        --sim "$SIM_COUNT" \
        >> "$log_file" 2>&1

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Test $test_id completed successfully" >> "$log_file"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Test $test_id failed with exit code $exit_code" >> "$log_file"
    fi

    return $exit_code
}

# Export function for parallel execution
export -f run_test
export EXECUTABLE CLUSTER_NUM DEPTH TEST_NUM SIM_COUNT LOG_DIR

echo "Starting parallel execution..."
echo "Progress will be logged to: $RESULT_DIR/run_progress.log"
echo ""

# Create master log file
MASTER_LOG="$RESULT_DIR/run_progress.log"
echo "Parallel Agreement Experiment - Started at $(date)" > "$MASTER_LOG"
echo "=======================================" >> "$MASTER_LOG"

# Run tests in parallel using GNU parallel or xargs
if command -v parallel &> /dev/null; then
    # Use GNU parallel (preferred)
    echo "Using GNU parallel for execution" | tee -a "$MASTER_LOG"
    seq $START_ID $END_ID | parallel -j "$PARALLEL_JOBS" --bar run_test {}
elif command -v xargs &> /dev/null; then
    # Use xargs as fallback
    echo "Using xargs for execution" | tee -a "$MASTER_LOG"
    seq $START_ID $END_ID | xargs -P "$PARALLEL_JOBS" -I {} bash -c 'run_test "$@"' _ {}
else
    # Manual parallel execution with background jobs
    echo "Using manual backgrounding for execution" | tee -a "$MASTER_LOG"

    running_jobs=0
    for test_id in $(seq $START_ID $END_ID); do
        # Wait if we've reached the parallel limit
        while [ $running_jobs -ge $PARALLEL_JOBS ]; do
            sleep 1
            running_jobs=$(jobs -r | wc -l)
        done

        # Start new job in background
        run_test $test_id &
        running_jobs=$((running_jobs + 1))

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launched test $test_id (running: $running_jobs/$PARALLEL_JOBS)" | tee -a "$MASTER_LOG"
    done

    # Wait for all background jobs to finish
    echo "Waiting for all tests to complete..." | tee -a "$MASTER_LOG"
    wait
fi

echo "" | tee -a "$MASTER_LOG"
echo "=======================================" | tee -a "$MASTER_LOG"
echo "All tests completed at $(date)" | tee -a "$MASTER_LOG"
echo "=======================================" | tee -a "$MASTER_LOG"

# Count successful and failed tests
SUCCESS_COUNT=$(find "$RESULT_DIR" -name "test_*_result.csv" | wc -l)
TOTAL_RUN=$((END_ID - START_ID + 1))
FAIL_COUNT=$((TOTAL_RUN - SUCCESS_COUNT))

echo "" | tee -a "$MASTER_LOG"
echo "Summary:" | tee -a "$MASTER_LOG"
echo "  Total tests: $TOTAL_RUN" | tee -a "$MASTER_LOG"
echo "  Successful: $SUCCESS_COUNT" | tee -a "$MASTER_LOG"
echo "  Failed: $FAIL_COUNT" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Results saved to: $RESULT_DIR" | tee -a "$MASTER_LOG"

# List any missing test results
if [ $FAIL_COUNT -gt 0 ]; then
    echo "" | tee -a "$MASTER_LOG"
    echo "Missing test results:" | tee -a "$MASTER_LOG"
    for test_id in $(seq $START_ID $END_ID); do
        result_file=$(printf "$RESULT_DIR/test_%03d_result.csv" $test_id)
        if [ ! -f "$result_file" ]; then
            echo "  Test $test_id" | tee -a "$MASTER_LOG"
        fi
    done
fi

echo ""
echo "Experiment complete!"
echo "Check detailed logs in: $LOG_DIR"
