#!/bin/bash
# Re-launch run_batch2.py until the batch is complete.
#
# Each invocation does a few chunks and exits, so memory is returned to the OS by
# process death rather than trusted to be freed inside a long-lived interpreter.
#
#   0  done          10 more work remains
#   20 disk floor    -> fatal, a human must free space
#   21 RAM floor     -> transient. Free RAM on this machine was observed swinging
#                       441 -> 1652 MB during one run because of other activity,
#                       so waiting is usually enough. Retry, do not give up.
PY=".venv/Scripts/python.exe"
CHUNKS_PER_RUN=4
MAX_INVOCATIONS=80
RAM_WAIT_SECONDS=180
MAX_CONSECUTIVE_RAM_PAUSES=12

ram_pauses=0
for i in $(seq 1 $MAX_INVOCATIONS); do
    echo "===== invocation $i ($(date '+%H:%M:%S')) ====="
    $PY -u run_batch2.py --run --max-chunks $CHUNKS_PER_RUN
    code=$?
    echo "===== invocation $i exited $code ($(date '+%H:%M:%S')) ====="
    case $code in
        0)  echo "BATCH COMPLETE"; exit 0 ;;
        10) ram_pauses=0 ;;
        21) ram_pauses=$((ram_pauses+1))
            if [ $ram_pauses -ge $MAX_CONSECUTIVE_RAM_PAUSES ]; then
                echo "RAM stayed low across $ram_pauses consecutive attempts - giving up"
                exit 21
            fi
            echo "RAM pause $ram_pauses/$MAX_CONSECUTIVE_RAM_PAUSES - waiting ${RAM_WAIT_SECONDS}s"
            sleep $RAM_WAIT_SECONDS ;;
        20) echo "STOPPED: disk floor - needs a human"; exit 20 ;;
        *)  echo "UNEXPECTED EXIT $code - stopping driver"; exit $code ;;
    esac
done
echo "hit MAX_INVOCATIONS without completing"
exit 1
