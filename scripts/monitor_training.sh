#!/bin/bash
# Training monitor - Real-time throughput and GPU balance tracking
# Usage: bash scripts/monitor_training.sh /path/to/train.log

LOG_FILE="$1"
if [[ -z "$LOG_FILE" ]]; then
  echo "Usage: $0 <log_file>"
  exit 1
fi

echo "Monitoring training: $LOG_FILE"
echo "Press Ctrl+C to stop"
echo "=" | head -c 80 && echo

LAST_STEP=0
LAST_TIME=$(date +%s)

while true; do
  if [[ -f "$LOG_FILE" ]]; then
    # Extract latest step and time/it
    LATEST=$(tail -50 "$LOG_FILE" | grep -oP '\d+%\|.*?\d+/\d+.*?\[.*?\]' | tail -1)
    
    if [[ -n "$LATEST" ]]; then
      # Parse step number
      CURRENT_STEP=$(echo "$LATEST" | grep -oP '\d+/\d+' | cut -d'/' -f1)
      
      # Parse time per iteration
      TIME_PER_IT=$(echo "$LATEST" | grep -oP '\d+\.\d+s/it' | grep -oP '\d+\.\d+')
      
      CURRENT_TIME=$(date +%s)
      ELAPSED=$((CURRENT_TIME - LAST_TIME))
      
      if [[ $ELAPSED -ge 10 ]] && [[ $CURRENT_STEP -gt $LAST_STEP ]]; then
        STEPS_DONE=$((CURRENT_STEP - LAST_STEP))
        THROUGHPUT=$(echo "scale=2; $STEPS_DONE / $ELAPSED" | bc)
        
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Step: $CURRENT_STEP | Speed: ${TIME_PER_IT}s/it | Throughput: ${THROUGHPUT} steps/s"
        
        # GPU memory balance check
        GPU_MEMS=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')
        echo "  GPU Mem (MB): $GPU_MEMS"
        
        LAST_STEP=$CURRENT_STEP
        LAST_TIME=$CURRENT_TIME
      fi
    fi
  else
    echo "[WARN] Log file not found: $LOG_FILE"
  fi
  
  sleep 10
done
