#!/bin/bash
# Continuously monitor TBSS randomise progress
# Usage: ./watch_tbss_progress.sh
# Press Ctrl+C to exit

ALL_METRICS="FA MD AD RD MK AK RK KFA FICVF ODI FISO"

while true; do
  clear
  echo "========================================"
  echo "TBSS Randomise Progress Monitor"
  echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "========================================"
  echo

  completed=0
  running=0

  for metric in $ALL_METRICS; do
    logfile="/mnt/bytopia/IRC805/analysis/tbss/$metric/randomise_output/randomise.log"

    if [ -f "$logfile" ]; then
      last_line=$(tail -1 "$logfile")

      if echo "$last_line" | grep -q "Starting permutation"; then
        perm=$(echo "$last_line" | grep -oP 'Starting permutation \K[0-9]+')
        progress=$(echo "scale=1; $perm * 100 / 5000" | bc)
        bar_width=30
        filled=$(echo "scale=0; $perm * $bar_width / 5000" | bc)
        empty=$((bar_width - filled))
        bar=$(printf "%${filled}s" | tr ' ' '█')$(printf "%${empty}s" | tr ' ' '░')
        printf "%-8s: [%s] %4d/5000 (%5.1f%%)\n" "$metric" "$bar" "$perm" "$progress"
        running=$((running + 1))
      elif echo "$last_line" | grep -q "Finished"; then
        printf "%-8s: ✓ COMPLETED\n" "$metric"
        completed=$((completed + 1))
      else
        echo "$metric: Running..."
        running=$((running + 1))
      fi
    else
      echo "$metric: Waiting to start..."
    fi
  done

  echo
  echo "========================================"
  printf "Summary: %d completed, %d running\n" "$completed" "$running"
  echo "Active randomise processes: $(ps aux | grep 'randomise.*all_' | grep -v grep | wc -l)"
  echo "========================================"
  echo
  echo "Press Ctrl+C to exit"

  # Exit if all completed
  if [ $completed -eq 11 ]; then
    echo
    echo "🎉 All metrics completed!"
    exit 0
  fi

  sleep 10
done
