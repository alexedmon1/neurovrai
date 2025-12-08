#!/bin/bash
# Monitor TBSS randomise progress
# Usage: ./check_tbss_progress.sh

ALL_METRICS="FA MD AD RD MK AK RK KFA FICVF ODI FISO"

echo "========================================"
echo "TBSS Randomise Progress Monitor"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo

for metric in $ALL_METRICS; do
  logfile="/mnt/bytopia/IRC805/analysis/tbss/$metric/randomise_output/randomise.log"

  if [ -f "$logfile" ]; then
    # Extract current permutation number
    last_line=$(tail -1 "$logfile")

    if echo "$last_line" | grep -q "Starting permutation"; then
      perm=$(echo "$last_line" | grep -oP 'Starting permutation \K[0-9]+')
      progress=$(echo "scale=1; $perm * 100 / 5000" | bc)
      printf "%-8s: Permutation %4d/5000 (%5.1f%%)\n" "$metric" "$perm" "$progress"
    elif echo "$last_line" | grep -q "Finished"; then
      echo "$metric: ✓ COMPLETED"
    else
      echo "$metric: Running... ($last_line)"
    fi
  else
    echo "$metric: Not started"
  fi
done

echo
echo "========================================"
echo "Active randomise processes: $(ps aux | grep 'randomise.*all_' | grep -v grep | wc -l)"
echo "========================================"
