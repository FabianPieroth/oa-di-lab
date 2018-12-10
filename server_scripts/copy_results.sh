#!/bin/bash
echo 'Welcome to the copy_results_script!'
read -p "Please enter your preferred copying interval in minutes: " COP_INT
SEC_IN_MIN=60
COP_INT_SEC=$((COP_INT*SEC_IN_MIN))
echo 'starting the automatic copying'
timestamp() {
  date +"%T"
}

while true; do
  echo '-------------------------------------------------'
  echo 'start copying at:' 
  timestamp
  cp -r -u /ssdtemp/local/di-lab/reports /ssdtemp/mounted/di-lab;
  echo '------------------------------------------------'
  echo 'done copying at:' 
  timestamp
  echo  'now wait for 5min'
  echo '------------------------------------------------'
  sleep $COP_INT_SEC
done

