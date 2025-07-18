#!/bin/bash

name=${1:-test_raytiling_speed}

while true ; do
    pid=$(pidof $name 2>/dev/null)
    if [ -n "$pid" ] ; then
        ps -p $pid -o pid,rss,vsize,%cpu,args
        nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,temperature.gpu,power.draw --format=csv
    else
        echo "no pid for $name"
    fi
    sleep 1
done
      
