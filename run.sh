#!/bin/bash

trap 'kill $(jobs -p)' SIGINT SIGTERM EXIT 
config=${2:-"Config/all.conf"} 
source $config 
if [ -z "$redis_server" ]; then
    redis_server=$3 
fi

echo "redis server:" $redis_server 

pids="" 
if $human_play; then
    if $leaner; then
    python Learner.py -c Train_human_play.yaml &
    pids="$pids $!"
	fi
	python Human_play.py -m l &
	pids="$pids $!"
fi
wait -f $pids

pids=""
if $leaner; then
	python Learner.py -c Train.yaml &
	pids="$pids $!"
fi

if $actor; then
    for i in `seq $num_actors`
    do
    python Actor.py -n $num_actors -i $(($i-1)) -r $redis_server -d $((($i-1)+$first_device)) -s $(($i-1)) &
    pids="$pids $!"
    done
fi

wait -n $pids
trap 'kill $pids' SIGINT SIGTERM EXIT
