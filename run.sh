#!/bin/bash

trap 'kill $(jobs -p)' SIGINT SIGTERM EXIT
config=${2:-"Config/all.conf"}
source $config
if [ -z "$redis_server" ]; then
    redis_server=$3
fi

echo "redis server:" $redis_server

pids=""

if $leaner; then
    if [ -z "$actor_device" ]; then
	python Learner.py -r $redis_server &
	pids="$pids $!"
    else
	python learner_node.py &
	pids="$pids $!"
    fi
fi

if $actor; then
    for i in seq $num_actors
    do
	python Actor.py -n $num_actors -i $(($i-1)) -r $redis_server -d $$(((($i-1)/2)+3)) &
	pids="$pids $!"
    done
fi

wait -n $pids
