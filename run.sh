#!/bin/bash
#EXIT is used to kill all child process when this script exit(finish all commands).
# Without EXIT, when it exits, the child processes will still be on, but the process of this script will disappear
trap 'kill $(jobs -p)' SIGINT SIGTERM EXIT

config=${2:-"Config/Self_Play.conf"}
source $config 
if [ -z "$redis_server" ]; then
    redis_server=$3 
fi

echo "redis server:" $redis_server 

pids="" 
if $human_play; then
    python Learner.py -rc $Train_human_play_config &
    pids="$pids $!"
	python Human_play.py -m l &
	pids="$pids $!"
fi
wait -f $pids
pids=""
if $leaner; then
	python Learner.py -rc $Train_config &
	pids="$pids $!"
fi

if $actor; then
    for i in `seq $num_actors`
    do
    python Actor.py -n $num_actors -i $(($i-1)) -rc $Train_config -r $redis_server -d $((($i-1)+$first_device)) -s $(($i-1)) &
    pids="$pids $!"
    done
fi
# Exit the this script when there's a child process is done, so all child process will be killed
wait -n $pids
