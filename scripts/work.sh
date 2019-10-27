# /bin/bash
b=40
while true 
do
	sec=`date +%M`
	hours=`date +%H`
	echo $hours:$sec
	fg=`expr $hours % 2`
	hours=`expr $fg \* 60`
	time=`expr $sec + $hours`
	time=`expr $time % $b`
	if [[ $time = 11 ]];
	then
		./start_scale.sh
		echo "start"
	fi
	
	if [[ $time = 31 ]];
	then 
		echo "stop"
		./stop_scale.sh
	fi
	sleep 20
done
