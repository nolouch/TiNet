# /bin/bash
# 周期 40 分钟
cycle=40
while true 
do
	sec=`date +%M`
	hours=`date +%H`
	echo $hours:$sec 
	fg=`expr $hours % 2`
	hours=`expr $fg \* 60`
	time=`expr $sec + $hours`
	time=`expr $time % $cycle`
	if [[ $time = 11 ]];
	then
		./start_scale.sh
		echo "start reject-region-scheduler"
	fi
	
	if [[ $time = 31 ]];
	then
		./stop_scale.sh
		echo "stop reject-region-scheduler"
	fi
	sleep 20
done
