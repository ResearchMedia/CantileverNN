#!/bin/bash
TIMEFORMAT='It takes %R seconds to complete this task...'
time {

# loading hyper parameter file
if [ $# -eq 0 ]
  then
    echo "Provide sweep list: bash NNsweep.sh sweep_list.txt"
    exit 1
fi

echo "Sweep List provided: $1"
file_as_string=$(<$1)
hyperparams=($file_as_string)
echo "${#hyperparams[@]} Parameter to run"
echo ${hyperparams[@]}

# handling GPU availability
pwd
echo "Clear all existing *.GPUFREE files"
rm *.GPUFREE

varstop=100
GPUCAP=3
# declare gpu free variables
GPU0=0
GPU1=0
GPU2=0
# declare gpu filenames
FGPU0=0.GPUFREE
FGPU1=1.GPUFREE
FGPU2=2.GPUFREE


i=0
until (( "$i" >= "${#hyperparams[@]}" ))
do
	if [ "$GPU0" -lt "$GPUCAP" ] ; then
		echo "${hyperparams[$i]} on GPU0"
		#echo "../NNrun.sh GPU2 ${hyperparams[$i]} 0 $varstop" 
		screen -dmS GPU0 bash -c "../NNrun.sh GPU0 ${hyperparams[$i]} 0 $varstop" 
		((GPU0++))
		((i++))
	elif [ "$GPU1" -lt "$GPUCAP" ] ; then
		echo "${hyperparams[$i]} on GPU1"
		#echo "../NNrun.sh GPU2 ${hyperparams[$i]} 1 $varstop" 
		screen -dmS GPU1 bash -c "../NNrun.sh GPU1 ${hyperparams[$i]} 1 $varstop"
		((GPU1++))
		((i++))
	elif [ "$GPU2" -lt "$GPUCAP" ] ; then
		echo "${hyperparams[$i]} on GPU2"
		#echo "../NNrun.sh GPU2 ${hyperparams[$i]} 2 $varstop" 
		screen -dmS GPU2 bash -c "../NNrun.sh GPU2 ${hyperparams[$i]} 2 $varstop"
		((GPU2++))
		((i++))
	fi
	# check if GPUs were freed
	if [ -f "$FGPU0" ]; then
		echo "$FGPU0 exists"
		((GPU0--))
		rm $FGPU0
	elif [ -f "$FGPU1" ]; then
		echo "$FGPU1 exists"
		((GPU1--))
		rm $FGPU1
	elif [ -f "$FGPU2" ]; then
		echo "$FGPU2 exists"
		((GPU2--))
		rm $FGPU2
	fi
	sleep 61
done
}
echo "Training Process Completed"
echo "Please press any key to confirm that all processes have returned."
echo "Deleting all .GPUFREE files"
while [ true ] ; do
	read -t 3 -n 1
	if [ $? = 0 ] ; then
		exit ;
	else
		rm *.GPUFREE
		sleep 15
	fi
done
rm *.GPUFREE
