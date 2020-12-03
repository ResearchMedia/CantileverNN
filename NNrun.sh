#!/bin/bash
echo "name : $1"
echo "hyperparam : $2"
echo "GPU : $3"
echo "varstop : $4"
echo "Now the real stuff runs"
echo "python main_class.py --hyperparam-json $2 --GPU $3 --varstop $4"
#touch "python main_broad_FEA.py --hyperparam-json $2 --GPU $3 --varstop $4"
if [ -f "/opt/anaconda/etc/profile.d/conda.sh" ]; then
    . "/opt/anaconda/etc/profile.d/conda.sh"
    CONDA_CHANGEPS1=false conda activate PyTorchNN
    echo "Activated conda env"
fi
{
	echo $1
	python main_class.py --hyperparam-json $2 --GPU $3 --varstop $4
} 2>&1 | tee -a "$2.log"

# Make sure the GPUFREE file doesn't get overwritten (wait until NNsweep.sh or NNtrain.sh delete the prev. version)
FILE="$3.GPUFREE" 
if test -f "$FILE"; then
	while test -f "$FILE"
	do
		sleep 15
	    echo "$FILE exists."
	done
fi
	touch "$3.GPUFREE"

#read -p "$*"
