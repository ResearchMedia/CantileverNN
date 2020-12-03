#!/bin/bash
echo "name : $1"
echo "hyperparam : $2"
echo "GPU : $3"
echo "varstop : $4"

pwd
echo "Clear all existing *.GPUFREE files"
rm *.GPUFREE

varstop=0.5
GPUCAP=1
# declare gpu free variables
GPU0=0
GPU1=0
GPU2=0
# declare gpu filenames
FGPU0=0.GPUFREE
FGPU1=1.GPUFREE
FGPU2=2.GPUFREE

declare -a hyperparams=("hyperparam_ef123_T.json" "hyperparam_ef123_T.json" "hyperparam_ef123_T.json")
#declare -a hyperparams=("hyperparam_ef123.json" "hyperparam_ef123.json" "hyperparam_ef123.json")
#declare -a hyperparams=("hyperparam_CurlDisp_Y.json" "hyperparam_CurlDisp_Y_T.json" "hyperparam_ef1.json" "hyperparam_ef1_T.json" "hyperparam_ef2.json" "hyperparam_ef2_T.json" "hyperparam_ef3.json" "hyperparam_ef3_T.json" "hyperparam_PrincStrain_X.json" "hyperparam_PrincStrain_X_T.json" "hyperparam_TotDisp.json" "hyperparam_TotDisp_T.json" "hyperparam_TotVonMises.json" "hyperparam_TotVonMises_T.json" "hyperparam_CurlDisp_Y.json" "hyperparam_CurlDisp_Y_T.json" "hyperparam_ef1.json" "hyperparam_ef1_T.json" "hyperparam_ef2.json" "hyperparam_ef2_T.json" "hyperparam_ef3.json" "hyperparam_ef3_T.json" "hyperparam_PrincStrain_X.json" "hyperparam_PrincStrain_X_T.json" "hyperparam_TotDisp.json" "hyperparam_TotDisp_T.json" "hyperparam_TotVonMises.json" "hyperparam_TotVonMises_T.json" "hyperparam_CurlDisp_Y.json" "hyperparam_CurlDisp_Y_T.json" "hyperparam_ef1.json" "hyperparam_ef1_T.json" "hyperparam_ef2.json" "hyperparam_ef2_T.json" "hyperparam_ef3.json" "hyperparam_ef3_T.json" "hyperparam_PrincStrain_X.json" "hyperparam_PrincStrain_X_T.json" "hyperparam_TotDisp.json" "hyperparam_TotDisp_T.json" "hyperparam_TotVonMises.json" "hyperparam_TotVonMises_T.json")
#declare -a hyperparams=("hyperparam_ef123.json" "hyperparam_TotDisp.json" "hyperparam_TotStress.json")
echo "${#hyperparams[@]} Parameter to run"

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
	sleep 65
done

rm *.GPUFREE
