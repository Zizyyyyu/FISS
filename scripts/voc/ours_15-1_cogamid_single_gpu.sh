#!/bin/bash

set -e

start=`date +%s`

START_DATE=$(date '+%Y-%m-%d')

PORT=29511
GPU=0
NB_GPU=1

DATA_ROOT='./data/PascalVOC12'
DATASET=voc
TASK=15-1
NAME=CoGaMiD
INCREMENTAL_METHOD=CoGaMiD
STEPS_GLOBAL=5
TASK_NUM=6
EPOCHS_GLOBAL=`expr ${STEPS_GLOBAL} \* ${TASK_NUM}`

CLASS_RATIO=0.5
SAMPLE_RATIO2=0.6
BATCH_SIZE=12
EPOCHS_LOCAL=6

GMM_COMPONENTS=3
GMM_MIN_FEATURES=30
GMM_MAX_FEATURES=20000
GMM_PSEUDO_THRESHOLD=0.7
GMM_EM_ITERS=30

COGAMID_MBCE=1.0
COGAMID_PKD=5.0
COGAMID_CONT=0.05
COGAMID_UNCER=0.1
COGAMID_POS_WEIGHT=4.0
COGAMID_REPLAY_MAX_PER_CLASS=512
COGAMID_FEATURE_NOISE=1.0

SEED=2023
echo ${EPOCHS_GLOBAL}

SCREENNAME="${DATASET}_${TASK}_${NAME} On GPUs ${GPU}"

RESULTSFILE=results/seed_${SEED}-ov/${START_DATE}_${DATASET}_${TASK}_${NAME}.csv
rm -f ${RESULTSFILE}

echo -ne "\ek${SCREENNAME}\e\\"

echo "Writing in ${RESULTSFILE}"

CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.run --master_port ${PORT} --nproc_per_node=${NB_GPU} fl_main.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --dataset ${DATASET} --name ${NAME} --task ${TASK} --incremental_method ${INCREMENTAL_METHOD} --class_ratio ${CLASS_RATIO} --sample_ratio2 ${SAMPLE_RATIO2} --batch_size ${BATCH_SIZE} --epochs_local ${EPOCHS_LOCAL} --steps_global ${STEPS_GLOBAL} --epochs_global ${EPOCHS_GLOBAL} --seed ${SEED} --gmm_components ${GMM_COMPONENTS} --gmm_min_features ${GMM_MIN_FEATURES} --gmm_max_features ${GMM_MAX_FEATURES} --gmm_pseudo_threshold ${GMM_PSEUDO_THRESHOLD} --gmm_em_iters ${GMM_EM_ITERS} --cogamid_mbce ${COGAMID_MBCE} --cogamid_pkd ${COGAMID_PKD} --cogamid_cont ${COGAMID_CONT} --cogamid_uncer ${COGAMID_UNCER} --cogamid_pos_weight ${COGAMID_POS_WEIGHT} --cogamid_replay_max_per_class ${COGAMID_REPLAY_MAX_PER_CLASS} --cogamid_feature_noise ${COGAMID_FEATURE_NOISE} --opt_level O1

echo ${SCREENNAME}

end=`date +%s`
runtime=$((end-start))
echo "Run in ${runtime}s"
