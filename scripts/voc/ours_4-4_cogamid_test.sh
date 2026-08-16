#!/bin/bash

set -e

start=`date +%s`

START_DATE=$(date '+%Y-%m-%d')

PORT=$((9000 + RANDOM % 1000))
GPU=0,1
NB_GPU=2

DATA_ROOT='./data/PascalVOC12'
DATASET=voc
TASK=4-4
NAME=CoGaMiD_test
INCREMENTAL_METHOD=CoGaMiD

STEPS_GLOBAL=1
TASK_NUM=2
EPOCHS_GLOBAL=`expr ${STEPS_GLOBAL} \* ${TASK_NUM}`

NUM_CLIENTS=1
ADD_CLIENTS=1
LOCAL_CLIENTS=1
CLASS_RATIO=0.5
SAMPLE_RATIO2=0.05
BATCH_SIZE=2
EPOCHS_LOCAL=1
CROP_SIZE=128
NUM_WORKERS=2

GMM_COMPONENTS=1
GMM_MIN_FEATURES=1
GMM_MAX_FEATURES=256
GMM_PSEUDO_THRESHOLD=0.0
GMM_EM_ITERS=1
COGAMID_REPLAY_MAX_PER_CLASS=8

SEED=2023
echo ${EPOCHS_GLOBAL}

SCREENNAME="${DATASET}_${TASK}_${NAME} On GPUs ${GPU}"

RESULTSFILE=results/seed_${SEED}-ov/${START_DATE}_${DATASET}_${TASK}_${NAME}.csv
rm -f ${RESULTSFILE}

echo -ne "\ek${SCREENNAME}\e\\"

echo "Writing in ${RESULTSFILE}"
echo "This smoke test runs Step 0 and Step 1 only"

CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.run --master_port ${PORT} --nproc_per_node=${NB_GPU} fl_main.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --dataset ${DATASET} --name ${NAME} --task ${TASK} --incremental_method ${INCREMENTAL_METHOD} --num_clients ${NUM_CLIENTS} --add_clients ${ADD_CLIENTS} --local_clients ${LOCAL_CLIENTS} --class_ratio ${CLASS_RATIO} --sample_ratio2 ${SAMPLE_RATIO2} --batch_size ${BATCH_SIZE} --epochs_local ${EPOCHS_LOCAL} --steps_global ${STEPS_GLOBAL} --epochs_global ${EPOCHS_GLOBAL} --crop_size ${CROP_SIZE} --num_workers ${NUM_WORKERS} --seed ${SEED} --backbone resnet50 --no_pretrained --gmm_components ${GMM_COMPONENTS} --gmm_min_features ${GMM_MIN_FEATURES} --gmm_max_features ${GMM_MAX_FEATURES} --gmm_pseudo_threshold ${GMM_PSEUDO_THRESHOLD} --gmm_em_iters ${GMM_EM_ITERS} --cogamid_replay_max_per_class ${COGAMID_REPLAY_MAX_PER_CLASS} --opt_level O1

echo ${SCREENNAME}

end=`date +%s`
runtime=$((end-start))
echo "Smoke test finished in ${runtime}s"
