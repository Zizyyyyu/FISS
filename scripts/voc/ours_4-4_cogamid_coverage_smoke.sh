#!/bin/bash

set -e
set -o pipefail

start=`date +%s`

START_DATE=$(date '+%Y-%m-%d')
RUN_ID=$(date '+%Y%m%d_%H%M%S')

PORT=$((10000 + RANDOM % 10000))
GPU=${GPU:-0,1}
NB_GPU=${NB_GPU:-2}

DATA_ROOT='./data/PascalVOC12'
DATASET=voc
TASK=4-4
NAME=CoGaMiD_coverage_smoke_${RUN_ID}
INCREMENTAL_METHOD=CoGaMiD

STEPS_GLOBAL=1
TASK_NUM=2
EPOCHS_GLOBAL=`expr ${STEPS_GLOBAL} \* ${TASK_NUM}`

NUM_CLIENTS=2
ADD_CLIENTS=1
LOCAL_CLIENTS=1
GMM_CLIENTS=2
CLASS_RATIO=0.5
SAMPLE_RATIO2=0.01
BATCH_SIZE=${BATCH_SIZE:-2}
EPOCHS_LOCAL=1
CROP_SIZE=64
NUM_WORKERS=0

GMM_COMPONENTS=1
GMM_MIN_FEATURES=1
GMM_MAX_FEATURES=64
GMM_BATCH_SIZE=1
GMM_EM_ITERS=1
COGAMID_REPLAY_MAX_PER_CLASS=4

SEED=2023
CHECKPOINT_ROOT=./checkpoints_smoke/${RUN_ID}
CHECKPOINT_DIR=${CHECKPOINT_ROOT}/seed_${SEED}-ov
LOG_DIR=logs
LOG_FILE=${LOG_DIR}/4-4_coverage_smoke_${RUN_ID}.log

mkdir -p ${LOG_DIR}

exec > >(tee -a ${LOG_FILE}) 2>&1

log(){
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "Smoke test log: ${LOG_FILE}"
log "Smoke test checkpoint directory: ${CHECKPOINT_DIR}"
log "This smoke test runs Step 0 and Step 1 with ${NUM_CLIENTS}->`expr ${NUM_CLIENTS} + ${ADD_CLIENTS}` clients"
log "Step 1 must collect coverage reports and select ${GMM_CLIENTS} clients for formal GMM fitting"

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.run --master_port ${PORT} --nproc_per_node=${NB_GPU} fl_main.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --test_on_val --dataset ${DATASET} --name ${NAME} --task ${TASK} --incremental_method ${INCREMENTAL_METHOD} --num_clients ${NUM_CLIENTS} --add_clients ${ADD_CLIENTS} --local_clients ${LOCAL_CLIENTS} --gmm_clients ${GMM_CLIENTS} --fit_final_gmm --class_ratio ${CLASS_RATIO} --sample_ratio2 ${SAMPLE_RATIO2} --batch_size ${BATCH_SIZE} --epochs_local ${EPOCHS_LOCAL} --steps_global ${STEPS_GLOBAL} --epochs_global ${EPOCHS_GLOBAL} --crop_size ${CROP_SIZE} --num_workers ${NUM_WORKERS} --checkpoint ${CHECKPOINT_ROOT} --distributed_timeout_hours 0.25 --seed ${SEED} --backbone resnet50 --no_pretrained --init_portion 1.0 --max_portion 1.0 --portion_step 0.0 --gmm_components ${GMM_COMPONENTS} --gmm_min_features ${GMM_MIN_FEATURES} --gmm_max_features ${GMM_MAX_FEATURES} --gmm_batch_size ${GMM_BATCH_SIZE} --gmm_em_iters ${GMM_EM_ITERS} --cogamid_replay_max_per_class ${COGAMID_REPLAY_MAX_PER_CLASS} --opt_level O1 2>&1 | while IFS= read -r line;do log "$line";done

for STEP in 0 1;do
    MODEL_PATH=${CHECKPOINT_DIR}/${DATASET}_${TASK}_${NAME}_step_${STEP}.pth
    GMM_PATH=${CHECKPOINT_DIR}/${DATASET}_${TASK}_${NAME}_gmm_pool_step_${STEP}.pth
    test -s ${MODEL_PATH}
    test -s ${GMM_PATH}
    CHECK_OUTPUT=$(python3 -c "import torch;pool=torch.load('${GMM_PATH}',map_location='cpu');assert len(pool)>0,'empty GMM pool';assert any(int(payload.get('step',-1))==${STEP} and not payload.get('stale',False) for payload in pool.values()),'step ${STEP} has no fresh GMM';print('step ${STEP} artifacts OK, GMM classes:',sorted(pool))")
    log "${CHECK_OUTPUT}"
done

grep -q 'collect lightweight class coverage from 3 clients' ${LOG_FILE}
grep -q 'GMM client coverage: new' ${LOG_FILE}
grep -q 'select 2 clients only for GMM construction' ${LOG_FILE}

end=`date +%s`
runtime=$((end-start))
log "Coverage-aware full-flow smoke test PASSED in ${runtime}s"
