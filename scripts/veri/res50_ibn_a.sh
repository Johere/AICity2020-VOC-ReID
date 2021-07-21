python tools/train.py --config_file='configs/veri.yml' \
MODEL.DEVICE_ID "('6')" \
MODEL.MODEL_TYPE "baseline" \
MODEL.NAME "('resnet50_ibn_a')" \
MODEL.PRETRAIN_PATH "('${HOME}/.cache/torch/checkpoints/resnet50_ibn_a-d9d0bb7b.pth')" \
MODEL.ID_LOSS_TYPE 'circle' \
SOLVER.COSINE_MARGIN 0.35 \
SOLVER.COSINE_SCALE 64 \
MODEL.POOLING_METHOD 'GeM' \
SOLVER.LR_SCHEDULER 'cosine' \
DATALOADER.NUM_INSTANCE 16 \
SOLVER.FREEZE_BASE_EPOCHS 2 \
SOLVER.MAX_EPOCHS 40 \
INPUT.SIZE_TRAIN '([320, 320])' \
INPUT.SIZE_TEST '([320, 320])' \
DATASETS.TRAIN "('veri',)" \
DATASETS.TEST "('veri',)" \
DATASETS.ROOT_DIR "('${HOME}/disk1/datasets')" \
OUTPUT_DIR "('./output/veri/debug/size320')"
#DATASETS.ROOT_DIR "('/home/linjiaojiao/disk1/datasets')" \