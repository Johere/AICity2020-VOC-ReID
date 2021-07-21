#------------------------- test with VOC-----------------------------------------
python tools/test.py --config_file='configs/veri.yml' \
MODEL.DEVICE_ID "('3')" \
MODEL.NAME "('resnet50_ibn_a')" \
MODEL.MODEL_TYPE "baseline_reduce" \
MODEL.POOLING_METHOD 'GeM' \
DATASETS.TRAIN "('veri',)" \
DATASETS.TEST "('veri',)" \
DATASETS.ROOT_DIR "('${HOME}/disk1/datasets')" \
MODEL.PRETRAIN_CHOICE "('self')" \
INPUT.SIZE_TRAIN '([320, 320])' \
INPUT.SIZE_TEST '([320, 320])' \
TEST.USE_VOC False \
TEST.CAM_DIST_PATH '' \
TEST.ORI_DIST_PATH './output/veri/results_baseline_reduce/feat_distmat.npy' \
TEST.WEIGHT "('./output/veri/baseline_reduce/size320/best.pth')"