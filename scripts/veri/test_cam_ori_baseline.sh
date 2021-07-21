#------------------------- test with VOC-----------------------------------------
python tools/test.py --config_file='configs/veri.yml' \
MODEL.DEVICE_ID "('6')" \
MODEL.NAME "('resnet50_ibn_a')" \
MODEL.MODEL_TYPE "baseline" \
MODEL.POOLING_METHOD 'GeM' \
DATASETS.TRAIN "('veri',)" \
DATASETS.TEST "('veri',)" \
DATASETS.ROOT_DIR "('${HOME}/disk1/datasets')" \
MODEL.PRETRAIN_CHOICE "('self')" \
INPUT.SIZE_TRAIN '([256, 256])' \
INPUT.SIZE_TEST '([256, 256])' \
TEST.FLIP_TEST False \
TEST.DO_RERANK False \
TEST.RERANK_PARAM "([50, 15, 0.5])" \
OUTPUT_DIR "./output/veri/test" \
TEST.USE_VOC False \
TEST.CAM_DIST_PATH '' \
TEST.ORI_DIST_PATH './pretrained_ckpt/ReOriID/veri/output/feat_distmat.npy' \
TEST.WEIGHT "('./pretrained_ckpt/ReOriID/best.pth')"
#TEST.WEIGHT "('./pretrained_ckpt/res50_ibn_a/size320/best.pth')"
#TEST.USE_VOC False \
#TEST.RERANK_PARAM "([50, 15, 0.5])" \
#TEST.CAM_DIST_PATH '' \
#TEST.ORI_DIST_PATH './pretrained_ckpt/ReOriID/veri/output/feat_distmat.npy' \
#TEST.CAM_DIST_PATH './pretrained_ckpt/ReCamID/veri/output/feat_distmat.npy' \