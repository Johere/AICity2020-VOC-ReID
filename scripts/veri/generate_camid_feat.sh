#------------------------- generate orientation-camera matrix--------------------
python tools/test.py --config_file='configs/aicity20.yml' \
MODEL.DEVICE_ID "('6')" \
MODEL.NAME "('resnet50_ibn_a')" \
MODEL.MODEL_TYPE "baseline" \
MODEL.POOLING_METHOD 'GeM' \
DATASETS.TRAIN "('veri',)" \
DATASETS.TEST "('veri',)" \
DATASETS.ROOT_DIR "('${HOME}/disk1/datasets')" \
MODEL.PRETRAIN_CHOICE "('self')" \
TEST.WRITE_RESULT True \
TEST.WEIGHT "('./pretrained_ckpt/ReCamID/best.pth')"

python ./tools/aicity20/compute_distmat_from_feats.py --src_dir ./pretrained_ckpt/ReCamID

echo 'Done'
