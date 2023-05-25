
GPU_ID=0
BATCH=8
cd code

CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py --data data/coco.yaml --cfg models/yolov4-csp-sppf.cfg --weights "" --img 640 --batch-size ${BATCH} --device ${GPU_ID}