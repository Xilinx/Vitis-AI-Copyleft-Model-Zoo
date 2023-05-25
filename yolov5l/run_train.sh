
GPU_ID=0
WEIGHTS=./../float/yolov5l.pt
BATCH=2
EPOCH=50
PORT=9050
CFG=models/yolov5l.yaml

cd code

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 train.py --data data/coco.yaml --cfg ${CFG} --batch-size ${BATCH} --epochs ${EPOCH} --weights ${WEIGHTS}