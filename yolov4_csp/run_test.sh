
GPU_ID=0

ROOT_DIR=${PWD}

WEIGHTS=./../float/yolov4_float.pt
cd code

echo "[Test mode]"
CUDA_VISIBLE_DEVICES=${GPU_ID} python test.py --cfg models/yolov4-csp-sppf.cfg --data data/coco.yaml --img 640 --conf 0.001 --iou 0.65 --weights ${WEIGHTS} --device ${GPU_ID}