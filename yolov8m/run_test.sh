
GPU_ID=0

ROOT_DIR=${PWD}

WEIGHTS=./../float/yolov8m.pt
cd code

echo "[Test mode]"
CUDA_VISIBLE_DEVICES=${GPU_ID} yolo detect val data="datasets/coco.yaml" model=${WEIGHTS}
