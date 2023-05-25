
GPU_ID=0
WEIGHTS=./../float/yolov8m.pt
BATCH=2
EPOCH=50
PORT=9050
cd code

CUDA_VISIBLE_DEVICES=${GPU_ID} yolo detect train data="datasets/coco.yaml" model=${WEIGHTS} pretrained=True \
    epochs=${EPOCH} batch=${BATCH} device=${GPU_ID}