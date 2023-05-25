
GPU_ID=0
WEIGHTS=./../float/yolov5l.pt

cd code
CUDA_VISIBLE_DEVICES=${GPU_ID} python val.py --data data/coco.yaml --img 640 --conf 0.001 --iou 0.65 --weights ${WEIGHTS} --device ${GPU_ID}
