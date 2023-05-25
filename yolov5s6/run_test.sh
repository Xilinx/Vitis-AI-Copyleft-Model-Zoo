
GPU_ID=0

# yolov5n weight
WEIGHTS=./../float/yolov5n_float.pt  
# # yolov5n weight
# WEIGHTS=./../float/yolov5l_float.pt
# # yolov5n weight
# WEIGHTS=./../float/yolov5s6_float.pt
cd code

echo "[Test mode]"
# img-size 640 for yolov5n and yolov5l
CUDA_VISIBLE_DEVICES=${GPU_ID} python val.py --data data/coco.yaml --img 640 --conf 0.001 --iou 0.65 --weights ${WEIGHTS} --device ${GPU_ID}
# img-size 640 for yolov5n and yolov5l
# CUDA_VISIBLE_DEVICES=${GPU_ID} python val.py --data data/coco.yaml --img 1280 --conf 0.001 --iou 0.65 --weights ${WEIGHTS} --device ${GPU_ID}