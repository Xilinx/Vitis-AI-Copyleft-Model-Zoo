GPU_ID=0
BATCH=2
# cd code

# img-size 640 for yolov5 nano
CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py --data data/coco.yaml --weights "" --batch-size ${BATCH} --device ${GPU_ID} --img 640 --cfg models/yolov5n_nndct.yaml
# img-size 640 for yolov5 large
# CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py --data data/coco.yaml --weights "" --batch-size ${BATCH} --device ${GPU_ID} --img 640 --cfg models/yolov5l_nndct.yaml
# img-size 1280 for yolov5s6
# CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py --data data/coco.yaml --weights "" --batch-size ${BATCH} --device ${GPU_ID} --img 1280 --cfg models/yolov5s6_nndct.yaml