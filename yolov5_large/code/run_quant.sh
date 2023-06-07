
GPU_ID=0

# yolov5 nano weight
# WEIGHTS=./../float/yolov5n_float.pt  
# yolov5 large weight
WEIGHTS=./../float/yolov5l_float.pt
# yolov5s6 weight
# WEIGHTS=./../float/yolov5s6_float.pt
QUANT_DIR=quantize_result
# cd code

echo "[Calib mode]"
# img-size 640 for yolov5 nano and yolov5 large
CUDA_VISIBLE_DEVICES=${GPU_ID} python val.py --data data/coco.yaml --img 640 --conf 0.001 --iou 0.65 --weights ${WEIGHTS} --device ${GPU_ID} --nndct_quant --quant_mode calib --output_path ${QUANT_DIR}
# img-size 1280 for yolov5s6
# CUDA_VISIBLE_DEVICES=${GPU_ID} python val.py --data data/coco.yaml --img 1280 --conf 0.001 --iou 0.65 --weights ${WEIGHTS} --device ${GPU_ID} --nndct_quant --quant_mode calib --output_path ${QUANT_DIR}

echo "[Test mode] testing"
# img-size 640 for yolov5 nano and yolov5 large
CUDA_VISIBLE_DEVICES=${GPU_ID} python val.py --data data/coco.yaml --img 640 --conf 0.001 --iou 0.65 --weights ${WEIGHTS} --device ${GPU_ID} --nndct_quant --quant_mode test --output_path ${QUANT_DIR}
# img-size 1280 for yolov5s6
# CUDA_VISIBLE_DEVICES=${GPU_ID} python val.py --data data/coco.yaml --img 1280 --conf 0.001 --iou 0.65 --weights ${WEIGHTS} --device ${GPU_ID} --nndct_quant --quant_mode test --output_path ${QUANT_DIR}