GPU_ID=0
QUANT_DIR=quantize_result

# weight for different models, choice one of them in yolov5 series
WEIGHTS=/group/modelzoo/internal-cooperation-models/pytorch/yolov5_v2/yolov5n_float.pt
# WEIGHTS=/group/modelzoo/internal-cooperation-models/pytorch/yolov5_v2/yolov5l_float.pt
# WEIGHTS=/group/modelzoo/internal-cooperation-models/pytorch/yolov5_v2/yolov5s6_float.pt
LOCALPATH=./model.pt

cp ${WEIGHTS} ${LOCALPATH}

# --img 1280 for yolov5s6
echo "[Calib mode]"
CUDA_VISIBLE_DEVICES=${GPU_ID} python val.py --data data/coco.yaml --img 640 --conf 0.001 --iou 0.65 --weight ${LOCALPATH} --device ${GPU_ID} --nndct_quant --quant_mode calib --output_path ${QUANT_DIR}

# --img 1280 for yolov5s6
echo "[Test mode] testing"
CUDA_VISIBLE_DEVICES=${GPU_ID} python val.py --data data/coco.yaml --img 640 --conf 0.001 --iou 0.65 --weight ${LOCALPATH} --device ${GPU_ID} --nndct_quant --quant_mode test --output_path ${QUANT_DIR}
