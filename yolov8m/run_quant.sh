
GPU_ID=0

ROOT_DIR=${PWD}
WEIGHTS=./../float/yolov8m.pt
cd code

echo "[Calib mode]"
CUDA_VISIBLE_DEVICES=${GPU_ID} yolo detect val data="datasets/coco.yaml" model=${WEIGHTS} \
    nndct_quant=True quant_mode=calib --nndct_convert_sigmoid_to_hsigmoid --nndct_convert_silu_to_hswish
echo "[Test mode] testing"
CUDA_VISIBLE_DEVICES=${GPU_ID} yolo detect val data="datasets/coco.yaml" model=${WEIGHTS} \
    nndct_quant=True quant_mode=test --nndct_convert_sigmoid_to_hsigmoid --nndct_convert_silu_to_hswish
