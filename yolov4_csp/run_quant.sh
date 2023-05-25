


GPU_ID=0

ROOT_DIR=${PWD}

WEIGHTS=./../quantized/yolov4_qat.pt
QUANT_DIR=quantize_result
cd code

echo "[Calib mode]"
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 test.py --cfg models/yolov4-csp-sppf.cfg --data data/coco.yaml --img 640 --conf 0.001 --iou 0.65 --weights ${WEIGHTS} --device ${GPU_ID} \
    --nndct_quant --quant_mode calib --output_path ${QUANT_DIR}

echo "[Test mode] testing"
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 test.py --cfg models/yolov4-csp-sppf.cfg --data data/coco.yaml --img 640 --conf 0.001 --iou 0.65 --weights ${WEIGHTS} --device ${GPU_ID} \
    --nndct_quant --quant_mode test --output_path ${QUANT_DIR}
