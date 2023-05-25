
GPU_ID=0
WEIGHTS=./../float/yolov5l.pt

cd code
echo "[Calib mode]"
CUDA_VISIBLE_DEVICES=${GPU_ID} python val.py --data data/coco.yaml --img 640 --conf 0.001 --iou 0.65 --weights ${WEIGHTS} --device ${GPU_ID} --nndct_quant --quant_mode calib \
                                            --nndct_convert_sigmoid_to_hsigmoid --nndct_convert_silu_to_hswish

echo "[Test mode] testing"
CUDA_VISIBLE_DEVICES=${GPU_ID} python val.py --data data/coco.yaml --img 640 --conf 0.001 --iou 0.65 --weights ${WEIGHTS} --device ${GPU_ID} --nndct_quant --quant_mode test \
                                            --nndct_convert_sigmoid_to_hsigmoid --nndct_convert_silu_to_hswish
