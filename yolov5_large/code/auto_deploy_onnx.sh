GPU_ID=3

ROOT_DIR=${PWD}
#ORIG_PYTHON_PATH=${PYTHON_PATH}
#QUANT_DIR=quantize_result

# weight for different models, choice one of them in yolov5 series
WEIGHTS=./../float/yolov5n_float.pt
#WEIGHTS=./../float/yolov5l_float.pt
#WEIGHTS=./../float/yolov5s6_float.pt
LOCALPATH=./model.pt

#cp ${WEIGHTS} ${LOCALPATH}

# --img 1280 for yolov5s6
echo "[Test mode] dumping"
CUDA_VISIBLE_DEVICES=${GPU_ID} python val.py --data data/coco.yaml --img 640 --conf 0.001 --iou 0.65 --weight ${LOCALPATH} --device 0 --nndct_quant --quant_mode test \
                                     --dump_onnx --batch-size 1 --nndct_equalization False --nndct_param_corr=False --output_path ${ROOT_DIR}
# export PYTHON_PATH=${ORIG_PYTHON_PATH}
