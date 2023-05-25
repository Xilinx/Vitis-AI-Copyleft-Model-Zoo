GPU_ID=0
QUANT_DIR=quantize_result

# weight for different models, choice one of them in yolov5 series
WEIGHTS=/group/modelzoo/internal-cooperation-models/pytorch/yolov5_v2/yolov5n_qat.pt
# WEIGHTS=/group/modelzoo/internal-cooperation-models/pytorch/yolov5_v2/yolov5l_qat.pt
# WEIGHTS=/group/modelzoo/internal-cooperation-models/pytorch/yolov5_v2/yolov5s6_qat.pt
LOCALPATH=./model.pt

cp ${WEIGHTS} ${LOCALPATH}

# --img 1280 for yolov5s6
echo "[Test mode] dumping"
CUDA_VISIBLE_DEVICES=${GPU_ID} python val.py --data data/coco.yaml --img 640 --conf 0.001 --iou 0.65 --weight ${LOCALPATH} --device 0 --nndct_quant --quant_mode test \
                                     --dump_xmodel --batch-size 1 --nndct_equalization False --nndct_param_corr=False --output_path ${QUANT_DIR}
