GPU_ID=6
ROOT_DIR=${PWD}
WEIGHTS=/group/modelzoo/internal-cooperation-models/pytorch/yolov4-csp/yolov4_float.pt
LOCALPATH=./model.pt

cp ${WEIGHTS} ${LOCALPATH}

# --img 1280 for yolov5s6
echo "[Test mode] dumping"
CUDA_VISIBLE_DEVICES=${GPU_ID} python test.py --cfg models/yolov4-csp-sppf.cfg --data data/coco.yaml --img 640 --conf 0.001 --iou 0.65 --weight ${WEIGHTS} --device 0 \
                                     --nndct_quant --quant_mode test --dump_onnx --batch-size 1 --nndct_equalization False --nndct_param_corr=False --output_path ${ROOT_DIR}
