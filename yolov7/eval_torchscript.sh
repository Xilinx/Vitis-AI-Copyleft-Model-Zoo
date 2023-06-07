cd yolov7
ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 python test_onnx.py --data data/coco.yaml --img 640 --batch 1 --conf 0.001 --iou 0.65 --weights yolov7.pt --onnx-file yolov7.torchscript.pt --name yolov7_640_val --onnx-runtime torchscript --device cpu
cd ../
