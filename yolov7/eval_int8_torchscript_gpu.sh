cd yolov7
python test_onnx.py --data data/coco.yaml --img 640 --batch 1 --conf 0.001 --iou 0.65 --weights yolov7.pt --onnx-file nndct/Model_int.pt --name yolov7_640_val --onnx-runtime torchscript --device 5 --precision int8
cd ../
