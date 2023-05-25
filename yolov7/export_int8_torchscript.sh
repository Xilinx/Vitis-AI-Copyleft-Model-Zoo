# python export.py --weights yolov7.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
set -e
cd yolov7/
echo "run calib"
python test_nndct.py --data data/coco.yaml --img 640 --batch 1 --conf 0.001 --iou 0.65 --device 1 --weights yolov7.pt --name yolov7_640_int8 --quant_mode calib
echo "run export"
python test_nndct.py --data data/coco.yaml --img 640 --batch 1 --conf 0.001 --iou 0.65 --device 1 --weights yolov7.pt --name yolov7_640_int8 --quant_mode test --dump_model
cd ../