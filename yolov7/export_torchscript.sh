# python export.py --weights yolov7.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
cd yolov7/
python export.py --weights yolov7.pt --grid --end2end --simplify --topk-all 10000 --iou-thres 0.65 --conf-thres 0.001 --img-size 640 640 --max-wh 640
cd ../