## YOLOv6 

### Hardware friendly changes
- Change the activation operation from Swish to ReLU
- When deployed, replace the elementwise multiply operation between scale parameter alpha and feature in the BottleRep module by depthwise convolution

### Prepare

#### Prepare the environment

##### For vitis-ai docker user
```bash
conda activate vitis-ai-pytorch
pip install -r yolov6/requirements.txt
```

##### Others
```bash
conda create -n yolov5 python=3.7
conda activate yolov5
pip install -r yolov5/requirements.txt
# you also need to install nndct and xir(optional for dump xmodel)
cd torch_rewriters
python setup.py install
```

#### Prepare the dataset
Put coco2017 dataset under the ./coco directory, dataset directory structure like:
```markdown
+ coco/
    + labels/
    + annotations/
    + images/
    + test-dev2017.txt 
    + train2017.txt
    + val2017.txt
```

### For yolov6m-opt Train/Eval

#### First search the scale of RepoptOptimizer

```bash
cd yolov6_nndct/
python -m torch.distributed.launch --nproc_per_node 8 tools_nndct/train.py --batch 256 --conf configs/repopt/yolov6m_hs.py --data data/coco.yaml --device 0,1,2,3,4,5,6,7 --name yolov6m_hs
```

#### Train with RepoptOptimizer
```bash
cd yolov6_nndct/
python -m torch.distributed.launch --nproc_per_node 7 tools_nndct/train.py --batch 256 --conf configs/repopt/yolov6m_opt.py --data data/coco.yaml --device 1,2,3,4,5,6,7 --name yolov6m_opt
```

#### Train with distillation
```bash
cd yolov6_nndct/
python -m torch.distributed.launch --nproc_per_node 7 tools_nndct/train.py --batch 256 --conf configs/repopt/yolov6m_opt.py --data data/coco.yaml --device 1,2,3,4,5,6,7 --name yolov6m_opt_distill --distill --teacher_model_path runs/train/yolov6m_opt/weights/best_ckpt.pt
```

#### Run Post-training quantization
```bash
cd yolov6_nndct/
# run calibration & test & dump xmodel
python tools_nndct/quantization/export_nndct.py --conf configs/repopt/yolov6m_opt.py --weights runs/train/yolov6m_opt_distill/weights/best_ckpt.pt --device 0 --batch-size 1 --calib-batch-number 1000
```

#### Quantization aware training 
```bash
cd yolov6_nndct/
python tools_nndct/train.py --name yolov6m_opt_distill_qat --conf configs/repopt/yolov6m_opt_qat.py --device 0 --quant --batch-size 32 --teacher_model_path runs/train/yolov6m_opt_distill/weights/best_ckpt.pt --distill --distill_feat --epochs 24 --workers 32
```

### Run Quantization aware training model quantization
```bash
cd yolov6_nndct/
# run calibration & test & dump xmodel
python tools_nndct/quantization/export_nndct.py --conf configs/repopt/yolov6m_opt_qat.py --weights runs/train/yolov6m_opt_distill_qat/weights/best_ckpt.pt --device 0 --batch-size 1 --qat
```

### Performance

| Model | Input Size | mAP | FLOPs |
|-------|------------|--------------|-------|
| YOLO6m-opt | 640 | 48.8% | 41.19GMac |
| YOLO6m-opt QUANT| 640 | 46.2% | - |
| YOLO6m-opt QAT| 640 | 48.1% | - |