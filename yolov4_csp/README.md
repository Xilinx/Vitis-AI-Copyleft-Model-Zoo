# YOLOv4
This repository is based on WongKinYiu/ScaledYOLOv4: https://github.com/WongKinYiu/ScaledYOLOv4/tree/yolov4-csp. 

##  Hardware friendly changes
Here is a brief description of changes that were made to get yolov4-csp-leaky from yolov4-csp:

- Mish activation is not well-supported in embedded devices. it's not quantization friendly as well. Hence, we change the activation operation from Swish to LeakyReLU(negative_slope=26./256), to be sure that the negative_slope of LeakyReLU must be set to 26./256

- (**Optional**) SPP module with maxpool(k=13, s=1), maxpool(k=9,s=1) and maxpool(k=5,s=1) are replaced with sppf module including various combinations of maxpool(k=5,s=1) for better used in embedded devices. This change will cause no difference to the model in floating-point.

  - maxpool(k=5, s=1) -> replaced with maxpool(k=5,s=1)
  - maxpool(k=9, s=1) -> replaced with maxpool(k=5,s=1)
  - maxpool(k=13, s=1) -> replaced with maxpool(k=5,s=1) as shown below:
  <p align="left"><img width="800" src="./image2022-10-26_14-39-45.png"></p>


## Performance

|Dataset |Model Name                      |Input Size |GFLOPS    |Official FLOAT AP[0.5:0.95]%|FLOAT AP[0.5:0.95]%|Quant AP[0.5:0.95]%|
|--------|------------------------------- |-----------|----------|----------------------------|-------------------|-------------------|
|COCO    |YOLOv4-csp                      |640x640    |**122**   |47.8                        |   47.0            | 46.3              |


## Prepare

### Prepare the environment

#### For vitis-ai docker user
```bash
pip install -r requirements.txt
```

#### Others
```bash
conda create -n yolov4 python=3.7
conda activate yolov4
pip install -r requirements.txt
# you also need to install nndct and xir(optional for dump xmodel)
```

### Prepare the dataset

1. Dataset description

    - download COCO2017 dataset.(refer to this repo https://github.com/WongKinYiu/ScaledYOLOv4/tree/yolov4-csp)

2. Dataset diretory structure
    - Put coco2017 dataset under the ./datasets directory, dataset directory structure like:
        ```markdown
        + datasets/
            + coco/
                + labels/
                + annotations/
                + images/
                + test-dev2017.txt 
                + train2017.txt
                + val2017.txt
        ```

3. ## coco2017 dataset
    - put the path to coco2017 in ./code/data/coco.yaml

## Train/Eval

### Training 
```
cd code
bash run_train.sh
```

### Validation
```bash
cd code
bash run_test.sh
```

### Run quantization
```bash
cd code
bash run_quant.sh
```

## post process(if need)
When doing the PTQ for YOLO V4-CSP, these codes below (in file:./models/models.py) implementing the yolo head's post-process should be excluded from the PTQ's Darknet forward function, which means that our post-training quantized model doesn't include the post process part in the original Darknet forward function of the YOLOV4-CSP.
```
# Post-Process in YOLO Haed
bs, _, ny, nx = p.shape  # bs, 255, 13, 13
if (self.nx, self.ny) != (nx, ny):
    self.create_grids((nx, ny), p.device)
 
# p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction
if self.training:
    return p
else:
    io = p.sigmoid()
    io[..., :2] = (io[..., :2] * 2. - 0.5 + self.grid)
    io[..., 2:4] = (io[..., 2:4] * 2) ** 2 * self.anchor_wh
    io[..., :4] *= self.stride
    return (io.view(bs, -1, self.no), p)
```

### GPU Model Weights Download: [YOLOv4_CSP](https://www.xilinx.com/bin/public/openDownload?filename=pt_yolov4csp_3.5.zip)

## **Pre-Compiled Models For Hardware Acceleration Platform**

- **Board: VEK280**
  - Type: xmodel
  - DownLoad Link: https://www.xilinx.com/bin/public/openDownload?filename=yolov4_csp_pt-vek280-r3.5.0.tar.gz
  - Checksum:3f010f3bff8489be845914ab5ee4f899
- **Board: V70**
  - Type: xmodel
  - DownLoad Link: https://www.xilinx.com/bin/public/openDownload?filename=yolov4_csp_pt-v70-DPUCV2DX8G-r3.5.0.tar.gz
  - Checksum: 56e1e8ced90b3efa7b80e0b6d4851c66

## **References**

[1] [WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4/tree/yolov4-csp) <br>
[2] Chien-Yao Wang, Hong-Yuan Mark Liao, Yueh-Hua Wu, Ping-Yang Chen, Jun-Wei Hsieh, and I-Hau Yeh. [CSPNet: A new backbone that can enhance learning capability of
cnn](https://arxiv.org/abs/1911.11929). Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshop (CVPR Workshop),2020. <br>
