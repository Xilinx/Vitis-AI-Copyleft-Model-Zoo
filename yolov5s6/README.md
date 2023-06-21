# YOLOv5
This repository is based on ultralytics/yolov5: https://github.com/ultralytics/yolov5. 

##  Hardware friendly changes
Here is a brief description of changes that were made to get hardware friendly YOLOv5 from official code:

- Change the activation operation from Swish to LeakyReLU(negative_slope=26./256), to be sure that the negative_slope of LeakyReLU must be set to 26./256


## Performance

|Dataset |Model Name                      |Input Size |GFLOPS    |Official FLOAT AP[0.5:0.95]%|FLOAT AP[0.5:0.95]%|Quant AP[0.5:0.95]%|
|--------|------------------------------- |-----------|----------|----------------------------|-------------------|-------------------|
|COCO    |YOLOv5 nano                     |640x640    |**4.5**   |   28.0                     | 27.0              | 26.2              |
|COCO    |YOLOv5 large                    |640x640    |**109.1** |   49.0                     | 47.2              | 45.5              |
|COCO    |YOLOv5s6                        |1280x1280  |**16.8**  |   44.8                     | 43.6              | 42.0              |


### Model Weights Download: [YOLOv5s6](https://www.xilinx.com/bin/public/openDownload?filename=pt_yolov5s6_3.5.zip)


## Prepare

### Prepare the environment

#### For vitis-ai docker user
```bash
pip install -r requirements.txt
```

#### Others
```bash
conda create -n yolov5 python=3.7
conda activate yolov5
pip install -r requirements.txt
# you also need to install nndct and xir(optional for dump xmodel)
```

### Prepare the dataset

1. Dataset description

    - download COCO2017 dataset.(refer to this repo https://github.com/ultralytics/yolov5)

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

3. Prepare dataset required by model:

    ```markdown
    cd code/data/scripts/ 
    bash get_coco.sh
    ```

## Train/Eval

### Training 
```bash
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

## **Pre-Compiled Models For Hardware Acceleration Platform**

- **Board: VEK280**
  - Type: xmodel
  - DownLoad Link: https://www.xilinx.com/bin/public/openDownload?filename=
  - Checksum:
- **Board: V70**
  - Type: xmodel
  - DownLoad Link: https://www.xilinx.com/bin/public/openDownload?filename=yolov5s6_pt-v70-DPUCV2DX8G-r3.5.0.tar.gz
  - Checksum: 2bbd09088e80cb18e69876757958b1fa

## **References**

[1] [Official YOLOV5 repository](https://github.com/ultralytics/yolov5/) <br>
