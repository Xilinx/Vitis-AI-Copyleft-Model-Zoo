## YOLOv5l 

### Hardware friendly changes
- Change the activation operation from SiLU to HardSiLU in quantization

### Model Weights Download: [YOLOv5l](https://www.xilinx.com/bin/public/openDownload?filename=pt_yolov5l_3.5.zip)

### Prepare
#### Prepare the environment
```markdown
bash env_setup.sh
```
#### Prepare the dataset
##### Put coco2017 dataset under the ./code/data directory, dataset directory structure like:
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
##### Modify the path of coco in coco.yaml to your custom dataset
### Train/Validation
#### Train
```markdown
# for float training
bash run_train.sh 
```
#### Test and Quant
##### Test
```markdown
bash run_test.sh
```
##### Quant
```markdown
bash run_quant.sh
```
### Performance
| Model             | Input Size | Float mAP   | Quant mAP   | FLOPs  |
|-------------------|------------|-------------|-------------|--------|
| YOLOv5l           | 640\*640   | 49.0%       | 45.7%       | 78.9G  |

### Model Weights Download: [YOLOv5l](https://www.xilinx.com/bin/public/openDownload?filename= )

## **Pre-Compiled Models For Hardware Acceleration Platform**

- **Board: VEK280**
  - Type: xmodel
  - DownLoad Link: https://www.xilinx.com/bin/public/openDownload?filename=yolov5l_pt-vek280-r3.5.0.tar.gz
  - Checksum:43197d598c67b2c66c99f3cbaa18e1b6
- **Board: V70**
  - Type: xmodel
  - DownLoad Link: https://www.xilinx.com/bin/public/openDownload?filename=
  - Checksum: 