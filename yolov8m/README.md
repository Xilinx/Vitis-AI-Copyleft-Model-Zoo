## YOLOv8m

### Hardware friendly changes
- Change the activation operation from SiLU to HardSiLU in quantization

### Prepare
#### Prepare the environment
```markdown
bash env_setup.sh
```
#### Prepare the dataset
##### Put coco2017 dataset under the ./code/datasets directory, dataset directory structure like:
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
| YOLOv8m           | 640*640    | 50.0%       | 48.7%       | 78.9G  |


### Model Weights Download: [YOLOv8m](https://www.xilinx.com/bin/public/openDownload?filename=pt_yolov8m_3.5.zip)

### **Pre-Compiled Models For Hardware Acceleration Platform**

- **Board: VEK280**
  - Type: xmodel
  - DownLoad Link: https://www.xilinx.com/bin/public/openDownload?filename=
  - Checksum:
- **Board: V70**
  - Type: xmodel
  - DownLoad Link: https://www.xilinx.com/bin/public/openDownload?filename=yolov8m_pt-v70-DPUCV2DX8G-r3.5.0.tar.gz
  - Checksum: 61c2a1d4364baac393a20a2d9687b168
