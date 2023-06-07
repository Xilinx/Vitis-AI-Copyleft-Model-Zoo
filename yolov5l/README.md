## YOLOv5l 

### Hardware friendly changes
- Change the activation operation from SiLU to HardSiLU in quantization

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