# Two-stage deep learning framework enables accurate diagnosis of disorders of consciousness from resting state fMRI images
![Framework](https://github.com/lingcongkong/DeepDOC/blob/main/Fig1_Framework.png)
## Introduction
The implementation of: <br>
[**Two-stage deep learning framework enables accurate diagnosis of disorders of consciousness from resting state fMRI images**](https://www.nature.com/articles/)
## Requirements
- python 3.9
- pytorch 1.8.1
- torchvision 0.11.2
- sklearn 1.0.2
- matplotlib 3.5.2
- nibabel 4.0.1
- nilearn 0.9.1
- shap 0.41.0
- opencv-python 4.5.3
- pandas 1.1.5


## Setup
### Installation
Clone the repo and install required packages:
```
git clone https://github.com/lingcongkong/DeepDOC.git
cd HoVerTrans
pip install -r requirements.txt
```

### Training
```
python train.py --data_path ./data/GDPH&SYSUCC/img --csv_path ./data/GDPH&SYSUCC/label.csv --batch_size 32 --class_num 2 --epochs 250 --lr 0.0001 
```
## Citation
If you find this repository useful or use our dataset, please consider citing our work:

```Two-stage deep learning framework enables accurate diagnosis of disorders of consciousness from resting state fMRI images```
