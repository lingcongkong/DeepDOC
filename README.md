# DeepDOC: a two-stage deep learning framework for discriminating minimally conscious state from unresponsive patients
![Framework](https://github.com/lingcongkong/DeepDOC/blob/main/workflow_DeepDOC_wu_00.png)
## Introduction
The implementation of: <br>
[**DeepDOC: a two-stage deep learning framework for discriminating minimally conscious state from unresponsive patients**](https://www.nature.com/articles/)
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
cd DeepDOC
pip install -r requirements.txt
```
### Repository Structure
Below are the main directories in the repository: 

- `Model_training/`: the model structure and train/validation loop
- `Gradcam_visualization/`: visualization the Grad-CAM result of the trained models on cerebral cortex surface/MNI space
- `Shap/`: implementation of machine learning models and SHAP value calculation

### Training
```
python Model_training/train.py --stage 1 --data_path ./data/ --batch_size 32 --class_num 2 --epochs 250 --lr 0.0001 
```


## Citation
If you find this repository useful or use our dataset, please consider citing our work after publishment:

```DeepDOC: a two-stage deep learning framework for discriminating minimally conscious state from unresponsive patients```
