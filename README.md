# Precise Detection of Awareness in Disorders of Consciousness Using Deep Learning Framework
![Framework](https://github.com/lingcongkong/DeepDOC/blob/main/Fig1.jpeg)
## Introduction
The implementation of: <br>
[**Precise Detection of Awareness in Disorders of Consciousness Using Deep Learning Framework**](https://www.nature.com/articles/)
## Requirements
- python 3.9
- pytorch 1.8.1
- torchvision 0.11.2
- sklearn 1.0.2
- matplotlib 3.5.2
- nibabel 4.0.1
- nilearn 0.9.1
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
python Model_training/train.py 
```


## Citation
If you find this repository useful or use our dataset, please consider citing our work after publishment:

```Precise Detection of Awareness in Disorders of Consciousness Using Deep Learning Framework```
