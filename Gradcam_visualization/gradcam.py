from cam_utils import *
from Model_training.model import DeepDOC
from Model_training.utils import get_dataset


def fetch_gradcam():
    model_name=''
    data_path=''
    with open(f'../cons/{model_name}.pth', 'rb') as f:
        f = torch.load(f)

    model = DeepDOC(model_depth=50)
    model.load_state_dict(state_dict=f)

    for i in get_dataset:
        img, y, p_id = '0'+str(i) if len(str(i)) == 1 else str(i)
        target_layers = [model.resnet.layer1]
        cam = GradCAM(model, target_layers, None)
        gc = cam(img, y)
        gc = gc[0][0]
        gc = resize3d(gc, img.shape)
        np.save('./BCIgc/p{}_gc.npy'.format(p_id), gc)