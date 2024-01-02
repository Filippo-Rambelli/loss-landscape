import os
import torch, torchvision
import cifar10.models.vgg as vgg
import cifar10.models.resnet as resnet
import cifar10.models.densenet as densenet

# map between model name and function
models = {
    'resnet18'              : resnet.ResNet18,
}

def load(model_name, model_file=None, data_parallel=False):
    net = models[model_name]()
    if data_parallel: # the model is saved in data parallel mode
        net = torch.nn.DataParallel(net)

    if model_file:
        assert os.path.exists(model_file), model_file + " does not exist."
        stored = torch.load(model_file, map_location=lambda storage, loc: storage)
        if 'state_dict' in stored.keys():
            state_dict = stored['state_dict']
        else:
            state_dict = stored
        # Remove "model." prefix from state_dict keys
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        net.load_state_dict(state_dict)

    if data_parallel: # convert the model back to the single GPU version
        net = net.module

    net.eval()
    return net
