import torch
import models


def load_mobilevit_weights(model_path):
    # Create an instance of the MobileViT model
    # XXS: 1.3M 、 XS: 2.3M 、 S: 5.6M
    net = models.MobileViT_S()

    # Load the PyTorch state_dict
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))['state_dict']

    # Since there is a problem in the names of layers, we will change the keys to meet the MobileViT model architecture
    for key in list(state_dict.keys()):
        state_dict[key.replace('module.', '')] = state_dict.pop(key)

    # Once the keys are fixed, we can modify the parameters of MobileViT
    net.load_state_dict(state_dict)

    return net


if __name__ == '__main__':
    img = torch.randn(1, 3, 256, 256) # target image
    net = load_mobilevit_weights("MobileViT_S_model_best.pth.tar")

    print("MobileViT-S params: ", sum(p.numel() for p in net.parameters()))
    print(f"Output shape: {net(img).shape}")