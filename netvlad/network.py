import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# --- NetVLAD Layer ---
class NetVLAD(nn.Module):
    def __init__(self, num_clusters=64, dim=512, normalize_input=True):
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.normalize_input = normalize_input

        self.centroids = nn.Parameter(torch.zeros(num_clusters, dim))
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1,1), bias=True)

    def forward(self, x):
        N, C, H, W = x.shape

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)

        soft_assign = self.conv(x)
        soft_assign = F.softmax(soft_assign.view(N, self.num_clusters, -1), dim=2)

        x_flatten = x.view(N, C, -1)
        vlad = torch.zeros([N, self.num_clusters, C], device=x.device)

        for k in range(self.num_clusters):
            residual = x_flatten - self.centroids[k:k+1].unsqueeze(-1)
            vlad[:, k, :] = torch.sum(soft_assign[:, k:k+1, :] * residual, dim=2)

        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.view(N, -1)
        vlad = F.normalize(vlad, p=2, dim=1)
        return vlad

# --- Backbone + NetVLAD with checkpoint loading ---
def get_netvlad_backbone(checkpoint_path="vgg16_netvlad_checkpoint/checkpoints/checkpoint.pth.tar",
                         num_clusters=64, map_location="cpu"):
    """
    Builds VGG16 + NetVLAD model and loads weights from a .pth.tar checkpoint.
    """
    # Backbone: VGG16 conv layers
    vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    backbone = nn.Sequential(*list(vgg.features.children())[:-2])  # drop last two layers
    model = nn.Sequential(backbone, NetVLAD(num_clusters=num_clusters, dim=512))

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=map_location)

    # Determine state dict
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    # Remove possible 'module.' prefix
    new_state = {}
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[len('module.'):]
        new_state[name] = v

    # Load into model
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print("NetVLAD checkpoint loaded from:", checkpoint_path)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model.eval()
    return model


