import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# --- NetVLAD Layer ---
class NetVLAD(nn.Module):
    def __init__(self, num_clusters=64, dim=384, normalize_input=True):
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.normalize_input = normalize_input
        self.centroids = nn.Parameter(torch.zeros(num_clusters, dim))
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=1, bias=True)

    def forward(self, x):  # x: (N, C, H, W)
        N, C, H, W = x.shape
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)

        soft_assign = self.conv(x)
        soft_assign = soft_assign.view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=2)

        x_flatten = x.view(N, C, -1)
        vlad = torch.zeros([N, self.num_clusters, C], device=x.device)
        for k in range(self.num_clusters):
            residual = x_flatten - self.centroids[k:k+1].unsqueeze(-1)
            vlad[:, k, :] = torch.sum(soft_assign[:, k:k+1, :] * residual, dim=2)

        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.view(N, -1)
        vlad = F.normalize(vlad, p=2, dim=1)
        return vlad


# --- DINOv2 + NetVLAD wrapper ---
class DinoNetVLAD():
    def __init__(self, device="cuda", num_clusters=64):
        self.device = device

        # Load DINOv2 via Torch Hub
        # self.dinov2_model = torch.hub.load(
        #     "facebookresearch/dinov2", "dinov2_vits14"
        # ).to(device)
        self.dinov2_model = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitb14"
        ).to(device)

        self.dinov2_model.eval()

        # NetVLAD layer (DINOv2-ViT-S/14 outputs 384-dim tokens)
        # self.netvlad = NetVLAD(num_clusters=num_clusters, dim=384).to(device) # DINO vits
        self.netvlad = NetVLAD(num_clusters=num_clusters, dim=768).to(device)   # DINO vitb
        self.netvlad.eval()

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])
        print("DINOv2 + NetVLAD ready")

    @torch.no_grad()
    def extract_descriptor(self, img_path):
        # Load & preprocess
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Get patch tokens from DINOv2
        tokens = self.dinov2_model.forward_features(tensor)["x_norm_patchtokens"]
        # tokens: (N, L, C), L = num patches

        N, L, C = tokens.shape
        side = int(L ** 0.5)  # assume square patch grid
        feat_map = tokens.transpose(1, 2).reshape(N, C, side, side)

        # Feed into NetVLAD
        desc = self.netvlad(feat_map)
        return desc.cpu().numpy().squeeze()
