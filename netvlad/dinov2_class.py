import torch
from torchvision import transforms
from PIL import Image

class DinoDescriptor():
    def __init__(self, device="cuda", model_type="vitb14"):
        self.device = device

        # Load DINOv2 via Torch Hub
        self.dinov2_model = torch.hub.load(
            "facebookresearch/dinov2", f"dinov2_{model_type}"
        ).to(device)
        self.dinov2_model.eval()

        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])
        print(f"DINOv2-{model_type} ready")

    @torch.no_grad()
    def extract_descriptor(self, img_path, pooling="avg"):
        """
        Extract global image descriptor from DINOv2.
        pooling: 'cls' or 'avg' (average over patch tokens)
        """
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Forward pass
        feats = self.dinov2_model.forward_features(tensor)["x_norm_patchtokens"]
        # feats: (N, L, C)

        if pooling == "cls":
            desc = feats[:, 0, :]          # CLS token
        elif pooling == "avg":
            desc = feats.mean(dim=1)       # average pooling over patch tokens
        else:
            raise ValueError("pooling must be 'cls' or 'avg'")

        # L2 normalize
        desc = torch.nn.functional.normalize(desc, p=2, dim=1)
        return desc.cpu().numpy().squeeze()
