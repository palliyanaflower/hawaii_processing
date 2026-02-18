import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np

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

    @torch.no_grad()
    def extract_keypoint_descriptors(self, img_path, keypoints):
        """
        keypoints: numpy array of shape (N, 2)
                each row = (u, v) in ORIGINAL image coordinates

        Returns:
            descriptors: (N, C) numpy array
        """

        img = Image.open(img_path).convert("RGB")
        W_orig, H_orig = img.size

        tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Extract patch tokens
        feats = self.dinov2_model.forward_features(tensor)["x_norm_patchtokens"]
        # (1, L, C)

        B, L, C = feats.shape
        grid_size = int(np.sqrt(L))   # 37 for vitb14
        patch_tokens = feats.view(1, grid_size, grid_size, C)
        patch_tokens = patch_tokens.permute(0, 3, 1, 2)  # (1, C, H, W)

        # --- 3x3 smoothing (robustness boost) ---
        patch_tokens = F.avg_pool2d(patch_tokens, kernel_size=3, stride=1, padding=1)

        # --- Convert keypoints to normalized grid coords ---
        keypoints = np.asarray(keypoints)

        # resize to 518 space
        u_resized = keypoints[:, 0] * 518 / W_orig
        v_resized = keypoints[:, 1] * 518 / H_orig

        # convert to patch grid coordinates
        u_grid = u_resized / 14.0
        v_grid = v_resized / 14.0

        # normalize to [-1, 1] for grid_sample
        u_norm = (u_grid / (grid_size - 1)) * 2 - 1
        v_norm = (v_grid / (grid_size - 1)) * 2 - 1

        grid = torch.tensor(
            np.stack([u_norm, v_norm], axis=1),
            dtype=torch.float32,
            device=self.device
        )

        grid = grid.view(1, -1, 1, 2)  # (1, N, 1, 2)

        # --- Bilinear sampling ---
        sampled = F.grid_sample(
            patch_tokens,
            grid,
            mode="bilinear",
            align_corners=True
        )

        # (1, C, N, 1) -> (N, C)
        descriptors = sampled.squeeze(0).squeeze(-1).permute(1, 0)

        # --- L2 normalize ---
        descriptors = F.normalize(descriptors, p=2, dim=1)

        return descriptors.cpu().numpy()
