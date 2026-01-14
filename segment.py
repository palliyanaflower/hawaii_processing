# # from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
# # import torch
# # from PIL import Image
# # import numpy as np

# # model_type = "vit_t"
# # sam_checkpoint = "./weights/mobile_sam.pt"

# # device = "cuda" if torch.cuda.is_available() else "cpu"

# # mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# # mobile_sam.to(device=device)
# # mobile_sam.eval()


# # mask_generator = SamAutomaticMaskGenerator(mobile_sam)
# # image = np.array(Image.open("data/haleiwa_neighborhood/processed_data/camera/0.png"))
# # masks = mask_generator.generate(image)

# import torch
# import numpy as np
# from PIL import Image

# from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

# device = "cuda" if torch.cuda.is_available() else "cpu"

# sam_checkpoint = "./weights/mobile_sam.pt"

# # ---- Load the MobileSAM model correctly ----
# mobile_sam = sam_model_registry["vit_t"](checkpoint=sam_checkpoint)
# mobile_sam.to(device)
# mobile_sam.eval()

# # ---- Mask generator ----
# mask_generator = SamAutomaticMaskGenerator(mobile_sam)

# # ---- Load image as numpy array ----
# image = np.array(Image.open("data/haleiwa_neighborhood/processed_data/camera/0.png"))

# # ---- Generate masks ----
# masks = mask_generator.generate(image)

# print("Generated masks:", len(masks))


# import matplotlib.pyplot as plt
# import random

# # Make a copy of the image as float32 in [0,1]
# image_rgb = image.astype(np.float32) / 255.0

# plt.figure(figsize=(12, 12))

# for mask_dict in masks:
#     mask = mask_dict['segmentation']  # boolean mask
#     color = np.array([random.random(), random.random(), random.random()])
    
#     # Overlay mask on original image
#     image_rgb[mask] = image_rgb[mask] * 0.5 + color * 0.5  # blend 50/50

# plt.imshow(image_rgb)
# plt.axis('off')
# plt.show()


import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
from transformers import CLIPProcessor, CLIPModel


# ----------------------------
# Load MobileSAM
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

sam_checkpoint = "./weights/mobile_sam.pt"
mobile_sam = sam_model_registry["vit_t"](checkpoint=sam_checkpoint).to(device).eval()

mask_generator = SamAutomaticMaskGenerator(
    model=mobile_sam,
    points_per_side=4,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.95,
    crop_n_layers=0,
    min_mask_region_area=5000,
    box_nms_thresh=0.3,
)

# Load image
image = np.array(Image.open("data/makalii_point/processed_data_imggps/cam2/bag_camera_2_2025_08_13-01_35_58_15/camera/0.png"))
H, W = image.shape[:2]


# ----------------------------
# Load CLIP for semantic filtering
# ----------------------------
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

semantic_labels = ["sky", "water", "ground", "tree", "building", "car", "road"]


# ----------------------------
# Generate instance masks
# ----------------------------
masks = mask_generator.generate(image)
filtered_masks = []


def crop_masked_region(image, mask):
    """Return cropped bounding box region for CLIP classification."""
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    return Image.fromarray(image[y0:y1, x0:x1])


# ----------------------------
# Filter masks by CLIP semantics
# ----------------------------
# for m in masks:
#     mask = m["segmentation"]
    
#     crop = crop_masked_region(image, mask)
#     if crop is None:
#         continue
    
#     # Run CLIP classification
#     inputs = clip_processor(text=semantic_labels, images=crop, return_tensors="pt", padding=True).to(device)
#     outputs = clip_model(**inputs)
    
#     probs = outputs.logits_per_image.softmax(dim=1)[0]
#     predicted_label = semantic_labels[probs.argmax().item()]
    
#     # only keep masks NOT labeled sky or water
#     if predicted_label not in ["sky", "water"]:
#         filtered_masks.append(m)

# print(f"Original masks: {len(masks)}")
# print(f"Filtered masks (excluding sky/water): {len(filtered_masks)}")

sky_water_masks = []
for m in masks:
    mask = m["segmentation"]
    
    crop = crop_masked_region(image, mask)
    if crop is None:
        continue
    
    # Run CLIP classification
    inputs = clip_processor(text=semantic_labels, images=crop, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    
    probs = outputs.logits_per_image.softmax(dim=1)[0]
    predicted_label = semantic_labels[probs.argmax().item()]
    if predicted_label in ["sky", "water"]:
        sky_water_masks.append(m["segmentation"])

H, W = image.shape[:2]
sky_water_union = np.zeros((H, W), dtype=bool)

for seg in sky_water_masks:
    sky_water_union |= seg
valid_region = ~sky_water_union



# ----------------------------
# Visualization
# ----------------------------
img_vis_unfiltered = image.astype(np.float32) / 255.0
img_vis_filtered   = image.astype(np.float32) / 255.0
img_vis_inverted =   image.astype(np.float32) / 255.0

plt.figure(figsize=(12, 6))

# --- Unfiltered masks ---
plt.subplot(1, 2, 1)
for m in masks:
    mask = m["segmentation"]
    color = np.random.rand(3)
    img_vis_unfiltered[mask] = (
        img_vis_unfiltered[mask] * 0.5 + color * 0.5
    )

plt.imshow(img_vis_unfiltered)
plt.title("Unfiltered Masks")
plt.axis("off")

# # --- Filtered masks ---
# plt.subplot(1, 2, 2)
# for m in filtered_masks:
#     mask = m["segmentation"]
#     color = np.random.rand(3)
#     img_vis_filtered[mask] = (
#         img_vis_filtered[mask] * 0.5 + color * 0.5
#     )

# plt.imshow(img_vis_filtered)
# plt.title("Filtered Masks")
# plt.axis("off")

# --- Filtered masks ---
plt.subplot(1, 2, 2)
mask = valid_region
color = np.random.rand(3)
color = np.array([252, 53, 3]) / 252.0
img_vis_inverted[mask] = (
    img_vis_inverted[mask] * 0.5 + color * 0.5
)

plt.imshow(img_vis_inverted)
plt.title("Not Sky or Water Mask")
plt.axis("off")

plt.tight_layout()
plt.show()

