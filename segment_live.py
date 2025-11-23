import os
import cv2
import numpy as np
from fastsam import FastSAM, FastSAMPrompt
import matplotlib
matplotlib.use('Agg')  # headless backend

# --- Configuration ---
IMAGE_FOLDER = 'data/makalii_point/temp_processed/cam2/' 
DEVICE = 'cuda'
model = FastSAM('./weights/FastSAM-x.pt')
KEYFRAME_INTERVAL = 1  # run FastSAM every N frames

# --- Helper Function for Manual Mask Overlay ---
def create_mask_overlay(image_rgb, segments):
    if segments is None or segments.size == 0:
        return image_rgb

    h, w = image_rgb.shape[:2]
    overlay = np.zeros_like(image_rgb, dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.float32)

    for i, mask in enumerate(segments):
        np.random.seed(i * 10)
        color = np.random.randint(0, 256, 3, dtype=np.uint8)

        mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bool = mask_resized.astype(bool)

        overlay[mask_bool, 0] = color[0]
        overlay[mask_bool, 1] = color[1]
        overlay[mask_bool, 2] = color[2]
        alpha[mask_bool] = 0.5

    segmented_frame_rgb = (image_rgb * (1 - alpha[..., None]) + overlay * alpha[..., None]).astype(np.uint8)
    return segmented_frame_rgb

# --- Main Script ---
image_paths = sorted(
    [os.path.join(IMAGE_FOLDER, f) 
     for f in os.listdir(IMAGE_FOLDER) 
     if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
)

WINDOW_NAME = 'FastSAM Segmentation'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

prev_frame_gray = None
prev_mask = None  # propagated mask
segmask = None    # current keyframe masks

for frame_idx, img_path in enumerate(image_paths):
    print(f"Processing {img_path}...")

    frame = cv2.imread(img_path)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Decide whether to run FastSAM ---
    if frame_idx % KEYFRAME_INTERVAL == 0 or segmask is None:
        # Run FastSAM
        results = model(frame_rgb, device=DEVICE, retina_masks=False, imgsz=1024, conf=0.05, iou=0.5)
        prompt_process = FastSAMPrompt(frame_rgb, results, device=DEVICE)
        segmask = prompt_process.everything_prompt()
        if len(segmask) > 0:
            segmask = segmask.cpu().numpy()
        else:
            segmask = None
    else:
        # --- Propagate masks using optical flow ---
        if prev_frame_gray is not None and segmask is not None:
            h_mask, w_mask = segmask.shape[1:]  # (H, W)
            flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, frame_gray, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            # Warp each mask
            warped_masks = []
            for mask in segmask:
                mask_resized = cv2.resize(mask.astype(np.uint8), (flow.shape[1], flow.shape[0]), interpolation=cv2.INTER_NEAREST)
                h, w = mask_resized.shape
                flow_map = np.meshgrid(np.arange(w), np.arange(h))
                flow_map_x = (flow_map[0] + flow[..., 0]).astype(np.float32)
                flow_map_y = (flow_map[1] + flow[..., 1]).astype(np.float32)
                warped = cv2.remap(mask_resized, flow_map_x, flow_map_y, interpolation=cv2.INTER_NEAREST)
                warped_masks.append(warped)
            segmask = np.stack(warped_masks)

    # --- Overlay masks ---
    segmented_frame_rgb = create_mask_overlay(frame_rgb, segmask)
    segmented_frame = cv2.cvtColor(segmented_frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow(WINDOW_NAME, segmented_frame)

    # Store info for next iteration
    prev_frame_gray = frame_gray.copy()
    
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
