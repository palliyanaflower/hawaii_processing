import torch
import matplotlib.pyplot as plt

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from lightglue import viz2d


class LightGlueVisualizer:
    def __init__(self, max_keypoints=2048, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.extractor = SuperPoint(
            max_num_keypoints=max_keypoints
        ).eval().to(self.device)

        self.matcher = LightGlue(
            features="superpoint"
        ).eval().to(self.device)

    def _extract_and_match(self, image0, image1):
        """Internal helper: extract features and run LightGlue"""
        feats0 = self.extractor.extract(image0.to(self.device))
        feats1 = self.extractor.extract(image1.to(self.device))

        matches01 = self.matcher({"image0": feats0, "image1": feats1})

        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]

        return feats0, feats1, matches01

    def visualize_matches(self, image_path0, image_path1, title=True):
        """Visualize LightGlue matches between two images"""
        image0 = load_image(image_path0)
        image1 = load_image(image_path1)

        feats0, feats1, matches01 = self._extract_and_match(image0, image1)

        kpts0 = feats0["keypoints"]
        kpts1 = feats1["keypoints"]
        matches = matches01["matches"]

        m_kpts0 = kpts0[matches[..., 0]]
        m_kpts1 = kpts1[matches[..., 1]]

        viz2d.plot_images([image0, image1])
        viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)

        if title:
            viz2d.add_text(
                0, f"Stop after {matches01['stop']} layers", fs=20
            )

    def visualize_inliers(self, image_path0, image_path1, m_kpts0, m_kpts1, title=True):
        """Visualize LightGlue matches between two images"""
        image0 = load_image(image_path0)
        image1 = load_image(image_path1)

        viz2d.plot_images([image0, image1])
        viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)


    def visualize_keypoints(self, image_path0, image_path1):
        """Visualize keypoints with pruning confidence coloring"""
        image0 = load_image(image_path0)
        image1 = load_image(image_path1)

        feats0, feats1, matches01 = self._extract_and_match(image0, image1)

        kpts0 = feats0["keypoints"]
        kpts1 = feats1["keypoints"]

        kpc0 = viz2d.cm_prune(matches01["prune0"])
        kpc1 = viz2d.cm_prune(matches01["prune1"])

        viz2d.plot_images([image0, image1])
        viz2d.plot_keypoints(
            [kpts0, kpts1],
            colors=[kpc0, kpc1],
            ps=10
        )



    def get_matched_keypoints(self, image_path0, image_path1):
        image0 = load_image(image_path0)
        image1 = load_image(image_path1)

        feats0 = self.extractor.extract(image0.to(self.device))
        feats1 = self.extractor.extract(image1.to(self.device))
        matches01 = self.matcher({"image0": feats0, "image1": feats1})

        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]

        kpts0 = feats0["keypoints"]
        kpts1 = feats1["keypoints"]
        matches = matches01["matches"]

        m_kpts0 = kpts0[matches[:, 0]]
        m_kpts1 = kpts1[matches[:, 1]]

        # return as numpy pixel coords
        return (
            m_kpts0.cpu().numpy(),
            m_kpts1.cpu().numpy()
        )

    def get_matched_keypoints_and_descriptors(self, image_path0, image_path1):
        image0 = load_image(image_path0)
        image1 = load_image(image_path1)

        feats0 = self.extractor.extract(image0.to(self.device))
        feats1 = self.extractor.extract(image1.to(self.device))
        matches01 = self.matcher({"image0": feats0, "image1": feats1})

        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]

        kpts0 = feats0["keypoints"]
        kpts1 = feats1["keypoints"]

        desc0 = feats0["descriptors"]
        desc1 = feats1["descriptors"]

        matches = matches01["matches"]

        # Matched keypoints
        m_kpts0 = kpts0[matches[:, 0]]
        m_kpts1 = kpts1[matches[:, 1]]

        # Matched descriptors
        m_desc0 = desc0[matches[:, 0]]
        m_desc1 = desc1[matches[:, 1]]

        # Score for keypoint match
        scores = matches01["scores"]


        return (
            m_kpts0.cpu().numpy(),
            m_kpts1.cpu().numpy(),
            m_desc0.cpu().numpy(),
            m_desc1.cpu().numpy(),
            scores.cpu().detach().numpy()
        )
