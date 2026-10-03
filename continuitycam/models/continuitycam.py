"""The full ContinuityCam network (Fig. 3).

    motion branch     events -> multi-scale flow M_t and splatting metrics      (Sec. 3.1)
    synthesis branch  events + I_0 -> latent frame Î_t                           (Sec. 3.2)
    latent-frame flow RAFT(I_0, Î_t) -> M̃_t                                      (Sec. 3.3)
    fusion            splat FILM features/images of I_0 with M_t and M̃_t,
                      concatenate Î_t's pyramid, decode                          (Sec. 3.4)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

from .film import FeatureExtractor, Fusion, build_image_pyramid, concatenate_pyramids, splat_pyramid
from .motion import MotionNet
from .synthesis import TriPlaneSynthesisNet

PYRAMID_LEVELS = 7        # FILM feature extractor input levels


class ContinuityCamNet(nn.Module):
    """feature_extractor_ckpt: FILM's pretrained feature extractor (extract.pt), used frozen. RAFT-large (torchvision)
    is downloaded on first use."""

    def __init__(self, feature_extractor_ckpt="extract.pt"):
        super().__init__()
        self.motion = MotionNet()
        self.synthesis = TriPlaneSynthesisNet()
        self.feature_extractor = FeatureExtractor()
        self.feature_extractor.load_state_dict(torch.load(feature_extractor_ckpt, map_location="cpu"))
        self.raft = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False)
        self.fusion = Fusion(4, 3, 128, extra_channels=9)
        self.requires_grad_(False)
        self.eval()

    def raft_flow(self, image0, image1):
        """RAFT flow image0 -> image1 in pixels at the input resolution (RAFT runs at 520x960, inputs resized
        nearest-neighbour)."""
        flow = self.raft(F.interpolate(image0, [520, 960]), F.interpolate(image1, [520, 960]))[-1]
        flow[:, 0] *= image1.shape[-1] / 960.
        flow[:, 1] *= image1.shape[-2] / 520.
        return F.interpolate(flow, image1.shape[-2:], mode="bilinear", align_corners=False)

    def splat(self, pyramid, flows, metrics):
        """flows (coarse to fine, fractions of image size) are resized to each pyramid level
        (fine to coarse) and converted to pixels before splatting."""
        scaled_flows, scaled_metrics = [], []
        for level, flow, metric in zip(pyramid, flows[::-1], metrics[::-1]):
            h, w = level.shape[-2:]
            flow = F.interpolate(flow, (h, w), mode="bilinear", align_corners=False)
            scaled_flows.append(flow * torch.tensor([w, h], device=flow.device)[None, :, None, None])
            scaled_metrics.append(F.interpolate(metric, (h, w), mode="bilinear", align_corners=False))
        return splat_pyramid(pyramid, scaled_flows, scaled_metrics)

    def fuse(self, to_splat, flows, metrics, latent_flows, latent_frame):
        splatted = concatenate_pyramids(self.splat(to_splat, flows, metrics), self.splat(to_splat, latent_flows, metrics))
        latent_frame = F.interpolate(latent_frame, to_splat[0].shape[-2:], mode="bilinear", align_corners=False)
        return self.fusion(concatenate_pyramids(splatted, build_image_pyramid(latent_frame, PYRAMID_LEVELS)[:5]))

    def encode(self, prev_image, flow_volumes, synth_volumes):
        """Everything that does not depend on the target time: run once per window.
        prev_image: [B, 3, H, W] in [-1, 1]; flow_volumes / synth_volumes: [B, 60, H, W] (data/voxel.py)."""
        image_pyramid = build_image_pyramid((prev_image + 1) / 2, PYRAMID_LEVELS)
        return {"prev_image": prev_image, "motion": self.motion.encode(flow_volumes),
                "planes": self.synthesis.encode(synth_volumes, prev_image),
                "to_splat": concatenate_pyramids(image_pyramid[:5], self.feature_extractor(image_pyramid)[:5])}

    def decode(self, enc, t):
        """The frame at time t ([B] in [0, 1], a fraction of the window) as {"frame", "synthesized", "flow"}, images
        in [-1, 1], flow as a fraction of the image size."""
        prev_image = enc["prev_image"]
        out = self.motion.flows_at(enc["motion"], t, mask_flow_with_events=True)
        flows = out["flows"]
        H, W = flows[-1].shape[-2:]
        latent_frame = self.synthesis.decode(enc["planes"], t, prev_image.shape[-2:])
        latent_flow = self.raft_flow(prev_image, latent_frame)
        latent_flow = latent_flow / torch.tensor([float(W), float(H)], device=latent_flow.device)[None, :, None, None]
        latent_flows = [F.interpolate(latent_flow, size=f.shape[-2:]) for f in flows]
        frame = self.fuse(enc["to_splat"], flows, out["metrics"], latent_flows, latent_frame)
        return {"frame": frame, "synthesized": latent_frame, "flow": flows[-1]}

    def forward(self, prev_image, flow_volumes, synth_volumes, t):
        return self.decode(self.encode(prev_image, flow_volumes, synth_volumes), t)["frame"]


def load_model(ckpt, feature_extractor_ckpt="extract.pt", device="cuda"):
    """The pretrained model: ContinuityCamNet with the weights of `ckpt` (e.g. continuitycam_bsergb.ckpt)."""
    net = ContinuityCamNet(feature_extractor_ckpt)
    sd = torch.load(ckpt, map_location="cpu")
    sd = {k[len("net."):]: v for k, v in sd.get("state_dict", sd).items() if k.startswith("net.")}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith(("raft.", "feature_extractor."))]   # pretrained, loaded above
    if missing or unexpected:
        raise RuntimeError(f"loading {ckpt}: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return net.to(device).eval()
