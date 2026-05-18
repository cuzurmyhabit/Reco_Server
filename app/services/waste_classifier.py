"""이미지 기반 분리수거 재질 추정 (ML + 비전 휴리스틱)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models

from app.services.material_heuristics import (
    enforce_material_rules,
    fuse_material_scores,
    is_can_like,
    is_glass_like,
    is_paper_like,
    is_plastic_like,
    object_prior_vector,
    visual_material_scores,
)

MATERIAL_LABELS: Tuple[str, ...] = ("플라스틱", "유리", "종이", "금속", "기타")
SUMMARY_LABELS: Tuple[str, ...] = ("플라스틱", "유리", "금속", "기타")

_IMAGENET_TO_MATERIAL: Dict[int, Dict[str, float]] = {
    437: {"플라스틱": 0.75, "기타": 0.25},
    898: {"플라스틱": 0.85, "기타": 0.15},
    737: {"플라스틱": 0.45, "금속": 0.45, "기타": 0.1},
    514: {"플라스틱": 0.8, "기타": 0.2},
    616: {"플라스틱": 0.85, "기타": 0.15},
    429: {"플라스틱": 0.7, "기타": 0.3},
    460: {"유리": 0.8, "플라스틱": 0.1, "기타": 0.1},
    720: {"유리": 0.7, "플라스틱": 0.15, "기타": 0.15},
    849: {"유리": 0.85, "기타": 0.15},
    573: {"유리": 0.7, "기타": 0.3},
    478: {"종이": 0.85, "기타": 0.15},
    495: {"종이": 0.9, "기타": 0.1},
    921: {"종이": 0.8, "기타": 0.2},
    687: {"종이": 0.75, "기타": 0.25},
    506: {"금속": 0.9, "기타": 0.1},
    512: {"금속": 0.9, "기타": 0.1},
    657: {"금속": 0.92, "기타": 0.08},
    469: {"금속": 0.85, "기타": 0.15},
    767: {"금속": 0.7, "기타": 0.3},
    606: {"금속": 0.75, "기타": 0.25},
    545: {"금속": 0.8, "기타": 0.2},
    528: {"금속": 0.75, "기타": 0.25},
}


@dataclass(frozen=True)
class ClassificationResult:
    summary: Dict[str, float]
    detail: Dict[str, float]
    primary_material: str
    confidence: float


class WasteClassifier:
    def __init__(self, device: str | None = None, smooth_frames: int = 8):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.smooth_frames = smooth_frames
        self._history: deque[np.ndarray] = deque(maxlen=smooth_frames)

        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        self.model = models.mobilenet_v3_small(weights=weights).to(self.device)
        self.model.eval()
        self.preprocess = weights.transforms()

    def classify_bytes(self, image_bytes: bytes) -> ClassificationResult:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("이미지를 디코딩할 수 없습니다.")
        return self.classify_bgr(frame)

    def classify_bgr_instant(
        self,
        frame_bgr: np.ndarray,
        object_name: Optional[str] = None,
    ) -> ClassificationResult:
        dist = self._fused_distribution(frame_bgr, object_name)
        return self._dist_to_result(dist)

    def classify_bgr(self, frame_bgr: np.ndarray) -> ClassificationResult:
        dist = self._fused_distribution(frame_bgr, None)
        self._history.append(dist)
        smoothed = np.mean(self._history, axis=0)
        smoothed /= smoothed.sum()
        return self._dist_to_result(smoothed)

    def reset(self) -> None:
        self._history.clear()

    def _fused_distribution(
        self, frame_bgr: np.ndarray, object_name: Optional[str]
    ) -> np.ndarray:
        ml = self._predict_ml_distribution(frame_bgr)
        vis = visual_material_scores(frame_bgr)
        prior = object_prior_vector(object_name)
        fused = fuse_material_scores(
            ml,
            vis,
            prior,
            object_name,
            can_boost=is_can_like(frame_bgr),
            plastic_boost=is_plastic_like(frame_bgr),
            glass_boost=is_glass_like(frame_bgr),
            paper_boost=is_paper_like(frame_bgr),
        )
        return enforce_material_rules(frame_bgr, fused, object_name)

    def _dist_to_result(self, dist: np.ndarray) -> ClassificationResult:
        detail = {
            label: round(float(dist[i]) * 100, 1)
            for i, label in enumerate(MATERIAL_LABELS)
        }
        summary = _to_summary(detail)
        primary = max(MATERIAL_LABELS, key=lambda k: detail[k])
        return ClassificationResult(
            summary=summary,
            detail=detail,
            primary_material=primary,
            confidence=round(detail[primary] / 100.0, 4),
        )

    def _predict_ml_distribution(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = self.preprocess(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        scores = np.zeros(len(MATERIAL_LABELS), dtype=np.float64)
        label_index = {name: i for i, name in enumerate(MATERIAL_LABELS)}

        for class_id, material_weights in _IMAGENET_TO_MATERIAL.items():
            p = probs[class_id]
            if p < 1e-4:
                continue
            for material, weight in material_weights.items():
                scores[label_index[material]] += p * weight

        if scores.sum() < 1e-3:
            top5 = np.argsort(probs)[-5:][::-1]
            for idx in top5:
                name = _guess_from_imagenet_label(int(idx), float(probs[idx]))
                if name:
                    scores[label_index[name]] += probs[idx]

        if scores.sum() < 1e-6:
            scores[label_index["기타"]] = 1.0
        else:
            scores /= scores.sum()
        return scores


def _to_summary(detail: Dict[str, float]) -> Dict[str, float]:
    plastic = detail.get("플라스틱", 0.0)
    glass = detail.get("유리", 0.0)
    metal = detail.get("금속", 0.0)
    other = detail.get("종이", 0.0) + detail.get("기타", 0.0)
    total = plastic + glass + metal + other or 1.0
    p = round(plastic / total * 100, 1)
    g = round(glass / total * 100, 1)
    m = round(metal / total * 100, 1)
    o = round(100.0 - p - g - m, 1)
    return {
        "플라스틱": p,
        "유리": g,
        "금속": m,
        "기타": max(o, 0.0),
    }


def _guess_from_imagenet_label(class_id: int, prob: float) -> Optional[str]:
    plastic_hints = {437, 898, 514, 616, 429, 531, 776}
    glass_hints = {460, 573, 849, 720}
    paper_hints = {478, 495, 687, 921, 554}
    metal_hints = {469, 512, 657, 606, 767, 545, 528, 506, 737}

    if class_id in metal_hints:
        return "금속"
    if class_id in plastic_hints:
        return "플라스틱"
    if class_id in glass_hints:
        return "유리"
    if class_id in paper_hints:
        return "종이"
    if prob > 0.15:
        return "기타"
    return None
