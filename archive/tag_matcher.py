import cv2
import numpy as np

def rotate_keep_all(img, angle):
    """Rotate image without cropping corners."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += new_w / 2 - w / 2
    M[1, 2] += new_h / 2 - h / 2
    return cv2.warpAffine(img, M, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE)

def build_template_bank(templates, scales=(0.7, 0.85, 1.0, 1.2, 1.4),
                        angles=range(0, 360, 30)):
    """templates: dict {name: grayscale image}"""
    bank = []
    for name, tpl in templates.items():
        for s in scales:
            t = cv2.resize(tpl, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            for a in angles:
                bank.append((name, s, a, rotate_keep_all(t, a)))
    return bank

def match_tag(crop_bgr, bank):
    """Return (best_score, (cx, cy) in crop coords, template_name, scale, angle)."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    best = (-1.0, None, None, None, None)
    for name, s, a, t in bank:
        th, tw = t.shape[:2]
        if th >= gray.shape[0] or tw >= gray.shape[1]:
            continue
        res = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score > best[0]:
            best = (score, (loc[0] + tw // 2, loc[1] + th // 2), name, s, a)
    return best