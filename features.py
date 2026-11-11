import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist

_VAR_EPS = 1e-10


def harris_compute(img, k=0.05, sigma=0.5):
    """نقشه‌ی R هریس و نقشه‌ی جهت (درجه) را برمی‌گرداند."""
    img = img.astype(np.float64)
    ix = ndimage.sobel(img, axis=1, mode="reflect")
    iy = ndimage.sobel(img, axis=0, mode="reflect")

    def win(a):
        return ndimage.gaussian_filter(a, sigma, mode="reflect", truncate=4.0)

    sxx, syy, sxy = win(ix * ix), win(iy * iy), win(ix * iy)
    r = (sxx * syy - sxy ** 2) - k * (sxx + syy) ** 2
    orient = np.degrees(np.arctan2(win(iy), win(ix)))
    return r, orient


def detect_keypoints(img, n_max=500, rel_thresh=0.01):
    """فهرست (x, y, angle) نقاط کلیدی، مرتب بر اساس قدرت R."""
    r, orient = harris_compute(img)
    local_max = r == ndimage.maximum_filter(r, size=7, mode="reflect")
    mask = local_max & (r > rel_thresh * r.max()) & (r > 0)
    ys, xs = np.nonzero(mask)
    order = np.argsort(-r[ys, xs])[:n_max]
    ys, xs = ys[order], xs[order]
    return [(int(x), int(y), float(orient[y, x])) for x, y in zip(xs, ys)]


def simple_describe(img, kps):
    """توصیف‌گر ساده‌ی ۲۵بعدی؛ خروجی (N, 25)."""
    pad = 2
    padded = np.pad(img.astype(np.float32), pad, mode="reflect")
    out = np.zeros((len(kps), 25), dtype=np.float32)
    for i, (x, y, _) in enumerate(kps):
        out[i] = padded[y:y + 2 * pad + 1, x:x + 2 * pad + 1].ravel()
    return out


def mops_describe(img, kps, blur_sigma=2.0):
    """توصیف‌گر MOPS ساده‌شده‌ی ۶۴بعدی؛ خروجی (N, 64)."""
    src = img.astype(np.float32)
    if blur_sigma > 0:
        src = ndimage.gaussian_filter(src, blur_sigma, mode="reflect")

    out = np.zeros((len(kps), 64), dtype=np.float32)
    for i, (x, y, ang) in enumerate(kps):
        t = np.radians(ang)
        c, s = np.cos(t), np.sin(t)
        T1 = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]], dtype=np.float64)
        R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float64)
        S = np.diag([0.2, 0.2, 1.0])
        T2 = np.array([[1, 0, 3.5], [0, 1, 3.5], [0, 0, 1]], dtype=np.float64)
        M = (T2 @ S @ R @ T1)[:2]
        patch = cv2.warpAffine(src, M, (8, 8), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        patch = patch.astype(np.float64).ravel()
        patch -= patch.mean()
        var = patch.var()
        if var < _VAR_EPS:
            continue
        out[i] = patch / np.sqrt(var)
    return out


def ssd_match(d1, d2):
    """تطبیق با کمترین SSD؛ فهرست cv2.DMatch."""
    if len(d1) == 0 or len(d2) == 0:
        return []
    dist = cdist(d1, d2, "sqeuclidean")
    best = np.argmin(dist, axis=1)
    return [cv2.DMatch(int(i), int(j), float(dist[i, j])) for i, j in enumerate(best)]


def ratio_match(d1, d2):
    """تطبیق با آزمون نسبت؛ فهرست cv2.DMatch."""
    if len(d1) == 0 or len(d2) == 0:
        return []
    if len(d2) < 2:
        return ssd_match(d1, d2)
    dist = cdist(d1, d2, "sqeuclidean")
    idx = np.argpartition(dist, 1, axis=1)[:, :2]
    two = np.take_along_axis(dist, idx, axis=1)
    swap = two[:, 0] > two[:, 1]
    idx[swap] = idx[swap][:, ::-1]
    two = np.take_along_axis(dist, idx, axis=1)
    d_first, d_second = two[:, 0], two[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(d_second > 0, d_first / d_second, 1.0)
    return [cv2.DMatch(int(i), int(idx[i, 0]), float(ratio[i])) for i in range(len(d1))]


def evaluate(matches, kps1, kps2, M, tol=3.0):
    """خروجی: (fpr, tpr, auc, is_correct)."""
    if len(matches) == 0:
        return np.array([0, 1.0]), np.array([0, 1.0]), float("nan"), np.array([])
    p = np.array([[kps1[m.queryIdx][0], kps1[m.queryIdx][1], 1.0] for m in matches])
    q = np.array([[kps2[m.trainIdx][0], kps2[m.trainIdx][1]] for m in matches])
    pred = p @ np.asarray(M, dtype=np.float64).T
    correct = np.linalg.norm(pred - q, axis=1) <= tol
    dists = np.array([m.distance for m in matches])

    order = np.argsort(dists, kind="stable")
    c = correct[order]
    n_pos, n_neg = c.sum(), (~c).sum()
    tp = np.concatenate([[0], np.cumsum(c)])
    fp = np.concatenate([[0], np.cumsum(~c)])
    tpr = tp / n_pos if n_pos > 0 else np.zeros_like(tp, dtype=float)
    fpr = fp / n_neg if n_neg > 0 else np.zeros_like(fp, dtype=float)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    auc = float(trapz(tpr, fpr)) if n_pos > 0 and n_neg > 0 else float("nan")
    return fpr, tpr, auc, correct


def make_transform(img, kind):
    """(تصویر تبدیل‌یافته، ماتریس آفین ۲×۳) برای یکی از تبدیل‌های آزمون."""
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    eye = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)

    def bright(a):
        return np.clip(1.5 * a.astype(np.float32) - 20, 0, 255).astype(img.dtype)

    if kind == "translate":
        M = np.array([[1, 0, 30], [0, 1, 20]], dtype=np.float64)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT), M
    if kind in ("rot30", "rot90", "rot30_bright"):
        ang = 90 if kind == "rot90" else 30
        M = cv2.getRotationMatrix2D(center, ang, 1.0)
        out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return (bright(out) if kind == "rot30_bright" else out), M
    if kind == "bright":
        return bright(img), eye
    raise ValueError(f"unknown transform: {kind}")


def draw_keypoints(img, kps):
    """رسم دایره و خط جهت روی تصویر."""
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    for x, y, ang in kps:
        t = np.radians(ang)
        end = (int(round(x + 8 * np.cos(t))), int(round(y + 8 * np.sin(t))))
        cv2.circle(vis, (x, y), 4, (0, 255, 0), 1)
        cv2.line(vis, (x, y), end, (0, 0, 255), 1)
    return vis


def to_cv_keypoints(kps):
    """تبدیل به cv2.KeyPoint برای cv2.drawMatches."""
    return [cv2.KeyPoint(float(x), float(y), 8, float(a)) for x, y, a in kps]


compute_harris = harris_compute
describe_mops = mops_describe
match_ratio = ratio_match
