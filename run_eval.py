import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import features as F

TRANSFORMS = ["translate", "rot30", "rot90", "bright", "rot30_bright"]
CONFIGS = [("simple", "ssd"), ("simple", "ratio"), ("mops", "ssd"), ("mops", "ratio")]
os.makedirs("out", exist_ok=True)


def run_pair(g1, g2, M, desc, matcher):
    k1, k2 = F.detect_keypoints(g1), F.detect_keypoints(g2)
    d = F.simple_describe if desc == "simple" else F.mops_describe
    m = (F.ssd_match if matcher == "ssd" else F.ratio_match)(d(g1, k1), d(g2, k2))
    return F.evaluate(m, k1, k2, M), k1, k2, m


def main(paths):
    for p in paths:
        g1 = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        name = os.path.splitext(os.path.basename(p))[0]
        cv2.imwrite(f"out/{name}_harris.png", F.draw_keypoints(g1, F.detect_keypoints(g1)))
        print(f"\n== {name} ==")
        print("transform".ljust(14) + "".join(f"{d}+{m}".rjust(15) for d, m in CONFIGS))
        for t in TRANSFORMS:
            g2, M = F.make_transform(g1, t)
            plt.figure(figsize=(4, 4))
            row, counts = [], []
            for d, m in CONFIGS:
                (fpr, tpr, auc, correct), *_ = run_pair(g1, g2, M, d, m)
                row.append(auc)
                counts.append(int(correct.sum()))
                plt.plot(fpr, tpr, label=f"{d}+{m} ({auc:.3f})")
            plt.plot([0, 1], [0, 1], "k:", lw=0.8)
            plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"{name}: {t}")
            plt.legend(fontsize=7); plt.tight_layout()
            plt.savefig(f"out/{name}_{t}_roc.png", dpi=120); plt.close()
            print(t.ljust(14) + "".join(f"{a:15.3f}" for a in row))
            print("  #correct".ljust(14) + "".join(f"{c:15d}" for c in counts))


if __name__ == "__main__":
    main(sys.argv[1:])