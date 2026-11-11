import os
import sys

import cv2

import features as F


def main(path1, path2, n_show=60):
    g1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    g2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    k1, k2 = F.detect_keypoints(g1), F.detect_keypoints(g2)
    matches = F.ratio_match(F.mops_describe(g1, k1), F.mops_describe(g2, k2))
    matches = sorted(matches, key=lambda m: m.distance)[:n_show]
    out = cv2.drawMatches(g1, F.to_cv_keypoints(k1), g2, F.to_cv_keypoints(k2),matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    os.makedirs("out", exist_ok=True)
    cv2.imwrite("out/real_matches.png", out)
    print(f"{len(k1)} / {len(k2)} keypoints, drew {len(matches)} best matches")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 60)
