"""
30330 Image Analysis - Exercise 5: Correspondence Problem

Correlation based matching between the stereo pair PIC1_L.png / PIC1_R.png
and least-squares fitting of an affine transformation described by:

    x' = p0 + p1*x + p2*y
    y' = q0 + q1*x + q2*y

Steps:
  1. One patch from the left image is searched for in the whole right image
     with the SAD (m1), SSD (m2) and ZNCC measures.
  2. 30-50 patches are picked automatically (Harris corners, spread on a grid),
     matched, framed and numbered in both images, sorted by confidence.
     (OpenCV is used for I/O, colour conversion, template matching, Harris,
     RANSAC and warping; SAD and the forward-mapping hint stay in numpy.)
  3. The affine parameters are solved from 3 pairs (exact) and from all pairs
     (least squares), plus a RANSAC variant that rejects false matches.
  4. The left image is forward-mapped with the affine transform into a new
     image (as in the hint) and compared with the right image.

"""

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# Parameters
HERE = Path(__file__).resolve().parent
LEFT_FILE = HERE / "PIC1_L.png"
RIGHT_FILE = HERE / "PIC1_R.png"
OUT_DIR = HERE / "results"

HALF = 7                 # patch half size -> 15x15 patch
N_POINTS = 40            # number of areas to match (30-50)
GRID = (5, 8)            # rows x cols of the grid used to spread the points
# search window in the right image relative to (x, y) of the left patch
# (the right camera sees the scene shifted ~60-100 px to the left)
DX_RANGE = (-160, 40)
DY_RANGE = (-30, 30)
MIN_ZNCC = 0.8           # matches below this are considered unreliable
RANSAC_ITERS = 2000
RANSAC_THRESH = 4.0      # px
SHOW = True              # plt.show() at the end

# Image I/O
def load_gray(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

# Matching measures
def match_scores(patch, search):
    """Slide `patch` over `search` and return SAD, SSD and ZNCC maps.

    m1 = sum |f - g|      (lower is better)  - not in OpenCV, shifted-window sum
    m2 = sum (f - g)^2    (lower is better)  - cv2.TM_SQDIFF
    ZNCC = normalised cross correlation of zero-mean patches, in [-1, 1]
           (higher is better)               - cv2.TM_CCOEFF_NORMED
    Map index (r, c) corresponds to the window whose top-left corner is (r, c).
    """
    h, w = patch.shape
    H, W = search.shape[0] - h + 1, search.shape[1] - w + 1
    sad = np.zeros((H, W), np.float32)
    for i in range(h):
        for j in range(w):
            sad += np.abs(search[i:i + H, j:j + W] - patch[i, j])
    ssd = cv2.matchTemplate(search, patch, cv2.TM_SQDIFF)
    zncc = cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED)
    zncc = np.nan_to_num(zncc, nan=-1.0, posinf=-1.0, neginf=-1.0)  # flat windows
    return sad, ssd, zncc


def get_patch(img, x, y, half=HALF):
    return img[y - half:y + half + 1, x - half:x + half + 1]


def match_point(left, right, x, y, half=HALF, dx=DX_RANGE, dy=DY_RANGE):
    """Find the best ZNCC match in `right` of the patch centred at (x, y) in
    `left`. Returns x', y', zncc, confidence (peak ratio), sad, ssd."""
    H, W = right.shape
    x0 = max(half, x + dx[0]); x1 = min(W - half - 1, x + dx[1])
    y0 = max(half, y + dy[0]); y1 = min(H - half - 1, y + dy[1])
    search = right[y0 - half:y1 + half + 1, x0 - half:x1 + half + 1]
    patch = get_patch(left, x, y, half)
    sad, ssd, zncc = match_scores(patch, search)

    r, c = np.unravel_index(np.argmax(zncc), zncc.shape)
    best = zncc[r, c]
    # sub-pixel refinement with a parabola through the peak and its neighbours
    sx = sy = 0.0
    if 0 < c < zncc.shape[1] - 1:
        a, b, d = zncc[r, c - 1], zncc[r, c], zncc[r, c + 1]
        sx = 0.5 * (a - d) / (a - 2 * b + d + 1e-12)
    if 0 < r < zncc.shape[0] - 1:
        a, b, d = zncc[r - 1, c], zncc[r, c], zncc[r + 1, c]
        sy = 0.5 * (a - d) / (a - 2 * b + d + 1e-12)

    # Confidence: how much the best peak stands out from the second best peak
    # outside the neighbourhood of the best one (ambiguity / repeated texture).
    masked = zncc.copy()
    masked[max(0, r - half):r + half + 1, max(0, c - half):c + half + 1] = -1
    second = masked.max() if masked.size else -1
    confidence = (1 - second) / (1 - best + 1e-6)   # >1, large = unambiguous

    return (x0 + c + sx, y0 + r + sy, best, confidence,
            sad[r, c], ssd[r, c])


# Choosing the areas in the first image
def harris(img):
    return cv2.cornerHarris(cv2.GaussianBlur(img, (0, 0), 1.0), 5, 3, 0.04)


def select_points(img, n=N_POINTS, grid=GRID, half=HALF):
    """Pick the strongest Harris corner in every grid cell so the points are
    textured (good for correlation) AND spread over the whole image (good for
    the affine fit). The best cells are kept until `n` points are chosen."""
    R = harris(img)
    H, W = img.shape
    margin = half + 2
    ys = np.linspace(margin, H - margin, grid[0] + 1).astype(int)
    xs = np.linspace(margin, W - margin, grid[1] + 1).astype(int)
    cand = []
    for i in range(grid[0]):
        for j in range(grid[1]):
            cell = R[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            r, c = np.unravel_index(np.argmax(cell), cell.shape)
            cand.append((cell[r, c], xs[j] + c, ys[i] + r))
    cand.sort(reverse=True)
    return [(x, y) for _, x, y in cand[:n]]


# Affine transformation
def design_matrix(xy):
    """Rows [1, x, y] -> x' = A p, y' = A q."""
    return np.column_stack([np.ones(len(xy)), xy[:, 0], xy[:, 1]])


def affine_exact(xy, xy2):
    """Well-determined case: 3 point pairs -> 6 equations, 6 unknowns.
    cv2.getAffineTransform solves it; singular when the points are collinear."""
    A = design_matrix(xy[:3])
    if abs(np.linalg.det(A)) < 1e-6:
        raise ValueError("The three points are collinear - no unique solution")
    M = cv2.getAffineTransform(xy[:3].astype(np.float32), xy2[:3].astype(np.float32))
    return M[0, [2, 0, 1]].astype(float), M[1, [2, 0, 1]].astype(float)  # cv2: [a b c] -> [c a b]


def affine_lsq(xy, xy2):
    """Over-determined case (N > 3): minimise ||A p - x'||^2, ||A q - y'||^2,
    i.e. p = (A^T A)^-1 A^T x'."""
    A = design_matrix(xy)
    p = np.linalg.lstsq(A, xy2[:, 0], rcond=None)[0]
    q = np.linalg.lstsq(A, xy2[:, 1], rcond=None)[0]
    return p, q


def apply_affine(p, q, xy):
    A = design_matrix(np.atleast_2d(xy))
    return np.column_stack([A @ p, A @ q])


def affine_ransac(xy, xy2, iters=RANSAC_ITERS, thresh=RANSAC_THRESH):
    """False-match filter: cv2.estimateAffine2D (RANSAC) finds the inliers,
    then a least-squares refit on the inliers."""
    _, mask = cv2.estimateAffine2D(xy.astype(np.float32), xy2.astype(np.float32),
                                   method=cv2.RANSAC, ransacReprojThreshold=thresh,
                                   maxIters=iters, refineIters=0)
    best = mask.ravel().astype(bool)
    p, q = affine_lsq(xy[best], xy2[best])
    return p, q, best


def forward_warp(img, p, q, out_shape):
    """The hint of the exercise: for each pixel (x, y) of the old image compute
    (x', y'), skip it when outside the new image, otherwise copy the value."""
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W]
    xp = np.rint(p[0] + p[1] * xx + p[2] * yy).astype(int)
    yp = np.rint(q[0] + q[1] * xx + q[2] * yy).astype(int)
    ok = (xp >= 0) & (xp < out_shape[1]) & (yp >= 0) & (yp < out_shape[0])
    out = np.full(out_shape, np.nan)
    out[yp[ok], xp[ok]] = img[yy[ok], xx[ok]]
    return out


def inverse_warp(img, p, q, out_shape):
    """Hole-free alternative: cv2.warpAffine inverts the map internally and
    samples the source with bilinear interpolation."""
    M = np.array([[p[1], p[2], p[0]], [q[1], q[2], q[0]]], np.float64)   # cv2 layout
    return cv2.warpAffine(img, M, (out_shape[1], out_shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"))

# Visualisation helpers
def draw_boxes(ax, pts, color, half=HALF, labels=None):
    for i, (x, y) in enumerate(pts):
        ax.add_patch(Rectangle((x - half, y - half), 2 * half + 1, 2 * half + 1,
                               fill=False, ec=color, lw=1.2))
        lab = labels[i] if labels is not None else i + 1
        ax.text(x + half + 1, y - half, str(lab), color=color, fontsize=7,
                weight="bold", va="bottom")

def save(fig, name):
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / name, dpi=130, bbox_inches="tight")
    print(f"  saved {OUT_DIR / name}")

# Main
def main():
    left = load_gray(LEFT_FILE)
    right = load_gray(RIGHT_FILE)
    H, W = left.shape
    print(f"Images: {W}x{H}")
    OUT_DIR.mkdir(exist_ok=True)

    # Task 1: one area, full search over the right image
    print("\n[1] Single patch, exhaustive search")

    x, y = 350, 40                          # the "HUSK!" sign
    patch = get_patch(left, x, y, 12)
    sad, ssd, zncc = match_scores(patch, right)
    res = {}
    for name, m, fn in (("SAD m1", sad, np.argmin), ("SSD m2", ssd, np.argmin),
                        ("ZNCC", zncc, np.argmax)):
        r, c = np.unravel_index(fn(m), m.shape)
        res[name] = (c + 12, r + 12)
        print(f"  {name:7s}: ({x},{y}) -> ({c + 12},{r + 12})")

    fig, axs = plt.subplots(2, 3, figsize=(15, 7))
    axs[0, 0].imshow(left, cmap="gray"); axs[0, 0].set_title("Left: template")
    draw_boxes(axs[0, 0], [(x, y)], "lime", 12, ["T"])
    axs[0, 1].imshow(right, cmap="gray"); axs[0, 1].set_title("Right: best matches")
    for (name, (bx, by)), col in zip(res.items(), ("red", "orange", "lime")):
        draw_boxes(axs[0, 1], [(bx, by)], col, 12, [name])
    axs[0, 2].imshow(patch, cmap="gray"); axs[0, 2].set_title("Template 25x25")
    for a, (name, m) in zip(axs[1], (("SAD m1 (min)", sad), ("SSD m2 (min)", ssd),
                                     ("ZNCC (max)", zncc))):
        im = a.imshow(m, cmap="viridis"); a.set_title(name)
        fig.colorbar(im, ax=a, fraction=0.03)
    for a in axs.flat:
        a.axis("off")
    fig.tight_layout(); save(fig, "1_single_patch.png")

    # Task 2: 30-50 areas
    print(f"\n[2] Matching {N_POINTS} areas")
    pts = select_points(left)
    rows = []
    for (x, y) in pts:
        x2, y2, z, conf, s1, s2 = match_point(left, right, x, y)
        # left-right consistency: match back from right to left
        xb, yb, *_ = match_point(right, left, int(round(x2)), int(round(y2)),
                                 dx=(-DX_RANGE[1], -DX_RANGE[0]),
                                 dy=(-DY_RANGE[1], -DY_RANGE[0]))
        lr_err = np.hypot(xb - x, yb - y)
        rows.append((x, y, x2, y2, z, conf, s1, s2, lr_err))
    M = np.array(rows)
    # sort by confidence: ZNCC peak value first, then peak distinctiveness
    order = np.lexsort((-M[:, 5], -np.round(M[:, 4], 2)))
    M = M[order]
    good = (M[:, 4] >= MIN_ZNCC) & (M[:, 8] <= 2.0)

    print(" #     x     y      x'      y'    ZNCC  conf   SAD      SSD    LR  ok")
    for i, r in enumerate(M):
        print(f"{i + 1:2d} {r[0]:5.0f} {r[1]:5.0f} {r[2]:7.1f} {r[3]:7.1f}"
              f"  {r[4]:5.3f} {r[5]:5.1f} {r[6]:6.0f} {r[7]:8.0f} {r[8]:4.1f}"
              f"  {'y' if good[i] else '-'}")
    np.savetxt(OUT_DIR / "matches.csv",
               M, delimiter=",", fmt="%.3f",
               header="x,y,x2,y2,zncc,confidence,sad,ssd,lr_err", comments="")

    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    axs[0].imshow(left, cmap="gray"); axs[0].set_title("Left (x, y)")
    axs[1].imshow(right, cmap="gray"); axs[1].set_title("Right (x', y')  green = accepted, red = rejected")
    for i, r in enumerate(M):
        col = "lime" if good[i] else "red"
        draw_boxes(axs[0], [r[0:2]], col, labels=[i + 1])
        draw_boxes(axs[1], [r[2:4]], col, labels=[i + 1])
    for a in axs:
        a.axis("off")
    fig.tight_layout(); save(fig, "2_matches.png")

    # Task 3: affine parameters
    print("\n[3] Affine transformation")
    xy, xy2 = M[:, 0:2], M[:, 2:4]
    # exact solution from 3 well spread, reliable points
    g = np.flatnonzero(good)
    tri = [g[np.argmin(xy[g, 0] + xy[g, 1])], g[np.argmax(xy[g, 0])],
           g[np.argmax(xy[g, 1] - xy[g, 0])]]
    p3, q3 = affine_exact(xy[tri], xy2[tri])
    p_all, q_all = affine_lsq(xy, xy2)
    p, q, inl = affine_ransac(xy, xy2)

    def report(name, p, q, mask):
        e = np.linalg.norm(apply_affine(p, q, xy[mask]) - xy2[mask], axis=1)
        print(f"  {name:28s} p = [{p[0]:8.2f} {p[1]:7.4f} {p[2]:7.4f}]"
              f"  q = [{q[0]:8.2f} {q[1]:7.4f} {q[2]:7.4f}]"
              f"  RMS = {np.sqrt((e ** 2).mean()):5.2f} px (N={mask.sum()})")
    report(f"exact, 3 pts {[int(t) + 1 for t in tri]}", p3, q3, inl)
    report("least squares, all pts", p_all, q_all, inl)
    report("RANSAC + LSQ on inliers", p, q, inl)
    print(f"  RANSAC inliers: {inl.sum()} / {len(inl)}")

    # Task 4: apply the transform and check
    print("\n[4] Warping left -> right frame")
    big = (2 * H, 2 * W)                    # "twice as big as the original"
    fwd = forward_warp(left, p, q, big)
    inv = inverse_warp(left, p, q, right.shape)
    diff_before = np.abs(left - right)
    diff_after = np.abs(inv - right)
    print(f"  mean |L - R| before: {np.nanmean(diff_before):5.1f}"
          f"   after warping: {np.nanmean(diff_after):5.1f}")

    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    axs[0, 0].imshow(fwd, cmap="gray")
    axs[0, 0].set_title("Forward mapping into 2x canvas (hint method, note holes)")
    over = np.dstack([right / 255, np.nan_to_num(inv) / 255, right / 255])
    axs[0, 1].imshow(over.clip(0, 1))
    axs[0, 1].set_title("Overlay: warped left (green) vs right (magenta)")
    axs[1, 0].imshow(diff_before, cmap="magma", vmin=0, vmax=120)
    axs[1, 0].set_title(f"|L - R| before (mean {np.nanmean(diff_before):.1f})")
    axs[1, 1].imshow(diff_after, cmap="magma", vmin=0, vmax=120)
    axs[1, 1].set_title(f"|warp(L) - R| after (mean {np.nanmean(diff_after):.1f})")
    for a in axs.flat:
        a.axis("off")
    fig.tight_layout(); save(fig, "3_affine_warp.png")

    # Residuals per match: shows the parallax that an affine map can't model
    res = apply_affine(p, q, xy) - xy2
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(right, cmap="gray")
    ax.quiver(xy2[:, 0], xy2[:, 1], res[:, 0], res[:, 1],
              color=np.where(inl, "lime", "red"), angles="xy",
              scale_units="xy", scale=1)
    ax.set_title("Affine fit residual = parallax, true length (green = RANSAC inlier)")
    ax.axis("off")
    fig.tight_layout(); save(fig, "4_residuals.png")

    if SHOW:
        plt.show()


if __name__ == "__main__":
    main()
