import os
import sys

parentPath = os.path.abspath('..')
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)

from visualize import *


def draw_masks(image, masks):
    # Number of instances
    N = masks.shape[-1]

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    # plt.set_ylim(height + 10, -10)
    # plt.set_xlim(-10, width + 10)

    vertices = []

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = np.array(colors[i])

        # Mask
        mask = masks[:, :, i]
        masked_image = apply_mask(masked_image, mask, color)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            vertices.append((verts, color))
    return masked_image.astype(np.uint8), vertices
