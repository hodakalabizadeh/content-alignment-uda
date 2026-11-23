import torch
import torch.nn as nn
import torch.nn.functional as nnf


class SpatialTransformer(nn.Module):
    """Warp an image or label with a dense displacement field.

    Each output voxel is sampled from the input at (its own position + displacement), so a
    zero field leaves the input unchanged. Displacements are in voxels, with one channel per
    spatial axis in (H, W[, D]) order.

    Args:
        size: spatial size of the inputs, (H, W) or (H, W, D).
        mode: 'bilinear' (images, soft labels) or 'nearest' (hard labels).
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode

        # identity grid (1, ndims, *size): the voxel coordinates of every location.
        # It is a buffer so it follows the module to the GPU; it is also a key in the
        # saved checkpoints, so it has to stay.
        vectors = [torch.arange(0, s) for s in size]
        grid = torch.stack(torch.meshgrid(vectors, indexing='ij')).unsqueeze(0).float()
        self.register_buffer('grid', grid)

    def forward(self, source, flow):
        """source: (B, C, *size) to be warped; flow: (B, ndims, *size) displacement in voxels."""
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # grid_sample expects coordinates in [-1, 1] ...
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # ... with the coordinate axis last and in reversed (x, y[, z]) order
        new_locs = new_locs.movedim(1, -1)
        new_locs = new_locs[..., list(reversed(range(len(shape))))]

        return nnf.grid_sample(source, new_locs, align_corners=True, mode=self.mode)


class VecIntegration(nn.Module):
    """Integrate a stationary velocity field into a diffeomorphic displacement field.

    Scaling and squaring: the field is scaled down by 2**nb_steps, then composed with
    itself nb_steps times (v <- v + v o v).

    Args:
        inshape: spatial size of the field, (H, W) or (H, W, D).
        nb_steps: number of squaring steps; 0 returns the field unchanged.
    """

    def __init__(self, inshape, nb_steps):
        super().__init__()
        self.nb_steps = nb_steps
        self.scale = 1.0 / (2 ** nb_steps)
        self.stn = SpatialTransformer(inshape)

    def forward(self, v):
        v = v * self.scale
        for _ in range(self.nb_steps):
            v = v + self.stn(v, v)
        return v
