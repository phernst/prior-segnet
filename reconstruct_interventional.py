from os.path import join as pjoin
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch_radon import ConeBeam
from torch_radon.volumes import Volume3D
from torch_radon.filtering import FourierFilters

from ict_system import ArtisQSystem, DetectorBinning

CARMH_GT_UPPER_99_PERCENTILE: float = 1720.43359375


def apply_noise(projections: torch.Tensor, photon_flux: int):
    i0 = photon_flux
    intensity = i0*(-projections).exp()
    noisy_intensity = torch.max(torch.poisson(intensity), torch.ones_like(intensity))
    return -(noisy_intensity/i0).log()


def create_network_data(vol_ds: Tuple[np.ndarray, Tuple, Tuple], needle_projections: np.ndarray,
                        prior_ds: torch.Tensor, photon_flux: Optional[int] = None):
    '''
    vol_ds: ([w, h, 360], [3], [3]) tensor, voxel_spacing, volume_shape
    ndl_ds: [u, v, 360]
    prior_ds: [d, h, w, 1]
    '''
    combined = _modify_shape_to_z_fov(
        vol_ds[0], vol_ds[1], vol_ds[2], needle_projections
    )

    reco_volume = _reconstruct_3d(*combined, photon_flux)  # [d, h, w, 2]

    prior_ds = prior_ds.to(reco_volume.device)
    reco_volume, prior_ds = _equalize_z_dimensions(reco_volume, prior_ds)

    full_data = _create_gt_from_tensors(reco_volume, prior_ds)
    full_data = mu2hu(full_data)
    full_data = hu2mu(full_data)/hu2mu(CARMH_GT_UPPER_99_PERCENTILE)

    return full_data


# in/out: [u, v, 360], [3], [3], [u, v, 360]
def _modify_shape_to_z_fov(vol_projections: np.ndarray,
                           voxel_spacing: np.ndarray,
                           volume_shape: np.ndarray,
                           needle_projections: np.ndarray) \
                           -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    field_of_view = 200.  # FOV in z direction, 200mm
    num_fov_slices = int(field_of_view//voxel_spacing[2])
    num_fov_slices += (num_fov_slices - volume_shape[0]) % 2
    volume_shape[0] = min(volume_shape[0], num_fov_slices)
    return vol_projections, voxel_spacing, volume_shape, needle_projections


def _reconstruct_3d(vol_projections, voxel_size, volume_shape, ndl_projections,
                    photon_flux):
    full_radon = _create_radon(360)
    sparse_radon = _create_radon(13)

    num_sparse_projections = len(sparse_radon.angles)
    voxel_dims = (384, 384, volume_shape[0])

    # create interventional projections
    vol_ndl_projections = vol_projections + ndl_projections
    if photon_flux is not None:
        vol_ndl_projections = apply_noise(vol_ndl_projections, photon_flux)

    # create reconstruction of interventional volume w/ all projections
    full_needle = _reconstruct_volume_from_projections(
         ndl_projections, full_radon, voxel_dims, voxel_size)

    # create reconstruction of interventional volume w/ sparse projections
    sparse_with_needle = _reconstruct_volume_from_projections(
        vol_ndl_projections[...,
        ::(vol_ndl_projections.shape[-1] // num_sparse_projections + num_sparse_projections % 2)],
        sparse_radon, voxel_dims, voxel_size)

    return torch.stack((sparse_with_needle, full_needle), -1)


def _create_radon(num_views: int) -> ConeBeam:
    ct_system = ArtisQSystem(DetectorBinning.BINNING4x4)
    angles = np.linspace(0, 2*np.pi, 360, endpoint=False, dtype=np.float32)[::(360 // num_views + num_views % 2)]
    src_dist = ct_system.carm_span*4/6
    det_dist = ct_system.carm_span*2/6
    # src_det_dist = src_dist + det_dist
    det_spacing_v = ct_system.pixel_dims[1]
    return ConeBeam(
        det_count_u=ct_system.nb_pixels[0],
        angles=angles,
        src_dist=src_dist,
        det_dist=det_dist,
        det_count_v=ct_system.nb_pixels[1],
        det_spacing_u=ct_system.pixel_dims[0],
        det_spacing_v=det_spacing_v,
        pitch=0.0,
        base_z=0.0,
    )


def _reconstruct_volume_from_projections(projections: torch.Tensor, radon: ConeBeam,
                                         voxel_dims: Tuple[int, int, int],
                                         voxel_size: Tuple[float, float, float]) \
                                         -> np.ndarray:
    """
    returns reconstruction of input projections in HU, [x, y, z]
    """
    assert len(radon.angles) == projections.shape[-1]
    radon.volume = Volume3D(
            depth=voxel_dims[2],
            height=voxel_dims[1],
            width=voxel_dims[0],
            voxel_size=voxel_size)

    det_spacing_v = radon.projection.cfg.det_spacing_v
    src_dist = radon.projection.cfg.s_dist
    det_dist = radon.projection.cfg.d_dist
    src_det_dist = src_dist + det_dist

    with torch.inference_mode():
        projections_t = projections.float().cuda().permute(-1, -2, -3)
        projections_t = projections_t[None, None, ...]
        filtered_projections_t = _filter_sinogram_3d(projections_t, 'hann')
        reco_t = radon.backprojection(filtered_projections_t)
        reco_t = reco_t*det_spacing_v/src_det_dist*src_dist  # cone beam correction
        # reco_t = mu2hu(reco_t, 0.02)
        reco = reco_t[0, 0]
    return reco


def _filter_sinogram_3d(sinogram: torch.Tensor, filter_name="ramp"):
    fourier_filters = FourierFilters()
    sinogram = sinogram.permute(0, 1, 3, 2, 4)
    sino_shape = sinogram.shape
    sinogram = sinogram.reshape(np.prod(sino_shape[:-2]), sino_shape[-2], sino_shape[-1])
    size = sinogram.size(-1)
    n_angles = sinogram.size(-2)

    # Pad sinogram to improve accuracy
    padded_size = max(64, int(2 ** np.ceil(np.log2(2 * size))))
    pad = padded_size - size
    padded_sinogram = F.pad(sinogram.float(), (0, pad, 0, 0))

    sino_fft = torch.fft.rfft(padded_sinogram)

    # get filter and apply
    f = fourier_filters.get(padded_size, filter_name, sinogram.device)[..., 0]
    filtered_sino_fft = sino_fft * f

    # Inverse fft
    filtered_sinogram = torch.fft.irfft(filtered_sino_fft)
    filtered_sinogram = filtered_sinogram[:, :, :-pad] * (np.pi / (2 * n_angles))

    return filtered_sinogram.to(dtype=sinogram.dtype).reshape(sino_shape).permute(0, 1, 3, 2, 4)


def _equalize_z_dimensions(reco_volume: torch.Tensor,
                           prior_volume: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if prior_volume.shape[0] == reco_volume.shape[0]:
        return reco_volume, prior_volume
    zdiff = (prior_volume.shape[0] - reco_volume.shape[0])//2
    return reco_volume, prior_volume[zdiff:-zdiff]


def _create_gt_from_tensors(recos: torch.Tensor,
                            prior: torch.Tensor) -> torch.Tensor:
    reco_sparse, needle_full = recos[..., 0], recos[..., 1]
    full_reco = needle_full + prior[..., 0]
    return torch.cat((torch.stack([reco_sparse, full_reco, needle_full], -1), prior), dim=-1)


def mu2hu(volume: torch.Tensor, mu_water: float = 0.02) -> torch.Tensor:
    return (volume - mu_water)/mu_water * 1000


def hu2mu(volume: torch.Tensor, mu_water: float = 0.02) -> torch.Tensor:
    return (volume * mu_water)/1000 + mu_water


def test_create_network_data():
    def load_nifty(path: str):
        nii_file = nib.load(path)
        volume = nii_file.get_fdata().transpose()
        voxel_size = np.diag(nii_file.affine)[:-1]
        return volume, voxel_size

    volume, voxel_size_vol = load_nifty(
        pjoin('/mnt/nvme2/lungs/lungs3d', 'R_274.nii.gz'))
    volume = hu2mu(volume)

    full_radon = _create_radon(360)
    full_radon.volume = Volume3D(
        depth=volume.shape[0],
        height=volume.shape[1],
        width=volume.shape[2],
        voxel_size=voxel_size_vol,
    )

    print(f'{volume.shape=}, {voxel_size_vol=}')

    vol_projections = full_radon.forward(torch.from_numpy(volume)[None, None].float().cuda())
    vol_projections = vol_projections.nan_to_num().cpu().numpy()[0, 0].transpose()
    print(f'{vol_projections.shape=}')

    from matplotlib import pyplot as plt
    # plt.imshow(vol_projections[..., 0])
    # plt.show()

    # create needle projections
    ndl_volume, voxel_size = load_nifty('/home/phernst/Documents/git/ictdl/needles/Needle2_Pos2_12.nii.gz')
    volume_t = torch.from_numpy(ndl_volume).float().cuda()[None, None, ...]
    volume_t = hu2mu(volume_t)
    full_radon.volume = Volume3D(
            depth=ndl_volume.shape[0],
            height=ndl_volume.shape[1],
            width=ndl_volume.shape[2],
            voxel_size=voxel_size)
    needle_projections = full_radon.forward(volume_t).nan_to_num().cpu().numpy()[0, 0].transpose()
    print(f'{needle_projections.shape=}')

    # plt.imshow(needle_projections[..., 0])
    # plt.show()

    # load prior
    prior_ds = torch.load('/mnt/nvme2/lungs/lungs3d/priors/R_274.pt')['volume'][..., None]

    volume_shape = list(volume.shape)
    combined = create_network_data(
        (vol_projections, voxel_size_vol, volume_shape),
        needle_projections,
        prior_ds)

    print(f'{combined.shape=}')
    plt.imshow(combined[20, ..., 0].cpu().numpy(), vmax=0.04)
    plt.figure()
    plt.imshow(combined[20, ..., 1].cpu().numpy(), vmax=0.04)
    plt.show()


if __name__ == '__main__':
    test_create_network_data()
