import ctl

from ict_system import ArtisQSystem, DetectorBinning, LungsGESystem


def main(tube_voltage: float, mas: float):
    artisq_params = ArtisQSystem(DetectorBinning.BINNING4x4)
    lungsge_params = LungsGESystem()

    artisq_system = ctl.SimpleCTSystem(
        detector=ctl.FlatPanelDetector(
            artisq_params.nb_pixels,
            artisq_params.pixel_dims,
        ),
        gantry=ctl.TubularGantry(
            artisq_params.carm_span,
            artisq_params.carm_span*2/3,
        ),
        source=ctl.XrayTube(
            tube_voltage=tube_voltage,
            mas=mas,
        )
    )

    artisq_encoder = ctl.RadiationEncoder(artisq_system)
    print(f'{artisq_encoder.photons_per_pixel_mean()=}')

    nb_projections_per_second: int = 1000  # probably different
    nb_projections: int = nb_projections_per_second*lungsge_params.exposure_time/1000
    lungsge_system = ctl.SimpleCTSystem(
        detector=ctl.FlatPanelDetector(  # actually cylindrical
            (lungsge_params.detector_columns, lungsge_params.detector_rows),
            (lungsge_params.pixel_size, lungsge_params.pixel_size),
        ),
        gantry=ctl.TubularGantry(
            lungsge_params.source_to_detector,
            lungsge_params.source_to_isocenter,
        ),
        source=ctl.XrayTube(
            focal_spot_size=(lungsge_params.focal_spot_size,)*2,
            tube_voltage=lungsge_params.voltage,
            mas=lungsge_params.exposure/nb_projections,
        )
    )

    lungsge_encoder = ctl.RadiationEncoder(lungsge_system)
    print(f'{lungsge_encoder.photons_per_pixel_mean()=}')


if __name__ == '__main__':
    main(120.0, .02)
