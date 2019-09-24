import os.path as osp
config = {

    "AGIPD":dict(
        pixel_size=0.2e-3,
        source_name=["SPB_DET_AGIPD1M-1/DET/detector"],
        mask_rng=[0, 3500],
        geom_file=osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))),
                                      'geometry/agipd.geom'),
        run_folder='/Users/ebadkamil/spb-data',
        ),

    "TIME_OUT":1.,
    }