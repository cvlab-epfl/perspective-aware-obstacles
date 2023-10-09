
## Horizon level

### From provided calibration

Datasets Cityscapes and Lost&Found provide camera extrinsics for download (`camera.zip`). 
We use the specified pitch angle to determine horizon level.

Implementation in `cityscapes_pitch_angles.py`.
Demo:

```python
from src.a14_perspective.cityscapes_pitch_angles import show_frame_with_horizon

dset_laf = DatasetRegistry.get_implementation('LostAndFound-train')
dset_laf.dir_root = dset_laf.dset.DIR_LAF # for WrapWP

show_frame_with_horizon(dset_laf, 600)
```

Extract horizon:
```python
from src.a14_perspective.cityscapes_pitch_angles import read_cam_info, read_cam_info_laf, perspective_info_from_camera_info

def get_horizon_from_extrinsics(dset, idx_or_fid):
	fr = dset[idx_or_fid]
	try:
		cam_info = read_cam_info(dset, fr.fid)
	except:
		cam_info = read_cam_info_laf(dset, fr)

	fr.camera_info = cam_info
	fr.persp_info = perspective_info_from_camera_info(cam_info)

	print(fr.persp_info.horizon_level)

	return fr

```
