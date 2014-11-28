# Gaze segmentation using optimal segmented linear regression

Majorly WIP. Mostly for Teemu so he can work for integration.
To be continued.

Quick manual:
Compile the binary stuff by issuing `make`. After this
the (currently broken implementation of the) algorithm can be
run as:

```python
from fast_gaze_segmentation import nols
split_indices = nols(timestamps, gaze2d,
	split_rate=1.0/typical_pursuit_length,
	noise_std=np.array([x_noise_std, y_noise_std]))
```
