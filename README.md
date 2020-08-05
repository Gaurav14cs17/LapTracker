# WORK IN PROGRESS

# LapTracker
A simple linear assignment problem (LAP) object tracker that can handle gaps, splits and merges in object tracks. The tracking approach is inspired by: 

Jaqaman, K., Loerke, D., Mettlen, M. et al. Robust single-particle tracking in live-cell time-lapse sequences. Nat Methods 5, 695–702 (2008). https://doi.org/10.1038

However, some of the details were simplified, since this tracker is mainly supposed to be helpfull to track segmented objects (e.g. cells, nuclei) instead of single particles.

# Example #1: track objects from pandas DataFrame
There are at the moment two ways to use this tracker. The first one is to use it to track objects for which you have a pandas DataFrame of features. The DataFrame needs to at least include the x coordinates, y coordinates and timepoint. You can use the LapTracker in this case like this:

```python
from LapTracker import LapTracker
import numpy as np

columns = ['x_coordinates', 'y_coordinates', 'timepoints', 'labels']
df = np.random.choice(2000, [500, 4], replace=False)
df[:, 2] = np.repeat(np.arange(0, 50), 10)
df = pd.DataFrame(df, columns=columns)


tracker = LapTracker(max_distance=150, time_window=4, max_split_distance=150)
tracker.track_df(df, columns)
tracked_df = tracker.df
```

The df attribute of the tracker will be the same DataFrame with 3 additional columns: unique_id, segment_id and track_id. The track_id is the most important one, as it tells you which track the object is finally assigned to.

# Example #2: track objets from label image stack
In case you only have label images of your objects for each timepoint, you can use a different functionality of the tracker:

```python
from LapTracker import LapTracker
from skimage.io import imread

# load your image stack. should be t * x * y.
stack = imread('my_stack.tiff')
tracker = LapTracker(max_distance=150, time_window=4, max_split_distance=150)
tracker.track_label_images(stack)

```

After processing, the tracker object will have the attributes "relabeled_movie" (a numpy array in which the objects are relabeled according to their track id) and "df" (a pd.DataFrame containing the centroid measurements of the objects as well as unique_id, segment_id and track_id)
