from matplotlib import pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import numpy as np
from nuscenes.utils.geometry_utils import view_points

nusc = NuScenes(version='v1.0-mini', dataroot='/Users/david/nuscenes-data', verbose=True)

print("List of Scenes (different 20s clips):")
nusc.list_scenes()

my_scene = nusc.scene[9]
print("Scene 9 Info:", my_scene)

# Examine metadata
first_sample_token = my_scene['first_sample_token']
my_sample = nusc.get('sample', first_sample_token)
print("First sample (annotated frame) of scene 9:", my_sample)
nusc.list_sample(my_sample['token'])

# Examine raw lidar data (numpy.ndarray of size 4 x number of points)
#sd_record = self.nusc.get('sample_data', sample_data_token)
print(my_sample['data'])
sensor = 'LIDAR_TOP'
lidar_data = nusc.get('sample_data', my_sample['data'][sensor])
chan = lidar_data['channel']
pc, times = LidarPointCloud.from_file_multisweep(nusc, my_sample, chan, sensor, nsweeps=1)
print("Type of pc:", type(pc), "Type of times:", type(times))
print("Size of pointcloud:", pc.points.shape)
print("First n pts of raw LiDAR pointcloud of first sample of scene 9:")
print("         [x (meters), y (meters), z (meters), intensity]")
n = 25873
n_print = 100
for i in range(n_print):
    print("Point", i, ": ", pc.points[:,i])

# Visualize lidar data
print("Visualization ----")
print("LiDAR data of first sample of scene 9:", lidar_data)
viewpoint = np.eye(4)
_, ax = plt.subplots(1, 1, figsize=(9, 9))
points = view_points(pc.points[:3, :n], viewpoint, normalize=False)
dists = np.sqrt(np.sum(pc.points[:2, :n] ** 2, axis=0))
axes_limit = 40
#colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
# Filter out every point below specified z-axis limit
points_to_delete = []
for i in range(len(points[0])):
    if points[2,i] < -.75:
        points_to_delete.append(i)
points = np.delete(points, points_to_delete, axis=1)
for i in range(n_print):
    print("Point", i, ": ", points[:,i])
colors = points[2, :n]
point_scale = 0.2
scatter = ax.scatter(points[0, :n], points[1, :n], c=colors, s=point_scale)
ax.plot(0, 0, 'x', color='red')
ax.set_aspect('equal')
ax.axis('off')
#cam_data = nusc.get('sample_data', my_sample['data']['FRONT_CAM'])
nusc.render_sample_data(lidar_data['token'])
plt.show()

index = 1
while index < 13:
    sample_token = my_sample['next']
    my_sample = nusc.get('sample', sample_token)
    index += 1
while index <= 27:
    print("Sample ", index)
    sample_token = my_sample['next']
    my_sample = nusc.get('sample', sample_token)
    lidar_data = nusc.get('sample_data', my_sample['data'][sensor])
    nusc.render_sample_data(lidar_data['token'])
    plt.show()
    index += 1