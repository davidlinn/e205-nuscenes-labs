from matplotlib import pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import numpy as np
import os.path as osp
from nuscenes.utils.geometry_utils import view_points
from typing import List, Dict, Tuple
from pyquaternion import Quaternion

nusc = NuScenes(version='v1.0-mini', dataroot='/Users/david/nuscenes-data', verbose=True)

# Scene - a 20-second data selection - we've selected scene 9
# Sample - fully annotated dataframe (every .5 seconds) - we're selecting samples 13-27
# Token - a hash that corresponds to data in a hashmap - get data from hashmap with nusc.get('MAP_NAME', token)
# This function does not return tokens, but returns list of actual samples (which are of type dict)
def getSampleSet1() -> List[Dict]:
    samples = []
    my_scene = nusc.scene[9]
    sample_token = my_scene['first_sample_token']
    my_sample = nusc.get('sample', sample_token)
    index = 1
    while index < 13:
        sample_token = my_sample['next']
        my_sample = nusc.get('sample', sample_token)
        index += 1
    while index <= 27:
        sample_token = my_sample['next']
        my_sample = nusc.get('sample', sample_token)
        samples.append(my_sample)
        index = index + 1
    return samples

# Maps 3D lidar to 2D and renders each sample in samples
def renderSampleLidar(samples: List[Dict]):
    if isinstance(samples, list):
        index = 0
        for my_sample in samples:
            lidar_data = nusc.get('sample_data', my_sample['data']['LIDAR_TOP'])
            nusc.render_sample_data(lidar_data['token'])
            fig = plt.gcf()
            fig.canvas.set_window_title('Sample ' + str(index))
            plt.show(block = True)
            index += 1
    else:
        lidar_data = nusc.get('sample_data', samples['data']['LIDAR_TOP'])
        nusc.render_sample_data(lidar_data['token'])
        plt.show()

# Lidar points are colored and rendered within a camera frame
def renderSampleLidarInCam(samples: List[Dict]):
    if isinstance(samples, list):
        index = 0
        for my_sample in samples:
            nusc.render_pointcloud_in_image(my_sample['token'])
            fig = plt.gcf()
            fig.canvas.set_window_title('Sample ' + str(index))
            plt.show(block = True)
            index += 1
    else:
        nusc.render_pointcloud_in_image(samples['token'])
        plt.show()

# Annotation - any labeled object; contains a cuboid 3D bounding box and category (car, bicycle)
# We examine three annotations from this scene
def getFirstAnnotationTokens(samples: List[Dict]) -> List[str]:
    ann_tokens = []
    ann_tokens.append(samples[0]['anns'][2]) # front car
    ann_tokens.append(samples[0]['anns'][5]) # biker
    ann_tokens.append(samples[0]['anns'][15]) # second car
    return ann_tokens

def getNextAnnotationToken(token: str) -> str:
    curr_ann = nusc.get('sample_annotation', token)
    return curr_ann['next']

# Return list of lidar data between sample annotations (9 lidar sweeps in between each sample)
# Input: single sample (actual data, not token) / Output: list of lidar data (actual data, not token)
# This data is unannotated -- let's see your Kalman / Particle filter in action here!!
def lidarDataBetweenAnnotations(startSample: Dict) -> List:
    nextSample = nusc.get('sample', startSample['next'])
    currSampleLidarToken = startSample['data']['LIDAR_TOP']
    currLidarData = nusc.get('sample_data', currSampleLidarToken)
    currSampleLidarToken = currLidarData['next']
    nextSampleLidarToken = nextSample['data']['LIDAR_TOP']
    lidarData = []
    while currSampleLidarToken != nextSampleLidarToken:
        currLidarData = nusc.get('sample_data', currSampleLidarToken)
        lidarData.append(currLidarData)
        currSampleLidarToken = currLidarData['next']
    return lidarData

# Input: lidar datum, optionally list of custom bounding boxes from your filter
def renderNonsampleLidar(lidarDatum: Dict, customAnns=[]):
    # Plot all lidar data
    fig, axes = plt.subplots(1, 1, figsize=(9, 9))
    view: np.ndarray = np.eye(4)
    data_path = osp.join(nusc.dataroot,lidarDatum['filename'])
    LidarPointCloud.from_file(data_path).render_height(axes, view=view)
    # Limit visible range.
    axes.set_xlim(-40, 40)
    axes.set_ylim(-40, 40)
    # Plot state estimates

    plt.show(block=True)

# SI units (meters, m/s, radians from +x-axis, radians/s)
# Pos: [X,Y], Vel: [X,Y], Theta: radians, AngVel: radians/s, Size
class StateEstimate:
    def __init__(self, pos: np.ndarray, vel: np.ndarray, theta: float, angVel: float, size: List[float]):
        self.pos = pos
        self.vel = vel
        self.theta = theta
        self.angVel = angVel
        self.size = size

def annTokensToObjEsts(tokens: List[str]) -> List[StateEstimate]:
    objEsts = []
    for token in tokens:
        ann = nusc.get('sample_annotation', token)
        rotation = Quaternion(ann['rotation'])
        heading = rotation.yaw_pitch_roll[0]
        objEsts.append(StateEstimate(np.array(ann['translation'][0:2]),np.array([0,0]),heading,0,ann['size'][0:2]))
    return objEsts

# Input: starting sample token, optional annotations to start with (simple starts with
# tracking only 3 objects)
def testFilter(simple=True):
    timeBetweenFrames = .05
    samples = getSampleSet1()
    # If given starting annotation tokens, convert to starting object estimates
    if simple:
        startAnnTokens = getFirstAnnotationTokens(samples)
        objEstList = annTokensToObjEsts(startAnnTokens)
    else:
        objEstList = []
    
    for sample in samples:
        # Perform object correlation and filter update
        correlateObjsKalman(objEstList, sample)
        # Render sample with all annotations
        renderSampleLidar(sample)
        # Project and render all lidar data until next sample
        lidarData = lidarDataBetweenAnnotations(sample)
        timeSinceUpdate = .05
        for lidarDatum in lidarData:
            boundingBoxList = projectStateEstimates(objEstList, timeSinceUpdate)
            renderNonsampleLidar(lidarDatum, boundingBoxList)
            timeSinceUpdate += timeBetweenFrames
        


# David's Code - Copying the below is an Honor Code Violation

# Input: Previous list of obj estimates, time since last update for box projection (seconds)
# Output: List of state estimates
# Projects estimates over time assuming constant velocity.
def projectStateEstimates(objEstList: List[StateEstimate], projectionTime: float) -> List[StateEstimate]:
    for obj in objEstList:
        obj.pos += obj.vel * projectionTime
        obj.theta += obj.angVel * projectionTime
    return objEstList

# Input: Previous list of object estimates, New annotated sample
# Output: List of object estimates
# Correlate by minimum 2D Euclidean distance
# Object estimate input should already be projected to current time
def correlateObjsKalman(objEstList: List[StateEstimate], sample: Dict) -> List[StateEstimate]:
    pass
