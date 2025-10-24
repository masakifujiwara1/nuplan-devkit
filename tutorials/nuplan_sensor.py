import os
from pathlib import Path
import tempfile
import hydra
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非対話型バックエンドを明示的に設定
import matplotlib.pyplot as plt

# from tutorials.utils.tutorial_utils

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario, CameraChannel, LidarChannel
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioExtractionInfo


# NUPLAN_DB_FILES="/home/ubuntu/nuplan/dataset/nuplan-v1.1/splits/mini" # nuplan training data path (e.g., "/data/nuplan-v1.1/trainval")
NUPLAN_MAPS_ROOT="/home/ubuntu/nuplan/dataset/maps"
NUPLAN_MAP_VERSION="nuplan-maps-v1.1"
NUPLAN_DATA_ROOT="/home/ubuntu/nuplan/dataset"
NUPLAN_SENSOR_ROOT = f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/sensor_blobs"
TEST_DB_FILE = f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini/2021.05.12.22.00.38_veh-35_01008_01518.db"
MAP_NAME = "us-nv-las-vegas"
TEST_INITIAL_LIDAR_PC = "58ccd3df9eab54a3"
TEST_INITIAL_TIMESTAMP = 1620858198150622

scenario = NuPlanScenario(
            data_root=f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini",
            log_file_load_path=TEST_DB_FILE,
            initial_lidar_token=TEST_INITIAL_LIDAR_PC,
            initial_lidar_timestamp=TEST_INITIAL_TIMESTAMP,
            scenario_type="scenario_type",
            map_root=NUPLAN_MAPS_ROOT,
            map_version=NUPLAN_MAP_VERSION,
            map_name=MAP_NAME,
            scenario_extraction_info=ScenarioExtractionInfo(
                scenario_name="scenario_name", scenario_duration=20, extraction_offset=1, subsample_ratio=0.5
            ),
            ego_vehicle_parameters=get_pacifica_parameters(),
            sensor_root=NUPLAN_DATA_ROOT+"/nuplan-v1.1/sensor_blobs",
)

# sensors = scenario.get_sensors_at_iteration(0, [CameraChannel.CAM_F0, LidarChannel.MERGED_PC])
sensors = scenario.get_sensors_at_iteration(0, [CameraChannel.CAM_F0])
print(sensors)

img = sensors.images[CameraChannel.CAM_F0]
plt.figure(figsize=(10, 8))
plt.imshow(img.as_numpy)
plt.title('Camera F0 Image')
plt.axis('off')
plt.tight_layout()
plt.savefig('camera_f0_image.png', dpi=150, bbox_inches='tight')
print("Camera F0 image saved as 'camera_f0_image.png'")
plt.close()

# Get LiDAR point cloud separately
sensors_with_pc = scenario.get_sensors_at_iteration(0, [LidarChannel.MERGED_PC])
if sensors_with_pc.pointcloud and LidarChannel.MERGED_PC in sensors_with_pc.pointcloud:
    pc = sensors_with_pc.pointcloud[LidarChannel.MERGED_PC]
    plt.figure(figsize=(10, 8))
    plt.imshow(pc.render_image())
    plt.title('LiDAR Point Cloud')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('lidar_pointcloud.png', dpi=150, bbox_inches='tight')
    print("LiDAR point cloud saved as 'lidar_pointcloud.png'")
    plt.close()
else:
    print("LiDAR point cloud not available")

# All camera channels
sensors = scenario.get_sensors_at_iteration(0, [channel for channel in CameraChannel])


# Visualizing all 8 channels
fig, axarr = plt.subplots(4, 2, figsize=(15, 20))
channel_idx = 0
for channel, img in sensors.images.items():
    row, col = channel_idx // 2, channel_idx % 2
    axarr[row][col].imshow(img.as_numpy)
    axarr[row][col].set_title(channel.name)
    axarr[row][col].axis('off')
    channel_idx += 1

plt.tight_layout()
plt.savefig('all_camera_channels.png', dpi=150, bbox_inches='tight')
print("All camera channels saved as 'all_camera_channels.png'")
plt.close()
