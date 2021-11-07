import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def create_crop():
    print("Demo for manual geometry cropping")
    print("1) Press 'Y' twice to align geometry with negative direction of y-axis")
    print("2) Press 'K' to lock screen and to switch to selection mode")
    print("3) Drag for rectangle selection,")
    print("   or use ctrl + left click for polygon selection")
    print("4) Press 'C' to get a selected geometry and to save it")
    print("5) Press 'F' to switch to freeview mode")
    pcd = o3d.io.read_point_cloud(PATH)  
    o3d.visualization.draw_geometries_with_editing([pcd])

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], zoom=0.2, front=[0, 0, 1], lookat=[1.5, -2, 0.2], up=[0, 1, 0])

if __name__ =="__main__":
    PATH = "./data/0000000000.pcd"
    pcd = o3d.io.read_point_cloud(PATH)

    #-----------#
    # Filtering #
    #-----------#
    # – Downsampling
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.1)

    # – Cropping
    vol = o3d.visualization.read_selection_polygon_volume("z-crop.json")
    cropped_pcd = vol.crop_point_cloud(downsampled_pcd)

    #Remove statistical outliers
    cl, ind = cropped_pcd.remove_statistical_outlier(nb_neighbors=200, std_ratio=1.3)
    statistical_inliers = cropped_pcd.select_by_index(ind)

    #Remove radius outliers
    cl, ind = statistical_inliers.remove_radius_outlier(nb_points=12, radius=1.5)
    radial_inliers = statistical_inliers.select_by_index(ind)

    #--------------#
    # Segmentation #
    #--------------#
    plane_model, inliers = radial_inliers.segment_plane(distance_threshold=0.25,
                                            ransac_n=250,
                                            num_iterations=2000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    road_cloud = radial_inliers.select_by_index(inliers)
    road_cloud.paint_uniform_color([1.0, 0, 0])
    object_cloud = radial_inliers.select_by_index(inliers, invert=True)
    object_cloud.paint_uniform_color([0, 1.0, 0])

    #------------#
    # Clustering #
    #------------#
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(object_cloud.cluster_dbscan(eps=0.75, min_points=20, print_progress=True))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    object_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

    #------------------------#
    # Visualise the PCD file #
    #------------------------#
    o3d.visualization.draw_geometries([object_cloud], zoom=0.1, front=[0, 0, 1], lookat=[1.5, -2, 0.2], up=[0, 1, 0])