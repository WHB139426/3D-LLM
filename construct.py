import os
import glob
import numpy as np
import cv2
import open3d as o3d
import argparse
import random

def load_intrinsics(intrinsic_path):
    K = np.loadtxt(intrinsic_path)
    if K.shape == (4, 4):
        K = K[:3, :3]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return fx, fy, cx, cy

def load_extrinsics(extrinsic_path):
    T = np.loadtxt(extrinsic_path)
    if T.shape == (3, 4):
        # 有些外参只有3x4，需要补成4x4
        T = np.vstack([T, [0, 0, 0, 1]])
    return T

def depth_to_points(depth, fx, fy, cx, cy):
    """反投影为相机坐标"""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1)

def generate_point_cloud(scene_dir, output_name="scene.ply"):
    # 读取内参
    intrinsic_path = os.path.join(scene_dir, "intrinsic.txt")
    fx, fy, cx, cy = load_intrinsics(intrinsic_path)
    print(f"Loaded intrinsics from {intrinsic_path}")

    # 找到所有帧文件
    depth_files = sorted(glob.glob(os.path.join(scene_dir, "*.png")))
    rgb_files   = sorted(glob.glob(os.path.join(scene_dir, "*.jpg")))
    extr_files  = sorted(glob.glob(os.path.join(scene_dir, "[0-9]*.txt")))
    extr_files = [f for f in extr_files if not f.endswith("intrinsic.txt")]

    print(f"Found {len(depth_files)} depth, {len(rgb_files)} rgb, {len(extr_files)} extrinsics")

    # 随机取32张（若不足32张就取全部）
    target_imgs = len(rgb_files)
    num_samples = min(target_imgs, len(rgb_files))
    indices = random.sample(range(len(rgb_files)), num_samples)

    depth_files = [depth_files[i] for i in indices]
    rgb_files   = [rgb_files[i] for i in indices]
    extr_files  = [extr_files[i] for i in indices]
    print(f"Random get {target_imgs} images")

    all_points = []
    all_colors = []

    for i, depth_path in enumerate(depth_files):
        idx = os.path.splitext(os.path.basename(depth_path))[0]  # e.g. '00000'
        rgb_path = os.path.join(scene_dir, f"{idx}.jpg")
        extr_path = os.path.join(scene_dir, f"{idx}.txt")

        if not (os.path.exists(rgb_path) and os.path.exists(extr_path)):
            print(f"⚠️ Skipping frame {idx}: missing rgb or extrinsic")
            continue

        # ---- 读取图像 ----
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        rgb   = cv2.imread(rgb_path, cv2.IMREAD_COLOR)[:, :, ::-1]  # BGR→RGB

        # 若深度为毫米，则换算为米
        depth[depth == 0] = np.nan
        depth = depth / 1000.0

        # 对齐尺寸
        if rgb.shape[:2] != depth.shape:
            rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)

        # ---- 生成点云（相机坐标）----
        pts_cam = depth_to_points(depth, fx, fy, cx, cy)
        valid = ~np.isnan(pts_cam[..., 2])
        pts_cam = pts_cam[valid]
        colors = rgb[valid]
        colors = colors.astype(np.float32) / 255.0

        # ---- 外参变换 ----
        T = load_extrinsics(extr_path)
        ones = np.ones((pts_cam.shape[0], 1), dtype=np.float32)
        pts_h = np.concatenate([pts_cam, ones], axis=1)
        pts_world = (T @ pts_h.T).T[:, :3]

        # ---- 累加 ----
        all_points.append(pts_world)
        all_colors.append(colors)

        if (i + 1) % 20 == 0:
            print(f"Processed {i+1}/{len(depth_files)} frames")

    # ---- 合并并保存 ----
    if len(all_points) == 0:
        print("❌ No valid frames found.")
        return

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)

    print(f"Total raw points: {len(points):,}")

    # ==== 随机或均匀采样 ====
    target_n = 81369
    if len(points) > target_n:
        # 均匀采样（如果希望随机，可改成 np.random.choice）
        # idx = np.linspace(0, len(points) - 1, target_n).astype(int)
        idx = np.random.choice(len(points), target_n, replace=False)  # 随机采样
        points = points[idx]
        colors = colors[idx]
        print(f"Sampled {target_n} points.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    output_path = output_name
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"✅ Saved full-scene point cloud to {output_path}")
    print(f"Total points: {len(points):,}")

    # 判断并打印文件大小
    file_size = os.path.getsize(output_path)
    if file_size < 1024:
        print(f"✅ 文件已保存：{output_path} ({file_size:.2f} Bytes)")
    elif file_size < 1024 ** 2:
        print(f"✅ 文件已保存：{output_path} ({file_size / 1024:.2f} KB)")
    else:
        print(f"✅ 文件已保存：{output_path} ({file_size / 1024 ** 2:.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=str, default='/home/haibo/haibo_workspace/data/scannet-frames/scene0000_00', help="path to scene folder (e.g., scene0000_00)")
    parser.add_argument("--output", type=str, default="test.ply", help="output ply filename")
    args = parser.parse_args()

    generate_point_cloud(args.scene_dir, args.output)
