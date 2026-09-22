"""
cad_cropper.py
CAD 引导的扫描点云裁剪/分割模块
职责: 当输入点云包含大量背景（桌面、夹具、环境）时，利用 CAD 模型尺寸作为先验，
      自动提取与被测零件最匹配的密集点云簇，提高后续平面检测与 ICP 的稳定性。
"""

import logging
from pathlib import Path
from typing import Union

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from cad_loader import CADLoader


def crop_scan_to_cad_region(
    scan_pcd: o3d.geometry.PointCloud,
    stl_path: Union[str, Path],
    margin_mm: float = 20.0,
    voxel_mm: float = 3.0,
    eps_factor: float = 2.0,
    min_points_factor: float = 0.005,
) -> o3d.geometry.PointCloud:
    """
    将扫描点云裁剪到 CAD 模型所在的近似区域。

    算法:
    1. 加载 CAD 并计算其包围盒与对角线
    2. 用 bbox 对角线比值估算 scan → CAD 的初始比例尺 s0
    3. 将 scan 缩放到近似 mm 空间并体素降采样
    4. DBSCAN 聚类，分离场景中的不同物体
    5. 对每个聚类计算 OBB 尺寸，与 CAD OBB 尺寸比较，选择最相似的簇
    6. 以 margin_mm 为缓冲，将匹配簇附近的原始点云保留下来

    :param scan_pcd: 预处理后的扫描点云（虚拟单位）
    :param stl_path: CAD 模型 (STL) 路径
    :param margin_mm: 裁剪缓冲边距（物理 mm，默认 20）
    :param voxel_mm: 聚类前降采样体素大小（物理 mm，默认 3）
    :param eps_factor: DBSCAN 邻域半径 = voxel_mm * eps_factor（默认 2.0 ≈ 6mm）
    :param min_points_factor: 最小聚类点数 = max(30, 总点数 * min_points_factor)（默认 0.005）
    :return: 裁剪后的点云（仍保持原始虚拟单位坐标系）
    """
    logger = logging.getLogger("PointToCAD_System.CADCropper")
    stl_path = Path(stl_path)

    if scan_pcd.is_empty():
        return scan_pcd

    # 1. 加载 CAD
    cad_loader = CADLoader(num_points=20000)
    cad_pcd = cad_loader.load(str(stl_path))
    cad_pts = np.asarray(cad_pcd.points)
    cad_obb = cad_pcd.get_oriented_bounding_box()
    cad_extent = np.sort(cad_obb.extent)
    cad_volume = float(np.prod(cad_extent))
    cad_diag = float(np.linalg.norm(cad_extent))
    logger.info(
        f"[CAD Cropper] CAD OBB 尺寸: {cad_extent[0]:.2f} x {cad_extent[1]:.2f} x {cad_extent[2]:.2f} mm, "
        f"体积={cad_volume:.1f} mm³"
    )

    # 2. 初始比例尺（仅用于把 scan 拉到近似 mm 空间做聚类）
    scan_bbox = scan_pcd.get_axis_aligned_bounding_box()
    scan_diag = float(np.linalg.norm(scan_bbox.get_extent()))
    if scan_diag < 1e-12:
        logger.warning("[CAD Cropper] scan 包围盒退化，跳过裁剪")
        return scan_pcd
    s0 = cad_diag / scan_diag
    logger.info(f"[CAD Cropper] bbox 比例尺初值 s0={s0:.4f}")

    # 3. 缩放并降采样
    scan_scaled = o3d.geometry.PointCloud(scan_pcd)
    scan_scaled.scale(s0, center=(0.0, 0.0, 0.0))
    scan_down = scan_scaled.voxel_down_sample(voxel_mm)
    n_down = len(scan_down.points)
    if n_down < 100:
        logger.warning(f"[CAD Cropper] 降采样后点数过少 ({n_down})，跳过裁剪")
        return scan_pcd
    logger.info(f"[CAD Cropper] 聚类前降采样: {len(scan_scaled.points)} -> {n_down} 点")

    # 4. DBSCAN 聚类
    eps = voxel_mm * eps_factor
    min_points = max(30, int(n_down * min_points_factor))
    labels = np.array(scan_down.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    max_label = int(labels.max())
    n_clusters = max_label + 1
    noise_count = int((labels == -1).sum())
    logger.info(f"[CAD Cropper] DBSCAN 聚类: {n_clusters} 个簇, noise={noise_count}")

    if n_clusters == 0:
        logger.warning("[CAD Cropper] 未找到任何聚类，跳过裁剪")
        return scan_pcd

    # 5. 选择 OBB 尺寸与点数均最接近 CAD 的簇
    pts_down = np.asarray(scan_down.points)
    best_score = float("inf")
    best_indices_down = None

    # 根据 CAD 表面积估算该物体在 voxel_mm 分辨率下应有的点数
    l, w, h = cad_extent
    cad_surface_area = 2.0 * (l * w + l * h + w * h)
    expected_points = max(100.0, cad_surface_area / (voxel_mm ** 2) * 0.3)
    min_cluster_points = max(min_points, int(expected_points * 0.4))
    logger.info(f"[CAD Cropper] 预期目标点数 ≈ {expected_points:.0f}, 最小聚类点数 = {min_cluster_points}")

    for label in range(n_clusters):
        mask = labels == label
        cluster_pts = pts_down[mask]
        n_cluster = len(cluster_pts)
        if n_cluster < min_cluster_points:
            continue

        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster_pts)
        try:
            obb = cluster_pcd.get_oriented_bounding_box()
            cluster_extent = np.sort(obb.extent)
        except RuntimeError:
            # 完美共面等退化情况
            continue

        cluster_volume = float(np.prod(cluster_extent))
        if cluster_volume <= 0 or cad_volume <= 0:
            continue

        # 评分：尺寸差异 + 体积差异 + 点数差异（均为相对值）
        extent_diff = np.mean(np.abs(cluster_extent - cad_extent) / (cad_extent + 1e-9))
        volume_diff = abs(cluster_volume - cad_volume) / (cad_volume + 1e-9)
        size_diff = abs(n_cluster - expected_points) / expected_points
        score = extent_diff + 0.3 * volume_diff + 0.2 * size_diff

        logger.debug(
            f"[CAD Cropper] 簇 {label}: 点数={n_cluster}, "
            f"OBB={cluster_extent[0]:.1f}x{cluster_extent[1]:.1f}x{cluster_extent[2]:.1f}, "
            f"score={score:.3f}"
        )

        if score < best_score:
            best_score = score
            best_indices_down = np.where(mask)[0]

    if best_indices_down is None or len(best_indices_down) < 50:
        logger.warning("[CAD Cropper] 未找到与 CAD 匹配的簇，跳过裁剪")
        return scan_pcd

    selected_down = scan_down.select_by_index(best_indices_down.tolist())
    logger.info(f"[CAD Cropper] 最佳匹配簇: {len(best_indices_down)} 点, score={best_score:.3f}")

    # 6. 在原始（已缩放）点云中，保留匹配簇 margin_mm 范围内的点
    tree = cKDTree(np.asarray(selected_down.points))
    scan_scaled_pts = np.asarray(scan_scaled.points)
    dists, _ = tree.query(scan_scaled_pts, k=1)
    in_region = dists <= margin_mm
    n_keep = int(in_region.sum())
    n_total = len(scan_scaled_pts)
    logger.info(f"[CAD Cropper] 保留 {n_keep}/{n_total} 点 ({n_keep / n_total * 100:.1f}%)")

    if n_keep < 100:
        logger.warning("[CAD Cropper] 裁剪后点数过少，使用原始点云")
        return scan_pcd

    # 通过索引选择原始点云中的点，避免 scale 往返带来的数值误差
    cropped = scan_pcd.select_by_index(np.where(in_region)[0].tolist())
    return cropped
