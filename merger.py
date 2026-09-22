import logging
import numpy as np
import open3d as o3d
from plane import Plane
from utils import diagonal_of, resolve_threshold


class PlaneMerger:
    """
    工业级平面融合与去重模块
    职责: 基于法向量夹角和空间距离，将共面的“碎片平面”合并为单一的完整平面
    """

    #: 共面距离容差相对于场景对角线的比例。
    #: detector 用 0.008 × 对角线判定 RANSAC 内点；被同一方程吸收的两个碎片面，
    #: 其间距不应显著大于该内点容差，故取约 1.7× (≈0.0134 × 对角线) 作为上界。
    #: 该比例在当前参考数据集 (fused.ply，对角线 3.73862) 上精确复现原先硬编码的 0.05。
    DIST_THRESHOLD_RATIO = 0.0133739

    #: 接缝去重体素 = 平均点间距 ÷ 该系数。
    #: 平均点间距 ≈ 对角线 / sqrt(点数)，故此规则同时与"场景尺度"和"点密度"解耦。
    SEAM_VOXEL_SPACING_DIVISOR = 50.0

    def __init__(self, angle_threshold=3.0, dist_threshold=None,
                 dist_threshold_ratio=None, seam_voxel_divisor=None):
        """
        :param angle_threshold: 判定两个面共面的最大法向量夹角 (单位: 度)
        :param dist_threshold: 判定两个面共面的最大平行截距差 (绝对单位)。
                               默认 None → 按 "场景对角线 × DIST_THRESHOLD_RATIO" 自适应，
                               不再使用隐含"点云尺度≈1"假设的硬编码绝对值。
        :param dist_threshold_ratio: 覆盖默认比例 (仅 dist_threshold 为 None 时生效)
        :param seam_voxel_divisor: 覆盖接缝去重体素的分母 (默认 50)
        """
        self.logger = logging.getLogger("PointToCAD_System.Merger")
        self.angle_threshold = angle_threshold
        self.dist_threshold = dist_threshold
        self.dist_threshold_ratio = (
            self.DIST_THRESHOLD_RATIO if dist_threshold_ratio is None
            else float(dist_threshold_ratio)
        )
        self.seam_voxel_divisor = (
            self.SEAM_VOXEL_SPACING_DIVISOR if seam_voxel_divisor is None
            else float(seam_voxel_divisor)
        )

    def merge(self, raw_planes: list) -> list:
        """
        执行融合逻辑 (采用贪心聚类算法)

        :param raw_planes: detector.py 输出的未合并平面字典列表
        :return: 合并并重新封装后的 Plane 对象列表
        """
        if not raw_planes:
            return []

        self.logger.info(f"开始平面融合 (输入 {len(raw_planes)} 个原始碎片面)...")

        # 1. 将原始数据字典统一转化为 Plane 对象
        planes = [
            Plane(plane_id=p["id"], model=p["model"], cloud=p["cloud"])
            for p in raw_planes
        ]

        # 2. 按点数从大到小排序 (贪心策略: 总是以最大的面为基准去吸附小面)
        planes.sort(key=lambda p: len(p.cloud.points), reverse=True)

        # 3. 解析共面距离容差: 显式绝对值优先，否则随场景尺度自适应
        union_points = np.vstack([
            np.asarray(p.cloud.points) for p in planes if len(p.cloud.points) > 0
        ]) if any(len(p.cloud.points) > 0 for p in planes) else np.zeros((0, 3))
        scene_diagonal = diagonal_of(union_points)
        dist_thresh = resolve_threshold(
            self.dist_threshold, scene_diagonal,
            self.dist_threshold_ratio, floor=1e-12, name="merger.dist_threshold",
        )
        self.logger.debug(
            f"共面距离容差: {dist_thresh:.6g} "
            f"(场景对角线 {scene_diagonal:.6g} × {self.dist_threshold_ratio:g})"
        )

        merged_planes = []
        used_indices = set()

        for i in range(len(planes)):
            if i in used_indices:
                continue

            base_plane = planes[i]
            used_indices.add(i)

            # 准备收集所有被这个大面“吸附”的小面的点云
            clouds_to_merge = [base_plane.cloud]
            merged_count = 0

            # 用当前最大的面去和剩下的面一一比对
            for j in range(i + 1, len(planes)):
                if j in used_indices:
                    continue

                target_plane = planes[j]

                # 判断是否满足共面条件
                if self._is_coplanar(base_plane, target_plane, dist_thresh):
                    clouds_to_merge.append(target_plane.cloud)
                    used_indices.add(j)
                    merged_count += 1
                    self.logger.debug(f"  -> 碎片面 {target_plane.id} 被合并入大面 {base_plane.id}")

            # 如果发生了合并，我们需要把点云拼起来，并更新这个基础面
            if merged_count > 0:
                combined_cloud = self._combine_point_clouds(clouds_to_merge)
                base_plane.update_cloud(combined_cloud)
                self.logger.info(
                    f"面 {base_plane.id} 成功融合了 {merged_count} 个碎片面，合并后点数: {len(base_plane.cloud.points)}"
                )

            # 将（可能已融合更新过的）基准面加入最终列表
            merged_planes.append(base_plane)

        # 重新分配连续的 ID，方便后续报表输出
        for idx, p in enumerate(merged_planes):
            p.id = idx + 1

        self.logger.info(f"融合完成！最终保留 {len(merged_planes)} 个有效大平面。")
        return merged_planes

    def filter_table_plane(self, planes: list, area_ratio_thresh: float = 2.0,
                           parallel_angle_thresh: float = 10.0, low_margin_ratio: float = 0.1):
        """
        缺失面防护: 识别并剔除疑似"桌面"的超大平行平面。

        零件放在桌面上扫描时，底面缺失且桌面会被重建为一个大平面。
        若不剔除，桌面可能污染 3-2-1 基准选择、角点求解与 fused.ply (进而污染 ICP)。

        判定规则 (需同时满足):
          1. 与另一个平面近似平行 (夹角 <= parallel_angle_thresh)
          2. 面积 >= area_ratio_thresh × 该平行面的面积 (桌面远大于零件面)
          3. 沿公共法向位于所有平面质心投影的最低端 (零件坐在桌面上)

        :return: (kept_planes, table_plane_or_None, audit_or_None)
                 kept_planes 的 id 会重新顺序编号；audit 记录被剔平面信息与 ID 重映射，
                 供 report.json 持久化审计。
        """
        if len(planes) < 2:
            return planes, None, None

        n_planes = len(planes)
        for i in range(n_planes):
            for j in range(i + 1, n_planes):
                p, q = planes[i], planes[j]
                if p.angle_with(q) > parallel_angle_thresh:
                    continue

                if p.area >= q.area:
                    bi, si = i, j
                else:
                    bi, si = j, i
                bigger, smaller = planes[bi], planes[si]

                if bigger.area < area_ratio_thresh * max(smaller.area, 1e-12):
                    continue

                # 公共法向 (处理两平面法向可能反向的情况)
                n_avg = bigger.normal + smaller.normal
                if np.linalg.norm(n_avg) < 1e-6:
                    n_avg = bigger.normal.copy()
                n_avg = n_avg / np.linalg.norm(n_avg)

                projs = np.array([float(pl.centroid @ n_avg) for pl in planes])
                # 定向: 令桌面候选位于下方
                if projs[bi] > projs[si]:
                    projs = -projs
                spread = float(np.ptp(projs))

                if projs[bi] <= float(np.min(projs)) + low_margin_ratio * max(spread, 1e-12):
                    self.logger.info(
                        f"检测到疑似桌面平面 P{bigger.id} "
                        f"(面积 {bigger.area:.2f} ≥ {area_ratio_thresh}× 平行面 P{smaller.id} 的 {smaller.area:.2f})，已剔除"
                    )
                    # 审计记录: 被剔平面信息 + 旧 ID → 新 ID 重映射
                    audit = {
                        "removed_plane_id": int(bigger.id),
                        "removed_plane_equation": [round(float(v), 6) for v in bigger.model],
                        "removed_plane_area": round(float(bigger.area), 4),
                        "removed_plane_point_count": int(len(bigger.cloud.points)),
                    }
                    kept = [pl for k, pl in enumerate(planes) if k != bi]
                    id_remap = {}
                    for idx, pl in enumerate(kept):
                        id_remap[int(pl.id)] = idx + 1
                        pl.id = idx + 1
                    audit["id_remap"] = id_remap
                    return kept, bigger, audit

        return planes, None, None

    def _is_coplanar(self, plane_a: Plane, plane_b: Plane, dist_threshold: float) -> bool:
        """
        核心判定逻辑: 检查两个平面是否为同一个物理平面
        条件 1: 法向夹角极小
        条件 2: A面的质心到B面的距离极小

        :param dist_threshold: 共面距离容差，由 merge() 按场景尺度解析后传入
        """
        # 1. 检查法向夹角
        angle = plane_a.angle_with(plane_b)
        if angle > self.angle_threshold:
            return False

        # 2. 检查空间距离 (用 B 面的质心到 A 面的方程距离来衡量)
        dist = plane_a.distance_to_point(plane_b.centroid)
        if dist > dist_threshold:
            return False

        return True

    @classmethod
    def _seam_voxel_size(cls, combined: o3d.geometry.PointCloud) -> float:
        """
        计算拼接后的接缝去重体素尺寸 = 平均点间距 ÷ SEAM_VOXEL_SPACING_DIVISOR。

        旧实现硬编码 voxel_down_sample(0.001)，同样隐含"点云尺度≈1"假设。
        这里改为随尺度与密度自适应: 间距越密 → 体素越小，只剔除真正重叠的点，
        不会把相邻的采样点错误合并。
        """
        n = len(combined.points)
        scene_diagonal = diagonal_of(combined)
        if n < 2 or scene_diagonal <= 0:
            return 1e-9
        spacing = scene_diagonal / np.sqrt(n)
        return max(spacing / cls.SEAM_VOXEL_SPACING_DIVISOR, 1e-9)

    def _combine_point_clouds(self, clouds: list) -> o3d.geometry.PointCloud:
        """
        将多个点云对象拼装成一个点云对象，并做去重与轻量去噪。
        """
        combined = o3d.geometry.PointCloud()
        for c in clouds:
            combined += c

        # 1. 轻量级体素降采样，剔除接缝处的重叠点 (体素随尺度/密度自适应)
        combined = combined.voxel_down_sample(voxel_size=self._seam_voxel_size(combined))

        # 2. 统计滤波剔除明显离群点，防止相邻面的边缘飞点混入导致面积膨胀
        if len(combined.points) > 30:
            _, inlier_idx = combined.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0
            )
            combined = combined.select_by_index(inlier_idx)

        return combined