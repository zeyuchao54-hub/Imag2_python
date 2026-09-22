import argparse
import copy
import logging
import time
import sys
import os
import io
from pathlib import Path

import numpy as np
import open3d as o3d

# Windows 终端默认 GBK，强制 stdout 使用 UTF-8 以避免中文日志乱码/报错
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

# ==========================================
# 导入各个解耦的业务与功能模块
# ==========================================
try:
    from preprocess import PointCloudPreprocessor
    from detector import RansacDetector
    from merger import PlaneMerger
    from graph import PlaneGraphBuilder
    from geometry import GeometryAnalyzer
    from report import ReportGenerator
    from visualization import Visualizer
    from calibration import ScaleCalibrator, ScaleInfo
    from utils import validate_input_file
    from aligner import DatumAligner
    from cad_loader import CADLoader
    from registration import ICPRegistrar
    from deviation import DeviationAnalyzer
    from features import FeatureExtractor
    from tolerance import ToleranceAnalyzer
    from cad_cropper import crop_scan_to_cad_region
except ImportError as e:
    print(f"模块导入失败，请检查文件结构是否完整: {e}")
    sys.exit(1)


class IndustrialPipeline:
    """
    工业级点云处理流水线统筹类
    职责: 协调预处理、RANSAC提取、平面融合、拓扑求解、比例尺标定、基准坐标对齐、报告导出与 3D 可视化
    """

    def __init__(self):
        self.logger = self._setup_logger()
        self.args = self._parse_args()
        self._auto_scale_pending = False
        self._table_filter_audit = None
        self._prepare_environment()

    def _setup_logger(self):
        """配置终端与日志文件双向日志输出"""
        logger = logging.getLogger("PointToCAD_System")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            formatter = logging.Formatter(
                '[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

            # 1. 终端输出
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # 2. 文件日志输出
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def _parse_args(self):
        """命令行参数解析器"""
        parser = argparse.ArgumentParser(description="PointToCAD 工业级点云逆向工程与检测系统")
        parser.add_argument("--input", "-i", type=str, required=True, help="输入的 3D 点云文件路径 (.ply, .pcd)")
        parser.add_argument("--out_dir", "-o", type=str, default="outputs", help="结果导出目录路径")
        parser.add_argument("--export_cad", action="store_true",
                            help="是否导出 CAD 模型 (需要 CAD 内核后端; 未安装时跳过并告警，不会生成文件)")
        parser.add_argument("--batch", action="store_true", help="批处理/静默模式 (不显示 3D 可视化，跳过交互标定)")

        # ---------------------------------------------------------
        # ICP / CAD 配准相关参数
        # ---------------------------------------------------------
        parser.add_argument("--stl", type=str, default=None, help="输入的 STL CAD 模型路径 (启用 ICP 分支)")
        parser.add_argument("--icp", action="store_true", help="是否执行 CAD → Scan 的 ICP 配准与偏差分析")
        parser.add_argument("--cad_points", type=int, default=50000, help="STL 表面采样点数 (默认 50000)")
        parser.add_argument("--scale_factor", type=float, default=None,
                            help="物理尺度缩放因子覆盖 (batch 模式下必须提供，否则 fused.ply 与 STL 尺度不一致)")
        parser.add_argument("--primary_plane_id", type=int, default=1,
                            help="3-2-1 对齐主基准面 ID (对齐到 Z=0, 默认 1)")
        parser.add_argument("--secondary_plane_id", type=int, default=4,
                            help="3-2-1 对齐次基准面 ID (对齐到 X=0, 默认 4)")
        parser.add_argument("--icp_init_method", type=str, default="plane", choices=["pca", "plane", "fpfh"],
                            help="ICP 初始位姿方法: pca / plane / fpfh (默认 plane, 更适合 box-like 零件；fpfh 适合自由曲面)")
        parser.add_argument("--icp_coarse_voxel", type=float, default=2.0, help="粗 ICP voxel 尺寸 (默认 2.0 mm)")
        parser.add_argument("--icp_fine_voxel", type=float, default=0.5, help="精 ICP voxel 尺寸 (默认 0.5 mm)")
        parser.add_argument("--icp_coarse_max_corr", type=float, default=10.0, help="粗 ICP 最大对应点距离 (默认 10.0 mm)")
        parser.add_argument("--icp_fine_max_corr", type=float, default=1.0, help="精 ICP 最大对应点距离 (默认 1.0 mm)")
        parser.add_argument("--icp_coarse_iter", type=int, default=50, help="粗 ICP 最大迭代次数")
        parser.add_argument("--icp_fine_iter", type=int, default=50, help="精 ICP 最大迭代次数")
        parser.add_argument("--icp_plane_dist_thresh", type=float, default=2.0,
                            help="平面初始位姿的 RANSAC 距离阈值 (默认 2.0 mm)")
        parser.add_argument("--icp_plane_max_planes", type=int, default=10,
                            help="平面初始位姿最多检测平面数 (默认 10)")
        parser.add_argument("--icp_plane_min_points", type=int, default=500,
                            help="平面初始位姿单面最小点数 (默认 500)")
        parser.add_argument("--icp_plane_perp_thresh", type=float, default=15.0,
                            help="平面初始位姿判定垂直的角度容差 (默认 15°)")
        parser.add_argument("--icp_fpfh_voxel", type=float, default=2.0,
                            help="FPFH 特征提取降采样体素尺寸 (默认 2.0 mm)")
        parser.add_argument("--icp_fpfh_normal_factor", type=float, default=2.0,
                            help="FPFH 法向估计半径 = voxel * factor (默认 2.0)")
        parser.add_argument("--icp_fpfh_feature_factor", type=float, default=5.0,
                            help="FPFH 特征半径 = voxel * factor (默认 5.0)")
        parser.add_argument("--icp_fpfh_dist_factor", type=float, default=1.5,
                            help="FPFH RANSAC 距离阈值 = voxel * factor (默认 1.5)")
        parser.add_argument("--icp_fpfh_ransac_n", type=int, default=3,
                            help="FPFH RANSAC 每次采样点数 (默认 3)")
        parser.add_argument("--icp_fpfh_max_iter", type=int, default=100000,
                            help="FPFH RANSAC 最大迭代次数 (默认 100000)")
        parser.add_argument("--icp_fpfh_confidence", type=float, default=0.999,
                            help="FPFH RANSAC 置信度 (默认 0.999)")
        parser.add_argument("--tolerance", type=float, default=2.0,
                            help="公差判定阈值 (默认 ±2.0 mm, 用于自动化检测报告中的 PASS/FAIL 判定)")
        parser.add_argument("--detect_cylinders", action="store_true",
                            help="启用圆柱特征检测 (含孔/轴的零件可开启；纯平面零件建议关闭以避免假阳性)")
        # ---------------------------------------------------------
        # Datum-Weighted ICP / Auto-Scale / 缺失面防护参数
        # ---------------------------------------------------------
        parser.add_argument("--auto_scale", action="store_true", default=None,
                            help="保留兼容：提供 --stl + --icp 时 Auto-Scale 已默认启用；此标志无额外作用")
        parser.add_argument("--datum_weight", type=float, default=10.0,
                            help="Datum-Weighted ICP 基准区域对应点权重 (默认 10.0；设为 1.0 退化为标准 ICP)")
        parser.add_argument("--datum_dist_thresh", type=float, default=1.5,
                            help="判定 scan 点归属 datum 平面的距离阈值 mm (默认 1.5)")
        parser.add_argument("--icp_trim_fraction", type=float, default=1.0,
                            help="Trimmed ICP 保留的对应点比例 (默认 1.0；缺失面/遮挡扫描建议 0.9)")
        parser.add_argument("--keep_table_plane", action="store_true",
                            help="保留疑似桌面的超大平行平面 (默认自动剔除以防护缺失面扫描)")
        parser.add_argument("--table_area_ratio", type=float, default=2.0,
                            help="桌面平面判定的最小面积比 (默认 2.0，即疑似桌面面积 ≥ 2× 平行零件面)")
        parser.add_argument("--cad_crop", action="store_true",
                            help="启用 CAD 引导的扫描区域裁剪（当扫描场景包含大量背景时建议开启）")
        parser.add_argument("--cad_crop_margin", type=float, default=20.0,
                            help="CAD 引导裁剪的缓冲边距 (默认 20.0 mm)")
        parser.add_argument("--cad_crop_voxel", type=float, default=3.0,
                            help="CAD 引导裁剪前的聚类体素大小 (默认 3.0 mm)")
        return parser.parse_args()

    def _prepare_environment(self):
        """防御性文件校验与运行环境准备"""
        self.logger.info("正在执行系统环境初始化与输入文件校验...")
        validate_input_file(self.args.input, allowed_exts=[".ply", ".pcd"])
        Path(self.args.out_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _apply_scale_to_scene(planes, rest_pcd, factor):
        """
        统一对场景内可变点云对象施加物理比例尺。
        阶段 5 与阶段 5.5 (Auto-Scale) 共用此唯一入口，防止跨阶段重复/遗漏缩放。
        """
        for p in planes:
            p.cloud.scale(factor, center=(0, 0, 0))
        if rest_pcd is not None and not rest_pcd.is_empty():
            rest_pcd.scale(factor, center=(0, 0, 0))

    def run(self):
        """执行主流水线逻辑"""
        self.logger.info("==================================================")
        self.logger.info("启动 PointToCAD 工业级逆向工程流水线")
        self.logger.info("==================================================")

        t_start = time.time()

        try:
            # ---------------------------------------------------------
            # 阶段 1: 点云预处理 (降采样 / 去噪 / 法向)
            # ---------------------------------------------------------
            self.logger.info("[1/6] 正在进行点云预处理 (降采样 / 统计去噪 / 法向量估计)...")
            t1 = time.time()
            preprocessor = PointCloudPreprocessor()
            pcd_clean = preprocessor.process(self.args.input)
            self.logger.info(f"      预处理完成 -> 耗时: {time.time() - t1:.3f}s")

            # ---------------------------------------------------------
            # 阶段 1.5: CAD 引导的扫描区域裁剪（当提供 STL 时自动启用）
            # ---------------------------------------------------------
            if self.args.stl and self.args.cad_crop:
                self.logger.info("[1.5/6] 正在执行 CAD 引导的扫描区域裁剪...")
                t15 = time.time()
                pcd_clean = crop_scan_to_cad_region(
                    scan_pcd=pcd_clean,
                    stl_path=self.args.stl,
                    margin_mm=self.args.cad_crop_margin,
                    voxel_mm=self.args.cad_crop_voxel,
                )
                self.logger.info(f"      CAD 裁剪完成 -> 耗时: {time.time() - t15:.3f}s")

            # ---------------------------------------------------------
            # 阶段 2: RANSAC 多平面几何图元提取
            # ---------------------------------------------------------
            self.logger.info("[2/6] 正在执行 RANSAC 平面图元提取...")
            t2 = time.time()
            detector = RansacDetector()
            raw_planes, rest_pcd = detector.detect(pcd_clean)
            self.logger.info(f"      平面提取完成 -> 耗时: {time.time() - t2:.3f}s")

            # ---------------------------------------------------------
            # 阶段 3: 碎片面融合与去重
            # ---------------------------------------------------------
            self.logger.info("[3/6] 正在执行共面碎片融合...")
            t3 = time.time()
            merger = PlaneMerger()
            merged_planes = merger.merge(raw_planes)
            self.logger.info(f"      融合完成 -> 耗时: {time.time() - t3:.3f}s")

            # ---------------------------------------------------------
            # 阶段 3.5: 桌面平面剔除 (缺失面防护)
            # ---------------------------------------------------------
            if not self.args.keep_table_plane:
                merged_planes, table_plane, self._table_filter_audit = merger.filter_table_plane(
                    merged_planes, area_ratio_thresh=self.args.table_area_ratio
                )
                if table_plane is not None:
                    self.logger.info(
                        f"[3.5/6] 疑似桌面平面已剔除 (面积 {table_plane.area:.2f})，其点云并入残余点云"
                    )
                    # 重编号导致基准面 ID 重绑定时显式告警 (审计详情写入 report.json)
                    removed_id = self._table_filter_audit["removed_plane_id"]
                    affected = sorted(
                        d for d in (self.args.primary_plane_id, self.args.secondary_plane_id)
                        if d >= removed_id
                    )
                    if affected:
                        self.logger.warning(
                            f"桌面平面 P{removed_id} 剔除导致平面重编号，基准面 ID {affected} 已重绑定；"
                            f"若结果异常请核对 --primary_plane_id/--secondary_plane_id 或使用 --keep_table_plane"
                        )
                    if rest_pcd is None or rest_pcd.is_empty():
                        rest_pcd = table_plane.cloud
                    else:
                        rest_pcd = rest_pcd + table_plane.cloud

            # ---------------------------------------------------------
            # 阶段 4: 拓扑建图与 CAD 角点/顶点求解
            # ---------------------------------------------------------
            self.logger.info("[4/6] 正在构建拓扑关系图与求解 CAD 角点...")
            t4 = time.time()
            graph_builder = PlaneGraphBuilder()
            graph = graph_builder.build(merged_planes)

            geometry_analyzer = GeometryAnalyzer()
            vertices = geometry_analyzer.compute_intersection_vertices(graph)
            self.logger.info(f"      顶点求解完成 -> 耗时: {time.time() - t4:.3f}s | 捕获 {len(vertices)} 个几何角点")

            # ---------------------------------------------------------
            # 阶段 4.5: 业务级物理比例尺标定 (Calibration Engine)
            # ---------------------------------------------------------
            if self.args.scale_factor is not None:
                self.logger.info(f"[4.5/6] 使用命令行提供的比例尺因子: {self.args.scale_factor:.6f}")
                scale_info = ScaleInfo(
                    factor=self.args.scale_factor,
                    reference="CLI --scale_factor override",
                    feature_type="CLI Override",
                    real_distance_mm=0.0,
                    virtual_distance=0.0,
                )
            elif self.args.stl and self.args.icp:
                # 当提供 STL 并启用 ICP 时，默认使用 Auto-Scale，无需手动指定比例尺
                self.logger.info("[4.5/6] Auto-Scale 模式: 检测到 --stl + --icp，将自动估计全局比例尺，无需人工标定")
                scale_info = ScaleInfo(
                    factor=1.0,
                    reference="Pending auto-scale (Sim3 ICP)",
                    feature_type="Pending",
                )
                self._auto_scale_pending = True
            elif not self.args.batch:
                self.logger.info("[4.5/6] 启动交互式物理比例尺标定 Engine...")
                calibrator = ScaleCalibrator()
                scale_info = calibrator.interactive_calibrate(vertices, merged_planes)
            else:
                self.logger.info("[4.5/6] 批处理模式下未提供 STL/ICP，自动使用默认比例尺 (Scale Factor = 1.0)")
                scale_info = ScaleInfo()

            # ---------------------------------------------------------
            # 阶段 4.8: 3-2-1 工业基准坐标系对齐 (Datum Alignment)   【位置 2】
            # ---------------------------------------------------------
            self.logger.info("[4.8/6] 正在执行 3-2-1 工业基准坐标系对齐...")
            aligner = DatumAligner()
            merged_planes, vertices, T_matrix = aligner.align_to_datum(
                planes=merged_planes,
                vertices=vertices,
                primary_plane_id=self.args.primary_plane_id,
                secondary_plane_id=self.args.secondary_plane_id
            )

            # 同步变换未分配的剩余点云 (确保可视化时不脱节)
            if rest_pcd is not None and not rest_pcd.is_empty():
                rest_pcd.transform(T_matrix)

            # ---------------------------------------------------------
            # 阶段 5: 物理尺度施加与文件导出
            # ---------------------------------------------------------
            self.logger.info("[5/6] 正在按物理标定尺寸生成导出的检验文件...")
            t5 = time.time()
            reporter = ReportGenerator(out_dir=self.args.out_dir)

            # 1. 对融合后平面的点云与残余点云应用物理空间缩放
            #    Auto-Scale 挂起时 factor 必为 1.0 (占位)，显式跳过以避免与阶段 5.5 重复缩放
            if not self._auto_scale_pending:
                self._apply_scale_to_scene(merged_planes, rest_pcd, scale_info.factor)

            # 2. 导出包含绝对真实物理尺寸的融合点云 fused.ply (专门用于 ICP 对齐比对)
            fused_ply_path = reporter.export_fused_ply(merged_planes, filename="fused.ply")
            self.logger.info(f"      [ICP点云] 融合点云已保存至: {fused_ply_path}")

            # 3. 导出 JSON 几何检验报表 (嵌入结构化的 ScaleInfo 标定元数据 + 桌面剔除审计)
            report_path = reporter.export_json(
                merged_planes, vertices, scale_info=scale_info,
                extra_metadata={"table_filter": self._table_filter_audit} if self._table_filter_audit else None,
            )
            self.logger.info(f"      [几何报表] JSON 报告已保存至: {report_path}")

            # 4. 根据配置导出 STEP 格式 CAD 模型
            #    Auto-Scale 挂起时推迟到阶段 5.5 尺度估计完成后再导出，避免虚拟单位 STEP
            if self.args.export_cad and not self._auto_scale_pending:
                scaled_vertices = vertices * scale_info.factor if len(vertices) > 0 else vertices
                cad_path = reporter.export_step(scaled_vertices, merged_planes)
                if cad_path:
                    self.logger.info(f"      [CAD模型] STEP 文件已保存至: {cad_path}")
                else:
                    self.logger.warning(
                        "      [CAD模型] STEP 未生成 (当前环境无 CAD B-Rep 后端)，其余报告不受影响"
                    )

            self.logger.info(f"      文件导出完成 -> 耗时: {time.time() - t5:.3f}s")

            # ---------------------------------------------------------
            # 阶段 5.5: CAD 加载、ICP 配准与偏差分析
            # ---------------------------------------------------------
            if self.args.stl and self.args.icp:
                self.logger.info("[5.5/6] 启动 CAD → Scan ICP 配准与偏差分析...")
                t_icp = time.time()

                # 防御性校验
                validate_input_file(self.args.stl, allowed_exts=[".stl"])

                # 1. 加载 CAD 表面点云与原始网格顶点
                cad_loader = CADLoader(num_points=self.args.cad_points)
                cad_pcd = cad_loader.load(self.args.stl)
                cad_vertices_raw = cad_loader.get_vertices()

                # 2. 读取已导出的 fused.ply 作为 scan 输入
                scan_pcd = o3d.io.read_point_cloud(fused_ply_path)
                if scan_pcd.is_empty():
                    raise ValueError(f"无法读取或 fused.ply 为空: {fused_ply_path}")

                # 3. 构建 ICP 配准器
                registrar = ICPRegistrar(
                    coarse_voxel_size=self.args.icp_coarse_voxel,
                    fine_voxel_size=self.args.icp_fine_voxel,
                    coarse_max_correspondence_distance=self.args.icp_coarse_max_corr,
                    fine_max_correspondence_distance=self.args.icp_fine_max_corr,
                    coarse_max_iterations=self.args.icp_coarse_iter,
                    fine_max_iterations=self.args.icp_fine_iter,
                    init_method=self.args.icp_init_method,
                    plane_distance_threshold=self.args.icp_plane_dist_thresh,
                    plane_max_planes=self.args.icp_plane_max_planes,
                    plane_min_points=self.args.icp_plane_min_points,
                    plane_perp_angle_thresh=self.args.icp_plane_perp_thresh,
                    fpfh_voxel_size=self.args.icp_fpfh_voxel,
                    fpfh_normal_radius_factor=self.args.icp_fpfh_normal_factor,
                    fpfh_feature_radius_factor=self.args.icp_fpfh_feature_factor,
                    fpfh_distance_threshold_factor=self.args.icp_fpfh_dist_factor,
                    fpfh_ransac_n=self.args.icp_fpfh_ransac_n,
                    fpfh_max_iteration=self.args.icp_fpfh_max_iter,
                    fpfh_confidence=self.args.icp_fpfh_confidence,
                    datum_weight=self.args.datum_weight,
                    datum_dist_thresh=self.args.datum_dist_thresh,
                    trim_fraction=self.args.icp_trim_fraction,
                )

                # 3.5 Auto-Scale: Sim3 ICP 自动估计全局比例尺 (全自动标定)
                if self._auto_scale_pending:
                    scale_est = registrar.estimate_scale_factor(scan_pcd, cad_pcd)
                    scale_info = ScaleInfo(
                        factor=scale_est,
                        reference="Auto-estimated via Sim3 ICP (Weighted Umeyama)",
                        feature_type="Auto Sim3",
                        real_distance_mm=float(np.linalg.norm(cad_pcd.get_axis_aligned_bounding_box().get_extent())),
                        virtual_distance=float(np.linalg.norm(scan_pcd.get_axis_aligned_bounding_box().get_extent())),
                    )
                    # 将估计比例尺施加到 scan 点云、平面点云与残余点云 (统一入口)
                    scan_pcd.scale(scale_est, center=(0, 0, 0))
                    self._apply_scale_to_scene(merged_planes, rest_pcd, scale_est)
                    # 以物理单位重新导出 fused.ply 与 report.json (导出方法内部记录路径)
                    reporter.export_fused_ply(merged_planes, filename="fused.ply")
                    reporter.export_json(
                        merged_planes, vertices, scale_info=scale_info,
                        extra_metadata={"table_filter": self._table_filter_audit} if self._table_filter_audit else None,
                    )
                    self.logger.info(
                        f"      [Auto-Scale] scale_factor={scale_est:.6f} mm/虚拟单位，"
                        f"已重新导出物理单位 fused.ply 与 report.json"
                    )
                    # 阶段 5 被推迟的 STEP 导出: 按估计尺度重新计算并导出
                    if self.args.export_cad:
                        scaled_vertices = vertices * scale_info.factor if len(vertices) > 0 else vertices
                        cad_path = reporter.export_step(scaled_vertices, merged_planes)
                        if cad_path:
                            self.logger.info(f"      [CAD模型] STEP 文件已按估计比例尺导出: {cad_path}")
                        else:
                            self.logger.warning(
                                "      [CAD模型] STEP 未生成 (当前环境无 CAD B-Rep 后端)，其余报告不受影响"
                            )

                # 4. 构建 datum 平面列表 (主基准面 → Datum A, 次基准面 → Datum B, 物理单位)
                datum_planes = []
                datum_ids = {self.args.primary_plane_id, self.args.secondary_plane_id}
                for p in merged_planes:
                    if p.id in datum_ids:
                        datum_planes.append(
                            (np.array(p.normal, dtype=float), float(p.model[3]) * scale_info.factor)
                        )
                if self.args.datum_weight > 1.0:
                    if datum_planes:
                        self.logger.info(
                            f"      [Datum-Weighted ICP] 基准面: {sorted(datum_ids)} "
                            f"(匹配 {len(datum_planes)} 个, w={self.args.datum_weight})"
                        )
                    else:
                        self.logger.warning(
                            f"未在 merged_planes 中找到基准面 ID {sorted(datum_ids)}，Fine ICP 退化为标准 Point-to-Plane"
                        )

                # 5. 执行 coarse-to-fine ICP (Fine 阶段 Datum-Weighted)
                reg_result = registrar.register(
                    source=cad_pcd,
                    target=scan_pcd,
                    datum_planes=datum_planes if datum_planes else None,
                )

                if not reg_result.success:
                    self.logger.error(f"ICP 配准失败: {reg_result.message}")
                    raise RuntimeError(reg_result.message)

                if self._auto_scale_pending:
                    reg_result.estimated_scale = scale_est

                reg_path = reporter.export_registration_json(reg_result, filename="registration.json")
                self.logger.info(f"      [配准报告] 已保存至: {reg_path}")

                # 4. 将对齐后的 CAD 点云导出
                #    注意: open3d.transform() 是 in-place 操作，先 deep copy 再变换，避免污染原始 CAD 点云
                aligned_cad_pcd = copy.deepcopy(cad_pcd)
                aligned_cad_pcd.transform(reg_result.transformation)
                aligned_cad_path = reporter.export_point_cloud(
                    aligned_cad_pcd, filename="aligned_cad.ply"
                )
                self.logger.info(f"      [对齐 CAD] 已保存至: {aligned_cad_path}")

                # 同步变换原始 CAD 顶点，用于位置度计算
                if len(cad_vertices_raw) > 0:
                    T = reg_result.transformation
                    cad_vertices_aligned = (
                        cad_vertices_raw @ T[:3, :3].T + T[:3, 3]
                    )
                else:
                    cad_vertices_aligned = cad_vertices_raw

                # 5. 偏差分析
                analyzer = DeviationAnalyzer()
                dev_result, cad_colored, scan_colored = analyzer.analyze(
                    cad_pcd=aligned_cad_pcd, scan_pcd=scan_pcd
                )

                dev_path = reporter.export_deviation_json(dev_result, filename="deviation.json")
                self.logger.info(f"      [偏差报告] 已保存至: {dev_path}")

                deviation_map_path = reporter.export_point_cloud(
                    scan_colored, filename="deviation_map.ply"
                )
                self.logger.info(f"      [偏差色谱] 已保存至: {deviation_map_path}")

                # 6. 几何公差分析 (GD&T)
                self.logger.info("      [公差分析] 开始计算几何公差 (GD&T)...")
                feature_extractor = FeatureExtractor()

                # 扫描特征
                scan_planes = feature_extractor.extract_planes_from_scan(merged_planes, source="scan")
                # 点云已缩放，但 PlaneFeature 中的 centroid / d 仍为虚拟单位，需同步缩放
                for sp in scan_planes:
                    sp.centroid = sp.centroid * scale_info.factor
                    sp.d = sp.d * scale_info.factor

                scan_cylinders = []
                if self.args.detect_cylinders:
                    scan_cylinders = feature_extractor.detect_cylinders(
                        rest_pcd if rest_pcd is not None else o3d.geometry.PointCloud(),
                        source="scan"
                    )
                scan_lines = feature_extractor.extract_lines_from_planes(
                    scan_planes, source_pcd=scan_pcd, edge_width=1.0
                )
                scaled_vertices_for_tol = vertices * scale_info.factor if len(vertices) > 0 else vertices
                scan_points = feature_extractor.extract_points_from_array(scaled_vertices_for_tol, source="scan")

                # CAD 名义特征 (从对齐后的 CAD 点云提取)
                cad_planes = feature_extractor.extract_planes_from_pcd(
                    aligned_cad_pcd, max_planes=10, source="cad"
                )
                cad_cylinders = []
                if self.args.detect_cylinders:
                    cad_cylinders = feature_extractor.detect_cylinders(
                        aligned_cad_pcd, source="cad"
                    )
                # 使用原始网格顶点作为 CAD 名义顶点（比平面交点更稳定）
                cad_points = feature_extractor.extract_points_from_array(
                    cad_vertices_aligned, source="cad"
                )

                # 偏差 signed 数组用于轮廓度
                deviation_signed = dev_result.scan_to_cad_signed

                tolerance_analyzer = ToleranceAnalyzer(
                    tolerance_threshold_mm=self.args.tolerance
                )
                tolerance_results = tolerance_analyzer.analyze(
                    scan_planes=scan_planes,
                    cad_planes=cad_planes,
                    scan_cylinders=scan_cylinders,
                    cad_cylinders=cad_cylinders,
                    scan_lines=scan_lines,
                    scan_points=scan_points,
                    cad_points=cad_points,
                    deviation_signed=deviation_signed,
                )

                tolerance_path = reporter.export_tolerance_json(
                    tolerance_results, filename="tolerance.json"
                )
                self.logger.info(f"      [公差报告] 已保存至: {tolerance_path}")

                self.logger.info(f"      ICP、偏差与公差分析完成 -> 耗时: {time.time() - t_icp:.3f}s")

                # 保存到实例变量，供可视化阶段使用
                self._icp_visualization_data = {
                    "cad_colored": cad_colored,
                    "scan_colored": scan_colored,
                }
            elif self.args.icp and not self.args.stl:
                self.logger.warning("指定了 --icp 但未提供 --stl，跳过 ICP 配准。")

            # ---------------------------------------------------------
            # 阶段 6: 3D 可视化渲染 (仅在非 batch 模式下触发)
            # ---------------------------------------------------------
            if not self.args.batch:
                visualizer = Visualizer()

                # 如果有 ICP 偏差结果，优先显示偏差色谱
                if hasattr(self, "_icp_visualization_data") and self._icp_visualization_data:
                    self.logger.info("[6/6] 正在启动偏差色谱 3D 渲染引擎...")
                    visualizer.draw_deviation(
                        scan_colored=self._icp_visualization_data["scan_colored"],
                        cad_colored=self._icp_visualization_data["cad_colored"],
                    )
                else:
                    self.logger.info("[6/6] 正在启动 3D 可视化渲染引擎...")
                    scaled_vertices = vertices * scale_info.factor if len(vertices) > 0 else vertices
                    visualizer.draw_scene(merged_planes, scaled_vertices, rest_pcd)
            else:
                self.logger.info("[6/6] 当前为批处理模式，已跳过 3D 渲染界面显示。")

            # ---------------------------------------------------------
            # 阶段 7: 自动化综合检测报告生成
            # ---------------------------------------------------------
            t_total = time.time() - t_start
            self.logger.info("[7/6] 正在生成自动化综合检测报告...")
            reporter.export_inspection_summary(
                input_file=self.args.input,
                stl_file=self.args.stl,
                icp_init_method=self.args.icp_init_method if self.args.icp else "N/A",
                total_time=t_total,
                tolerance_mm=self.args.tolerance,
            )
            self.logger.info("      自动化报告生成完毕。")

            self.logger.info("==================================================")
            self.logger.info(f"流水线全部顺利执行完毕！总耗时: {t_total:.2f} 秒")
            self.logger.info("==================================================")

        except Exception as e:
            self.logger.error(f"流水线发生错误: {str(e)}", exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    pipeline = IndustrialPipeline()
    pipeline.run()