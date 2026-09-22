import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import open3d as o3d

# 导入项目内部依赖
from plane import Plane
from calibration import ScaleInfo
from registration import RegistrationResult
from deviation import DeviationResult


class ReportGenerator:
    """
    工业级检验报告与模型导出模块
    负责: 生成含有物理绝对尺寸的 JSON 报表、融合点云 (PLY) 以及 CAD (STEP) 模型
    """

    def __init__(self, out_dir: str = "outputs"):
        self.out_dir = Path(out_dir)
        # 确保输出目录存在
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("PointToCAD_System.Reporter")

    def export_fused_ply(self, planes: List[Plane], filename: str = "fused.ply") -> str:
        """
        将所有识别并融合后的平面点云合并，导出为单独的 fused.ply 文件 (用于 ICP 比对)
        注意：传入此处的 planes 的点云应当在外部已经被施加了 Scale 缩放。
        """
        file_path = self.out_dir / filename
        self.logger.info(f"开始导出融合点云: {file_path} ...")

        if not planes:
            self.logger.warning("没有可导出的平面点云数据。")
            return ""

        # 1. 拼接所有平面的点云
        combined_pcd = o3d.geometry.PointCloud()
        for p in planes:
            combined_pcd += p.cloud

        # 2. 保存为 PLY 格式
        o3d.io.write_point_cloud(str(file_path), combined_pcd)
        self.logger.info(f"fused.ply 成功保存！总点数: {len(combined_pcd.points)}")

        return str(file_path.absolute())

    def export_json(
            self,
            planes: List[Plane],
            vertices: np.ndarray,
            filename: str = "report.json",
            scale_info: Optional[ScaleInfo] = None,
            extra_metadata: Optional[Dict] = None,
    ) -> str:
        """
        将检测结果与比例尺标定元数据导出为 JSON 报告
        """
        file_path = self.out_dir / filename

        # 如果未传入 scale_info，则默认使用缩放因子为 1.0 的未标定状态
        scale_info = scale_info or ScaleInfo()
        scale_factor = scale_info.factor

        self.logger.info(f"开始生成 JSON 检验报告 (Scale Factor: {scale_factor:.6f}): {file_path} ...")

        # 1. 结构化平面数据 (乘以 scale_factor 修正截距 d, 质心 centroid 与面积 area)
        planes_data = []
        for p in planes:
            a, b, c, d = p.model.tolist()
            cx, cy, cz = p.centroid.tolist()
            planes_data.append({
                "plane_id": p.id,
                "equation": {
                    "a": round(a, 6),
                    "b": round(b, 6),
                    "c": round(c, 6),
                    "d": round(d * scale_factor, 6)  # 偏移量 d 随空间线性缩放
                },
                "point_count": len(p.cloud.points),
                "estimated_area_mm2": round(p.area * (scale_factor ** 2), 6),  # 面积按平方缩放
                "centroid": [
                    round(cx * scale_factor, 6),
                    round(cy * scale_factor, 6),
                    round(cz * scale_factor, 6)
                ]
            })

        # 2. 结构化顶点数据 (坐标按比例放大)
        vertices_data = []
        if len(vertices) > 0:
            for idx, v in enumerate(vertices):
                vx, vy, vz = (v * scale_factor).tolist()
                vertices_data.append({
                    "vertex_id": idx + 1,
                    "x": round(vx, 6),
                    "y": round(vy, 6),
                    "z": round(vz, 6)
                })

        # 3. 组合最终 JSON (嵌入结构化的 scale 对象，方便学术论文引用与工业检测溯源)
        metadata = {
            "system": "PointToCAD Industrial Inspection Engine",
            "scale": scale_info.to_dict(),
            "summary": {
                "total_planes_detected": len(planes),
                "total_vertices_found": len(vertices)
            }
        }
        # 附加审计元数据 (如桌面平面剔除记录)
        if extra_metadata:
            metadata.update(extra_metadata)

        report_content = {
            "metadata": metadata,
            "planes": planes_data,
            "vertices": vertices_data
        }

        # 4. 写入本地文件
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(report_content, f, indent=4, ensure_ascii=False)

        self.logger.info("JSON 报告导出成功！")
        return str(file_path.absolute())

    #: 可写出 STEP (B-Rep) 的 CAD 内核后端，按优先级排列。
    #: STEP 是边界表示格式，必须借助 OpenCASCADE 系内核，trimesh/numpy 均无法直接写出。
    _CAD_BACKENDS = ("cadquery", "OCP", "OCC")

    @classmethod
    def _detect_cad_backend(cls) -> Optional[str]:
        """探测当前环境可用的 CAD 内核后端，无则返回 None。"""
        for name in cls._CAD_BACKENDS:
            try:
                __import__(name)
                return name
            except ImportError:
                continue
        return None

    def export_step(self, vertices: np.ndarray, planes: List[Plane],
                    filename: str = "model.step") -> Optional[str]:
        """
        导出为 CAD 的 STEP 格式。

        重要: 本方法**不会**在没有 CAD 内核的环境下谎报成功。
        STEP 需要 OpenCASCADE 系后端 (cadquery 或 pythonocc-core)，
        若均未安装，则返回 None 并明确告警，磁盘上不会留下任何文件。

        :param vertices: CAD 角点坐标 (应为已施加 scale_factor 的绝对物理坐标)
        :param planes: 融合后的平面列表
        :return: 成功时返回文件绝对路径；无 CAD 后端时返回 None
        """
        file_path = self.out_dir / filename
        backend = self._detect_cad_backend()

        if backend is None:
            self.logger.warning(
                "STEP 导出已跳过: 未检测到 CAD 内核后端 "
                f"(已尝试 {', '.join(self._CAD_BACKENDS)})。"
                f"文件 '{file_path.name}' 未被创建。"
                "如需真正的 B-Rep STEP 输出，请安装: pip install cadquery"
            )
            return None

        self.logger.info(f"正在生成 STEP 文件 (后端: {backend}): {file_path} ...")

        # ---------------------------------------------------------
        # B-Rep 重建接入点
        # ---------------------------------------------------------
        # 此处需要把 vertices / planes 转成面片 (Face) → 壳 (Shell) → 体 (Solid)。
        # 以 cadquery 为例:
        #     import cadquery as cq
        #     wp = cq.Workplane("XY")
        #     for p in planes:
        #         hull2d = <p.cloud 投影到 p.normal 局部坐标系的 2D 凸包>
        #         wp = wp.add(cq.Face.makeFromWires(
        #             cq.Wire.makePolygon([cq.Vector(*v) for v in hull3d], close=True)
        #         ))
        #     cq.exporters.export(wp, str(file_path), exportType="STEP")
        # 在完成实现并使 tests/test_report.py 覆盖该分支之前，
        # 这里主动返回 None 而不是写一个空文件后谎报成功。
        self.logger.warning(
            f"检测到 CAD 后端 '{backend}'，但 B-Rep 重建逻辑尚未实现，"
            f"'{file_path.name}' 未被创建。请在此处接入 OpenCASCADE 建模调用。"
        )
        return None

    def export_registration_json(
            self,
            reg_result: RegistrationResult,
            filename: str = "registration.json"
    ) -> str:
        """
        导出 ICP 配准结果（fitness、rmse、变换矩阵等）。
        """
        file_path = self.out_dir / filename
        self.logger.info(f"开始导出配准报告: {file_path} ...")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(reg_result.to_dict(), f, indent=4, ensure_ascii=False)

        self.logger.info("配准报告导出成功！")
        return str(file_path.absolute())

    def export_deviation_json(
            self,
            dev_result: DeviationResult,
            filename: str = "deviation.json"
    ) -> str:
        """
        导出偏差分析统计量。
        """
        file_path = self.out_dir / filename
        self.logger.info(f"开始导出偏差统计报告: {file_path} ...")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(dev_result.to_dict(), f, indent=4, ensure_ascii=False)

        self.logger.info("偏差统计报告导出成功！")
        return str(file_path.absolute())

    def export_point_cloud(
            self,
            pcd: o3d.geometry.PointCloud,
            filename: str = "cloud.ply"
    ) -> str:
        """
        通用点云导出方法，用于保存 aligned CAD、deviation map 等。
        """
        file_path = self.out_dir / filename
        self.logger.info(f"开始导出点云: {file_path} ...")

        if pcd.is_empty():
            self.logger.warning("点云为空，跳过导出。")
            return ""

        o3d.io.write_point_cloud(str(file_path), pcd)
        self.logger.info(f"点云导出成功！总点数: {len(pcd.points)}")
        return str(file_path.absolute())

    def export_tolerance_json(
            self,
            tolerance_results: Dict[str, List],
            filename: str = "tolerance.json"
    ) -> str:
        """
        导出几何公差 (GD&T) 计算结果。
        """
        file_path = self.out_dir / filename
        self.logger.info(f"开始导出公差报告: {file_path} ...")

        content = {}
        for category, results in tolerance_results.items():
            content[category] = [r.to_dict() if hasattr(r, "to_dict") else r for r in results]

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=4, ensure_ascii=False)

        self.logger.info("公差报告导出成功！")
        return str(file_path.absolute())

    def export_inspection_summary(
            self,
            input_file: str,
            stl_file: Optional[str],
            icp_init_method: str,
            total_time: float,
            tolerance_mm: float = 2.0,
            filename: str = "inspection_report.txt"
    ) -> str:
        """
        读取所有 JSON 输出，生成一份 human-readable 的自动化综合检测报告。

        :param input_file: 输入点云文件路径
        :param stl_file: STL CAD 文件路径 (可能为 None)
        :param icp_init_method: ICP 初始位姿方法名
        :param total_time: 流水线总耗时 (秒)
        :param tolerance_mm: 公差判定阈值 (默认 ±2.0 mm)
        :param filename: 报告输出文件名
        :return: 生成的报告文件绝对路径
        """
        from datetime import datetime

        file_path = self.out_dir / filename
        self.logger.info(f"开始生成自动化综合检测报告: {file_path} ...")

        lines = []
        sep = "=" * 60

        # ---------------------------------------------------------
        # Header
        # ---------------------------------------------------------
        lines.append(sep)
        lines.append("PointToCAD 工业检测自动化报告")
        lines.append(sep)
        lines.append(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"输入点云: {input_file}")
        if stl_file:
            lines.append(f"CAD 模型: {stl_file}")
        lines.append(f"总耗时: {total_time:.2f} 秒")
        lines.append("")

        # ---------------------------------------------------------
        # 1. 几何检测摘要 (report.json)
        # ---------------------------------------------------------
        report_json_path = self.out_dir / "report.json"
        if report_json_path.exists():
            with open(report_json_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)
            meta = report_data.get("metadata", {})
            scale = meta.get("scale", {})
            summary = meta.get("summary", {})

            lines.append("--- 1. 几何检测摘要 ---")
            lines.append(f"识别平面数: {summary.get('total_planes_detected', 'N/A')}")
            lines.append(f"识别角点数: {summary.get('total_vertices_found', 'N/A')}")
            lines.append(f"比例尺因子: {scale.get('factor', 'N/A')} mm/虚拟单位")
            lines.append(f"比例尺来源: {scale.get('reference', 'N/A')}")

            planes = report_data.get("planes", [])
            if planes:
                lines.append("")
                lines.append("平面详情:")
                for p in planes:
                    eq = p.get("equation", {})
                    lines.append(
                        f"  Plane {p.get('plane_id', '?')}: "
                        f"a={eq.get('a', '?'):.6f}, b={eq.get('b', '?'):.6f}, c={eq.get('c', '?'):.6f}, d={eq.get('d', '?'):.4f} | "
                        f"点数={p.get('point_count', '?')}, 面积={p.get('estimated_area_mm2', '?'):.2f} mm²"
                    )
            lines.append("")
        else:
            lines.append("--- 1. 几何检测摘要 ---")
            lines.append("  (report.json 不存在，跳过)")
            lines.append("")

        # ---------------------------------------------------------
        # 2. ICP 配准结果 (registration.json)
        # ---------------------------------------------------------
        reg_json_path = self.out_dir / "registration.json"
        if reg_json_path.exists():
            with open(reg_json_path, "r", encoding="utf-8") as f:
                reg_data = json.load(f)

            lines.append("--- 2. ICP 配准结果 ---")
            lines.append(f"初始位姿方法: {icp_init_method}")
            lines.append(f"配准状态: {'成功' if reg_data.get('success') else '失败'}")
            lines.append(f"Fitness: {reg_data.get('fitness', 'N/A'):.6f}")
            lines.append(f"Inlier RMSE: {reg_data.get('inlier_rmse', 'N/A'):.6f} mm")
            lines.append(f"Source (CAD) 点数: {reg_data.get('num_source_points', 'N/A')}")
            lines.append(f"Target (Scan) 点数: {reg_data.get('num_target_points', 'N/A')}")

            trans = reg_data.get("translation", [])
            if len(trans) == 3:
                lines.append(f"平移向量 (X, Y, Z): {trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f} mm")
            est_scale = reg_data.get("estimated_scale", 1.0)
            if est_scale is not None and abs(float(est_scale) - 1.0) > 1e-9:
                lines.append(f"自动比例尺 (Sim3 ICP): {float(est_scale):.6f} mm/虚拟单位")
            lines.append("")
        else:
            lines.append("--- 2. ICP 配准结果 ---")
            lines.append("  (registration.json 不存在，跳过)")
            lines.append("")

        # ---------------------------------------------------------
        # 3. 偏差分析 (deviation.json)
        # ---------------------------------------------------------
        dev_json_path = self.out_dir / "deviation.json"
        if dev_json_path.exists():
            with open(dev_json_path, "r", encoding="utf-8") as f:
                dev_data = json.load(f)

            scan_to_cad = dev_data.get("scan_to_cad", {})
            cad_to_scan = dev_data.get("cad_to_scan", {})

            lines.append("--- 3. 偏差分析 (Scan → CAD) ---")
            if scan_to_cad:
                signed = scan_to_cad.get("signed", {})
                unsigned = scan_to_cad.get("unsigned", {})
                lines.append(f"点数: {scan_to_cad.get('count', 'N/A')}")
                lines.append(f"Signed 均值: {signed.get('mean', 'N/A'):.4f} mm")
                lines.append(f"Signed 中位数: {signed.get('median', 'N/A'):.4f} mm")
                lines.append(f"RMS: {signed.get('rms', 'N/A'):.4f} mm")
                lines.append(f"最大值: {signed.get('max', 'N/A'):.4f} mm")
                lines.append(f"最小值: {signed.get('min', 'N/A'):.4f} mm")
                lines.append(f"标准差: {signed.get('std', 'N/A'):.4f} mm")
                lines.append(f"P95: {signed.get('p95', 'N/A'):.4f} mm")
                lines.append(f"P99: {signed.get('p99', 'N/A'):.4f} mm")
                lines.append(f"正偏差点数: {scan_to_cad.get('positive_count', 'N/A')} (均值 {scan_to_cad.get('positive_mean', 'N/A'):.4f} mm)")
                lines.append(f"负偏差点数: {scan_to_cad.get('negative_count', 'N/A')} (均值 {scan_to_cad.get('negative_mean', 'N/A'):.4f} mm)")
            else:
                lines.append("  (scan_to_cad 数据缺失)")
            lines.append("")

            lines.append("--- 4. 偏差分析 (CAD → Scan) ---")
            if cad_to_scan:
                signed = cad_to_scan.get("signed", {})
                unsigned = cad_to_scan.get("unsigned", {})
                lines.append(f"点数: {cad_to_scan.get('count', 'N/A')}")
                lines.append(f"Signed 均值: {signed.get('mean', 'N/A'):.4f} mm")
                lines.append(f"Signed 中位数: {signed.get('median', 'N/A'):.4f} mm")
                lines.append(f"RMS: {signed.get('rms', 'N/A'):.4f} mm")
                lines.append(f"Unsigned 均值: {unsigned.get('mean', 'N/A'):.4f} mm")
                lines.append(f"Unsigned 最大值: {unsigned.get('max', 'N/A'):.4f} mm")
            else:
                lines.append("  (cad_to_scan 数据缺失)")
            lines.append("")

            # ---------------------------------------------------------
            # 4b. CAD 表面覆盖度 (缺失面检测)
            # ---------------------------------------------------------
            coverage = dev_data.get("coverage", {})
            if coverage:
                total = coverage.get("covered_count", 0) + coverage.get("uncovered_count", 0)
                lines.append("--- 4b. CAD 表面覆盖度 (缺失面检测) ---")
                lines.append(f"覆盖判定阈值: {coverage.get('threshold_mm', 'N/A')} mm")
                lines.append(
                    f"覆盖率: {coverage.get('covered_ratio', 0) * 100:.2f}% "
                    f"({coverage.get('covered_count', 0)}/{total})"
                )
                lines.append(
                    f"覆盖区域偏差均值: {coverage.get('covered_mean', 'N/A')} mm | "
                    f"RMS: {coverage.get('covered_rms', 'N/A')} mm"
                )
                if coverage.get("uncovered_mean_distance") is not None:
                    lines.append(
                        f"未覆盖区域平均距离: {coverage.get('uncovered_mean_distance')} mm "
                        f"(疑似贴桌面/遮挡等缺失面)"
                    )
                lines.append("")

            # ---------------------------------------------------------
            # 5. 公差判定
            # ---------------------------------------------------------
            lines.append("--- 5. 公差判定 ---")
            lines.append(f"判定标准: ±{tolerance_mm:.2f} mm")

            if scan_to_cad and "signed" in scan_to_cad:
                signed = scan_to_cad["signed"]
                p95 = signed.get("p95", None)
                p99 = signed.get("p99", None)
                max_dev = signed.get("max", None)
                min_dev = signed.get("min", None)

                if p95 is not None:
                    lines.append(f"P95 偏差: {p95:.4f} mm {'(在公差内)' if abs(p95) <= tolerance_mm else '(超出公差)'}")
                if p99 is not None:
                    lines.append(f"P99 偏差: {p99:.4f} mm {'(在公差内)' if abs(p99) <= tolerance_mm else '(超出公差)'}")

                # 工业质检通常以 P95 为统计判定标准，同时记录极值异常
                stat_pass = p95 is not None and abs(p95) <= tolerance_mm
                strict_pass = (
                    max_dev is not None and min_dev is not None and
                    max_dev <= tolerance_mm and min_dev >= -tolerance_mm
                )

                if stat_pass and strict_pass:
                    status = "PASS ✅ (统计与极值均通过)"
                elif stat_pass and not strict_pass:
                    status = "PASS ⚠️ (P95 通过，但存在极值 outliers 超出公差)"
                else:
                    status = "FAIL ❌ (P95 超出公差)"
                lines.append(f"判定结果: {status}")
            else:
                lines.append("  (无偏差数据，无法判定)")
            lines.append("")
        else:
            lines.append("--- 3. 偏差分析 ---")
            lines.append("  (deviation.json 不存在，跳过)")
            lines.append("")

        # ---------------------------------------------------------
        # 6. 几何公差 (GD&T)
        # ---------------------------------------------------------
        tolerance_json_path = self.out_dir / "tolerance.json"
        if tolerance_json_path.exists():
            with open(tolerance_json_path, "r", encoding="utf-8") as f:
                tol_data = json.load(f)

            lines.append("--- 6. 几何公差 (GD&T) ---")
            for category, results in tol_data.items():
                if not results:
                    continue
                lines.append(f"  [{category.upper()}]")
                for r in results:
                    name = r.get("name", "N/A")
                    value = r.get("value", "N/A")
                    unit = r.get("unit", "mm")
                    status = r.get("status", "N/A")
                    lines.append(f"    {name:40s} {value:10.4f} {unit:6s}  [{status}]")
            lines.append("")
        else:
            lines.append("--- 6. 几何公差 (GD&T) ---")
            lines.append("  (tolerance.json 不存在，跳过)")
            lines.append("")

        # ---------------------------------------------------------
        # 7. 输出文件清单
        # ---------------------------------------------------------
        lines.append("--- 7. 输出文件清单 ---")
        for f in sorted(self.out_dir.iterdir()):
            if f.is_file():
                size_kb = f.stat().st_size / 1024.0
                lines.append(f"  {f.name:30s}  {size_kb:10.2f} KB")
        lines.append("")

        lines.append(sep)
        lines.append("报告生成完毕。")
        lines.append(sep)

        # 写入文件
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        self.logger.info(f"自动化综合检测报告生成成功: {file_path}")
        return str(file_path.absolute())
