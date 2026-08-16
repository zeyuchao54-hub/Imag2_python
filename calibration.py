import re
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np


@dataclass
class ScaleInfo:
    """
    标定结果元数据对象 (数据传递与学术/工程 JSON 序列化)
    """
    factor: float = 1.0
    reference: str = "Uncalibrated (Default 1.0)"
    feature_type: str = "None"
    real_distance_mm: float = 0.0
    virtual_distance: float = 0.0
    input_unit: str = "mm"

    def to_dict(self) -> dict:
        """输出适合 JSON 报告和论文引用的结构化字典"""
        return {
            "factor": round(self.factor, 6),
            "reference": self.reference,
            "feature_type": self.feature_type,
            "real_mm": round(self.real_distance_mm, 4),
            "virtual": round(self.virtual_distance, 6),
            "standard_unit": "mm",
            "user_input_unit": self.input_unit
        }


class ScaleCalibrator:
    """
    工业级物理比例尺标定 Engine
    支持: 几何特征智能聚类降维、单位自动换算解析、标定元数据封装
    """

    def __init__(self, cluster_tolerance_ratio: float = 0.03):
        """
        :param cluster_tolerance_ratio: 几何线段聚类容差比例 (默认 3% 差异内的线段归为同一种特征簇)
        """
        self.logger = logging.getLogger("PointToCAD_System.Calibration")
        self.cluster_tolerance_ratio = cluster_tolerance_ratio

    def interactive_calibrate(self, vertices: np.ndarray, planes: list) -> ScaleInfo:
        """
        交互式标定入口
        """
        print("\n" + "=" * 65)
        print("【工业级物理比例尺标定 Engine (Interactive Mode)】")
        print("=" * 65)

        # 1. 提取并智能聚类特征
        clusters = self._extract_and_cluster_features(vertices, planes)

        if not clusters:
            self.logger.warning("未检测到有效几何特征，保持默认比例 (Factor=1.0)。")
            return ScaleInfo()

        # 2. 打印智能归并后的特征选项 (解决方案四: 避免选项爆炸)
        print("\n [提取出的关键几何特征类别 (已自动聚类去重)]:")
        for idx, clst in enumerate(clusters, start=1):
            sample_pairs = ", ".join(clst["pair_labels"][:3])
            if len(clst["pair_labels"]) > 3:
                sample_pairs += f" 等共 {len(clst['pair_labels'])} 组"

            print(
                f"  [{idx}] 【{clst['category']}】 "
                f"虚拟均值: {clst['avg_virtual_dist']:.4f} | "
                f"样例: ({sample_pairs})"
            )

        # 3. 用户选择特征类别
        selected_cluster = None
        while True:
            try:
                user_input = input(f"\n 请选择参照特征编号 (1 ~ {len(clusters)}) [直接回车或输入 0 跳过]: ").strip()
                if user_input == "" or user_input == "0":
                    print("跳过交互标定，保持默认比例 1.0。")
                    return ScaleInfo()

                choice = int(user_input)
                if 1 <= choice <= len(clusters):
                    selected_cluster = clusters[choice - 1]
                    print(f"已选中: 【{selected_cluster['category']}】(虚拟距离 = {selected_cluster['avg_virtual_dist']:.4f})")
                    break
                else:
                    print(f"❌ 编号超出范围，请输入 1 ~ {len(clusters)} 之间的整数。")
            except ValueError:
                print("❌ 无效输入，请输入数字。")

        # 4. 用户输入真实物理尺寸 (解决方案三: 自动单位解析)
        real_mm, input_unit = self._prompt_real_distance()

        # 5. 计算并封装 ScaleInfo (解决方案二: 返回数据对象)
        virtual_dist = selected_cluster["avg_virtual_dist"]
        scale_factor = real_mm / virtual_dist

        scale_info = ScaleInfo(
            factor=scale_factor,
            reference=selected_cluster["pair_labels"][0],
            feature_type=selected_cluster["category"],
            real_distance_mm=real_mm,
            virtual_distance=virtual_dist,
            input_unit=input_unit
        )

        print("\n" + "-" * 65)
        print(f"标定成功！Scale Factor = {scale_info.factor:.6f}")
        print(f"   (基准: {scale_info.reference} | 真实尺寸: {scale_info.real_distance_mm} mm)")
        print("=" * 65 + "\n")

        return scale_info

    def _extract_and_cluster_features(self, vertices: np.ndarray, planes: list) -> List[dict]:
        """
        核心算法: 计算所有顶点对距离，并按照相对长度差进行 1D 聚类
        """
        raw_pairs = []

        # 计算所有顶点间的距离
        if len(vertices) >= 2:
            for i in range(len(vertices)):
                for j in range(i + 1, len(vertices)):
                    dist = float(np.linalg.norm(vertices[i] - vertices[j]))
                    raw_pairs.append({
                        "type": "Vertex-Vertex",
                        "label": f"Vertex_{i+1}<->Vertex_{j+1}",
                        "dist": dist
                    })

        if not raw_pairs:
            return []

        # 按距离从小到大排序
        raw_pairs.sort(key=lambda x: x["dist"])

        # 贪心聚类: 距离差异小于 ratio 的归为同一类别
        clusters = []
        for pair in raw_pairs:
            matched = False
            for clst in clusters:
                # 检查当前距离是否与该聚类均值贴近
                if abs(pair["dist"] - clst["avg_virtual_dist"]) / clst["avg_virtual_dist"] <= self.cluster_tolerance_ratio:
                    clst["members"].append(pair)
                    # 动态更新聚类均值
                    clst["avg_virtual_dist"] = float(np.mean([m["dist"] for m in clst["members"]]))
                    clst["pair_labels"].append(pair["label"])
                    matched = True
                    break

            if not matched:
                clusters.append({
                    "avg_virtual_dist": pair["dist"],
                    "members": [pair],
                    "pair_labels": [pair["label"]]
                })

        # 为每个聚类赋予直观的几何名称 (如: 主特征边、次长对角线等)
        for idx, clst in enumerate(clusters):
            count = len(clst["members"])
            clst["category"] = f"特征簇 {idx + 1} [均值: {clst['avg_virtual_dist']:.3f}, 包含 {count} 条同长线段]"

        return clusters

    def _prompt_real_distance(self) -> Tuple[float, str]:
        """
        解析包含单位的字符串输入 (例如: "70mm", "7 cm", "0.07m") 并统一换算为 mm
        """
        while True:
            raw_str = input("请输入该特征的【真实物理尺寸】(支持 70mm, 7cm, 0.07m 等): ").strip().lower()
            parsed = self.parse_length_with_unit(raw_str)
            if parsed is not None:
                return parsed
            print("❌ 无法解析输入！示例合法格式: '70mm', '7cm', '0.07m', 或纯数字 '70'。")

    @staticmethod
    def parse_length_with_unit(user_input: str) -> Optional[Tuple[float, str]]:
        """
        单位自动换算静态工具函数
        """
        if not user_input:
            return None

        # 正则匹配 数字 + 可选的单位
        match = re.match(r"^([0-9.]+)\s*([a-zA-Z]*|cm|mm|m)$", user_input)
        if not match:
            return None

        val_str, unit_str = match.groups()
        try:
            val = float(val_str)
            if val <= 0:
                return None
        except ValueError:
            return None

        unit = unit_str.strip() if unit_str else "mm"

        # 换算逻辑矩阵 (统一到 mm)
        scale_map = {
            "": 1.0, "mm": 1.0, "毫米": 1.0,
            "cm": 10.0, "厘米": 10.0,
            "m": 1000.0, "米": 1000.0,
            "in": 25.4, "inch": 25.4, "英寸": 25.4
        }

        if unit in scale_map:
            val_in_mm = val * scale_map[unit]
            return val_in_mm, unit
        else:
            return None