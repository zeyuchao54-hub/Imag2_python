import logging
from typing import List, Dict, Tuple
from plane import Plane


class PlaneGraph:
    """
    平面关系图数据结构
    存储平面节点(Nodes)以及它们之间的几何约束关系(Edges)
    """

    def __init__(self, planes: List[Plane]):
        # 节点：以平面 ID 为 Key 的字典，方便快速索引
        self.nodes: Dict[int, Plane] = {p.id: p for p in planes}

        # 边：记录平面之间的关系，格式为 (id1, id2) -> {"type": "垂直", "angle": 89.5}
        self.edges: Dict[Tuple[int, int], dict] = {}

    def add_edge(self, id1: int, id2: int, rel_type: str, angle: float):
        """添加无向边"""
        self.edges[(id1, id2)] = {"type": rel_type, "angle": angle}
        self.edges[(id2, id1)] = {"type": rel_type, "angle": angle}

    def get_relationship(self, id1: int, id2: int) -> str:
        """查询两个面之间的拓扑关系"""
        if (id1, id2) in self.edges:
            return self.edges[(id1, id2)]["type"]
        return "未知"

    def find_perpendicular_triplets(self) -> List[Tuple[Plane, Plane, Plane]]:
        """
        核心算法：在关系图中寻找“互相垂直的三个平面”组合。
        (这在工业中代表一个标准的直角顶点，比如立方体的角)
        """
        triplets = []
        plane_ids = list(self.nodes.keys())
        n = len(plane_ids)

        # 遍历所有可能的三面组合
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    id_a, id_b, id_c = plane_ids[i], plane_ids[j], plane_ids[k]

                    # 检查这三个面是否两两互相垂直
                    if (self.get_relationship(id_a, id_b) == "perpendicular" and
                            self.get_relationship(id_b, id_c) == "perpendicular" and
                            self.get_relationship(id_c, id_a) == "perpendicular"):
                        triplets.append((self.nodes[id_a], self.nodes[id_b], self.nodes[id_c]))

        return triplets

    def find_independent_triplets(self) -> List[Tuple[Plane, Plane, Plane]]:
        """
        寻找“三个面互不平行”的组合。
        实际扫描点云的平面法向存在噪声，不一定严格垂直，但只要三个面互不平行，
        它们的交点仍对应一个有效的几何顶点（如立方体的角）。
        """
        triplets = []
        plane_ids = list(self.nodes.keys())
        n = len(plane_ids)

        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    id_a, id_b, id_c = plane_ids[i], plane_ids[j], plane_ids[k]

                    # 只要三个面没有任意两个是平行的，就视为一个有效顶点候选
                    if (self.get_relationship(id_a, id_b) != "parallel" and
                            self.get_relationship(id_b, id_c) != "parallel" and
                            self.get_relationship(id_c, id_a) != "parallel"):
                        triplets.append((self.nodes[id_a], self.nodes[id_b], self.nodes[id_c]))

        return triplets


class PlaneGraphBuilder:
    """
    工业级拓扑关系构建器
    职责: 分析所有 Plane 对象，计算空间夹角，生成 PlaneGraph
    """

    def __init__(self, parallel_thresh=5.0, perp_thresh=5.0):
        """
        :param parallel_thresh: 判定平行的角度容差 (例如 < 5度 或 > 175度)
        :param perp_thresh: 判定垂直的角度容差 (例如 90±5度)
        """
        self.logger = logging.getLogger("PointToCAD_System.Graph")
        self.parallel_thresh = parallel_thresh
        self.perp_thresh = perp_thresh

    def build(self, planes: List[Plane]) -> PlaneGraph:
        """
        构建并返回平面图
        """
        self.logger.info(f"开始构建拓扑关系图 (节点数: {len(planes)})...")
        graph = PlaneGraph(planes)

        parallel_count = 0
        perp_count = 0
        intersect_count = 0

        # 两两计算平面夹角
        for i in range(len(planes)):
            for j in range(i + 1, len(planes)):
                p1 = planes[i]
                p2 = planes[j]

                angle = p1.angle_with(p2)

                # 判定关系分类
                rel_type = self._classify_relationship(angle)

                graph.add_edge(p1.id, p2.id, rel_type, angle)

                if rel_type == "parallel":
                    parallel_count += 1
                elif rel_type == "perpendicular":
                    perp_count += 1
                else:
                    intersect_count += 1

        self.logger.info(
            f"图构建完成: 发现 {parallel_count} 对平行面，{perp_count} 对垂直面，{intersect_count} 对普通相交面。"
        )
        return graph

    def _classify_relationship(self, angle: float) -> str:
        """根据夹角判断拓扑关系类型"""
        # 趋近于 0 度或 180 度，视为平行
        if angle <= self.parallel_thresh or angle >= (180.0 - self.parallel_thresh):
            return "parallel"

        # 趋近于 90 度，视为垂直
        elif (90.0 - self.perp_thresh) <= angle <= (90.0 + self.perp_thresh):
            return "perpendicular"

        # 其他角度视为普通的斜交
        else:
            return "intersecting"