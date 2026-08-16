import logging
from typing import List, Optional
import numpy as np
import open3d as o3d
from plane import Plane


class Visualizer:
    """
    工业级 3D 可视化渲染引擎
    职责: 将抽象的数学平面、CAD 顶点与原始残余点云进行高亮渲染与组合展示
    """

    def __init__(self, sphere_radius=0.02, show_origin=True):
        """
        :param sphere_radius: 渲染 CAD 角点时使用的红色小球半径 (根据实际点云尺寸调整)
        :param show_origin: 是否在画面中心显示 XYZ 世界坐标系
        """
        self.logger = logging.getLogger("PointToCAD_System.Visualizer")
        self.sphere_radius = sphere_radius
        self.show_origin = show_origin

        # 预设的高对比度工业配色表 (RGB格式, 范围 0-1)
        self.color_palette = [
            [0.8, 0.2, 0.2],  # 工业红
            [0.2, 0.8, 0.2],  # 荧光绿
            [0.2, 0.2, 0.8],  # 宝玉蓝
            [0.8, 0.8, 0.2],  # 警示黄
            [0.2, 0.8, 0.8],  # 青色
            [0.8, 0.2, 0.8],  # 品红
            [0.9, 0.5, 0.1],  # 亮橙
            [0.5, 0.1, 0.9],  # 深紫
        ]

    def draw_scene(self, planes: List[Plane], vertices: np.ndarray, rest_pcd: Optional[o3d.geometry.PointCloud] = None):
        """
        组合并渲染整个 3D 场景
        """
        self.logger.info("正在准备 3D 渲染数据...")

        geometries_to_draw = []

        # 1. 渲染环境残余点云 (半透明/灰色，作为背景衬托)
        if rest_pcd is not None and len(rest_pcd.points) > 0:
            rest_pcd.paint_uniform_color([0.6, 0.6, 0.6])  # 统一涂成中灰色
            geometries_to_draw.append(rest_pcd)

        # 2. 渲染提取出的平面 (赋予不同的高亮颜色)
        for idx, plane in enumerate(planes):
            color = self._get_color(idx)
            # 拷贝一份点云用于显示，避免修改原始数据
            display_cloud = plane.cloud
            display_cloud.paint_uniform_color(color)
            geometries_to_draw.append(display_cloud)

            # 可选: 同时画出平面的物理边界框 (OBB)
            obb = plane.obb
            obb.color = color
            geometries_to_draw.append(obb)

        # 3. 渲染计算出的 CAD 直角顶点 (用高亮红色小球表示)
        if vertices is not None and len(vertices) > 0:
            sphere_meshes = self._create_vertex_spheres(vertices)
            geometries_to_draw.extend(sphere_meshes)

        # 4. 添加世界坐标系原点
        if self.show_origin:
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.5, origin=[0, 0, 0]
            )
            geometries_to_draw.append(coordinate_frame)

        # 5. 启动 Open3D 渲染引擎
        self.logger.info("启动渲染窗口。提示: 使用鼠标左键旋转，右键平移，滚轮缩放。关闭窗口以退出程序。")
        o3d.visualization.draw_geometries(
            geometries_to_draw,
            window_name="PointToCAD 工业级逆向工程预览",
            width=1280,
            height=720,
            point_show_normal=False
        )

    def _get_color(self, index: int) -> list:
        """从预设调色板中安全获取颜色，如果平面过多则循环使用配色"""
        palette_size = len(self.color_palette)
        return self.color_palette[index % palette_size]

    def _create_vertex_spheres(self, vertices: np.ndarray) -> List[o3d.geometry.TriangleMesh]:
        """
        纯数学坐标是一个“没有体积的理想点”，肉眼是看不见的。
        这个函数在每个顶点坐标的位置生成一个 3D 红色小球，方便人类观察。
        """
        spheres = []
        for v in vertices:
            # 创建标准球体网格
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.sphere_radius)

            # 将球体移动到目标顶点位置
            sphere.translate(v)

            # 涂成极其醒目的正红色并计算法线（为了光照渲染效果更好）
            sphere.paint_uniform_color([1.0, 0.0, 0.0])
            sphere.compute_vertex_normals()

            spheres.append(sphere)

        return spheres

    def draw_deviation(
            self,
            scan_colored: o3d.geometry.PointCloud,
            cad_colored: Optional[o3d.geometry.PointCloud] = None,
            window_name: str = "PointToCAD 偏差色谱 (蓝=负/内凹, 红=正/外凸)",
    ):
        """
        渲染偏差色谱图。

        :param scan_colored: 已经按 signed deviation 着色的扫描点云
        :param cad_colored:  已经按 signed deviation 着色的 CAD 点云 (可选)
        """
        self.logger.info("正在准备偏差色谱 3D 渲染数据...")

        geometries_to_draw = [scan_colored]

        # 可选: 同时显示 CAD 点云作为参考
        if cad_colored is not None and len(cad_colored.points) > 0:
            geometries_to_draw.append(cad_colored)

        # 添加世界坐标系原点
        if self.show_origin:
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=10.0, origin=[0, 0, 0]
            )
            geometries_to_draw.append(coordinate_frame)

        self.logger.info("启动偏差色谱渲染窗口。关闭窗口以退出程序。")
        o3d.visualization.draw_geometries(
            geometries_to_draw,
            window_name=window_name,
            width=1280,
            height=720,
            point_show_normal=False,
        )
