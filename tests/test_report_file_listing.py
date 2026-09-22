"""#9: 综合报告的"输出文件清单"只能列本次运行生成的文件。"""

import _bootstrap  # noqa: F401  # 必须先于 open3d 导入，固定 OpenMP 线程数
import os
import sys

import tempfile  # noqa: E402
import unittest  # noqa: E402

import numpy as np  # noqa: E402

from report import ReportGenerator  # noqa: E402

class TestWrittenFileTracking(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.reporter = ReportGenerator(out_dir=self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_starts_empty(self):
        self.assertEqual(self.reporter.list_written_files(), [])

    def test_successful_export_is_recorded(self):
        path = self.reporter.export_json(
            planes=[], vertices=np.zeros((0, 3)), filename="report.json"
        )
        written = self.reporter.list_written_files()
        self.assertEqual(len(written), 1)
        self.assertEqual(written[0].name, "report.json")
        self.assertTrue(written[0].exists())

    def test_exports_without_file_are_not_recorded(self):
        """STEP 无后端时不落盘，也不应被记入清单。"""
        self.reporter.export_step(np.zeros((2, 3)), planes=[])
        self.assertEqual(self.reporter.list_written_files(), [])
        self.assertEqual(os.listdir(self.tmp.name), [])

    def test_repeated_export_of_same_file_recorded_once(self):
        """Auto-Scale 会重复导出 fused.ply / report.json，清单不应重复。"""
        from plane import Plane
        import open3d as o3d

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))
        plane = Plane(plane_id=1, model=[0, 0, 1, 0.0], cloud=cloud)

        self.reporter.export_fused_ply([plane], filename="fused.ply")
        self.reporter.export_fused_ply([plane], filename="fused.ply")
        self.assertEqual(len(self.reporter.list_written_files()), 1)

    def test_multiple_distinct_files_all_recorded_in_order(self):
        self.reporter.export_json([], np.zeros((0, 3)), filename="a.json")
        self.reporter.export_json([], np.zeros((0, 3)), filename="b.json")
        names = [f.name for f in self.reporter.list_written_files()]
        self.assertEqual(names, ["a.json", "b.json"])

class TestSummaryListsOnlyThisRun(unittest.TestCase):
    """复用旧输出目录时，清单不得把历史残留算作本次产物。"""

    def test_stale_files_are_not_listed_as_products(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = ReportGenerator(out_dir=tmp)

            # 1) 上一次运行留下的残骸
            stale = os.path.join(tmp, "leftover_from_previous_run.json")
            with open(stale, "w", encoding="utf-8") as f:
                f.write("{}")

            # 2) 本次只导出 report.json
            reporter.export_json([], np.zeros((0, 3)), filename="report.json")

            path = reporter.export_inspection_summary(
                input_file="dummy.ply", stl_file=None,
                icp_init_method="N/A", total_time=1.0,
            )
            with open(path, encoding="utf-8") as fh:
                text = fh.read()

            section = text.split("--- 7. 输出文件清单")[1]

            # 本次产物必须出现
            self.assertIn("report.json", section)
            # 残留文件不得出现在"本次产物"区域，只允许出现在"注: ...并非本次生成"里
            self.assertNotIn("leftover_from_previous_run.json",
                             section.split("注:")[0])
            self.assertIn("并非本次生成", text)
            self.assertIn("leftover_from_previous_run.json", text)

    def test_report_itself_is_listed(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = ReportGenerator(out_dir=tmp)
            reporter.export_json([], np.zeros((0, 3)), filename="report.json")
            path = reporter.export_inspection_summary(
                input_file="d.ply", stl_file=None,
                icp_init_method="N/A", total_time=1.0,
            )
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            section = text.split("--- 7. 输出文件清单")[1].split("注:")[0]
            self.assertIn("report.json", section)
            self.assertIn("inspection_report.txt", section)

    def test_clean_directory_has_no_stale_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = ReportGenerator(out_dir=tmp)
            reporter.export_json([], np.zeros((0, 3)), filename="report.json")
            path = reporter.export_inspection_summary(
                input_file="d.ply", stl_file=None,
                icp_init_method="N/A", total_time=1.0,
            )
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            self.assertNotIn("并非本次生成", text)

if __name__ == "__main__":
    unittest.main(verbosity=2)
