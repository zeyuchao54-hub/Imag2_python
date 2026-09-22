"""#2: STEP 导出必须诚实——没有 CAD 后端时不得返回路径、不得落盘。"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import tempfile  # noqa: E402
import unittest  # noqa: E402
import logging  # noqa: E402

import numpy as np  # noqa: E402

from report import ReportGenerator  # noqa: E402


class TestStepExportHonesty(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.reporter = ReportGenerator(out_dir=self.tmp.name)
        self.vertices = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        # 捕获本测试期间的告警日志
        self.records = []

        handler = logging.Handler()
        handler.emit = lambda record: self.records.append(record)
        self.logger = logging.getLogger("PointToCAD_System.Reporter")
        self.logger.addHandler(handler)

    def tearDown(self):
        self.tmp.cleanup()

    def test_returns_none_and_writes_nothing_without_backend(self):
        """当前环境 (无 cadquery/OCP/OCC) 下必须返回 None 且不创建任何文件。"""
        result = self.reporter.export_step(self.vertices, planes=[])

        self.assertIsNone(result, "无 CAD 后端时 export_step 必须返回 None，而不是返回一个不存在的路径")
        self.assertEqual(
            os.listdir(self.tmp.name), [],
            "STEP 导出被跳过时，输出目录里不应出现任何文件"
        )

    def test_warning_is_explicit(self):
        self.reporter.export_step(self.vertices, planes=[])
        warnings = [r for r in self.records if r.levelno >= logging.WARNING]
        self.assertTrue(warnings, "跳过 STEP 导出时必须发出 WARNING")

        text = " ".join(r.getMessage() for r in warnings)
        self.assertIn("STEP", text)
        self.assertIn("未被创建", text)

    def test_backend_detection_agrees_with_environment(self):
        """后端探测结果必须与实际可 import 的模块一致。"""
        detected = ReportGenerator._detect_cad_backend()
        importable = [n for n in ReportGenerator._CAD_BACKENDS
                      if self._can_import(n)]
        expected = importable[0] if importable else None
        self.assertEqual(detected, expected)

    @staticmethod
    def _can_import(name):
        try:
            __import__(name)
            return True
        except ImportError:
            return False

    def test_never_returns_a_path_to_a_nonexistent_file(self):
        """无论走哪条分支，返回值要么是 None，要么必须指向真实存在的文件。"""
        result = self.reporter.export_step(self.vertices, planes=[])
        if result is not None:
            self.assertTrue(os.path.isfile(result),
                            f"export_step 返回了 {result}，但该文件并不存在")


if __name__ == "__main__":
    unittest.main(verbosity=2)
