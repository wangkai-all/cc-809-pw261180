"""Web API 合约测试: 真实启动 HTTP 服务器, 用标准库 urllib 请求。"""
import json
import tempfile
import threading
import unittest
import urllib.request
from urllib.error import HTTPError

from ordinal_app import synth
from ordinal_app.store import EventStore
from ordinal_app.web import serve


class WebApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        store = EventStore(cls.tmp.name)
        ds = synth.make_base(seed=7)
        store.import_batch(ds.rows, ds.categories, batch_id="b")
        cls.httpd = serve(store, host="127.0.0.1", port=0)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever,
                                      daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.tmp.cleanup()

    def _get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def _get_raw(self, path):
        with urllib.request.urlopen(self.base + path, timeout=60) as resp:
            return resp.status, resp.read(), resp.headers.get_content_type()

    def test_static_pages(self):
        for p in ["/", "/index.html", "/project.html", "/thresholds.html",
                  "/data.html", "/reports.html", "/styles.css", "/app.js"]:
            status, body, ctype = self._get_raw(p)
            self.assertEqual(status, 200)
            self.assertTrue(len(body) > 100)

    def test_overview(self):
        _, d = self._get("/api/overview")
        self.assertFalse(d["empty"])
        self.assertEqual(len(d["abilities"]), len(d["projects"]))
        self.assertIn("diagnostics", d)
        self.assertIn("fit_stats", d)

    def test_project_probabilities_sum_to_one(self):
        _, d = self._get("/api/project?id=P0")
        self.assertEqual(d["project"], "P0")
        for row in d["probability_rows"]:
            self.assertAlmostEqual(sum(row["probs"]), 1.0, places=9)
            self.assertEqual(len(row["probs"]), len(d["categories"]))
        # 原文本保留
        self.assertTrue(all("raw_label" in r for r in d["observed"]))

    def test_thresholds_and_selection(self):
        _, d = self._get("/api/thresholds")
        self.assertIn(d["selection"]["selected"], ("fixed", "drift"))
        self.assertTrue(d["drift"]["drift_summary"])
        # 阈值严格有序
        K = len(d["categories"])
        for row in d["drift"]["threshold_table"]:
            taus = [row[f"tau_{k}"] for k in range(K - 1)]
            self.assertTrue(all(b > a for a, b in zip(taus, taus[1:])))

    def test_data_and_reports_views(self):
        _, d = self._get("/api/data")
        self.assertTrue(d["records"])
        self.assertTrue(d["versions"])
        _, r = self._get("/api/reports")
        self.assertIn("reports", r)

    def test_404_and_bad_project(self):
        with self.assertRaises(HTTPError) as cm:
            self._get("/api/project?id=DOES_NOT_EXIST")
        self.assertEqual(cm.exception.code, 404)
        with self.assertRaises(HTTPError) as cm:
            self._get("/no-such-page.html")
        self.assertEqual(cm.exception.code, 404)

    def test_path_traversal_blocked(self):
        with self.assertRaises(HTTPError) as cm:
            self._get("/../../etc/passwd")
        self.assertEqual(cm.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
