import csv
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from main import PlanarTracker, track


class TrackerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(52)
        cls.image = np.full((240, 360, 3), 90, np.uint8)
        for _ in range(600):
            x, y = rng.integers([0, 0], [360, 240])
            color = tuple(int(c) for c in rng.integers(0, 256, 3))
            cv2.circle(cls.image, (int(x), int(y)), int(rng.integers(2, 8)), color, -1)

    def setUp(self):
        cv2.setRNGSeed(42)

    def assert_corners(self, polygon, expected, tolerance=2):
        self.assertIsNotNone(polygon)
        self.assertLess(float(np.linalg.norm(polygon - expected, axis=2).mean()), tolerance)

    def test_scale_rotation_and_perspective(self):
        tracker = PlanarTracker(self.image, (0, 0))
        methods = set()
        for i, scale in enumerate(np.linspace(1, 0.2, 45)):
            affine = cv2.getRotationMatrix2D((180, 120), i * 0.5, float(scale))
            affine[:, 2] += [100 * (1 - scale), 60 * (1 - scale)]
            matrix = np.vstack((affine, [0.000003 * i, 0.000002 * i, 1]))
            frame = cv2.warpPerspective(self.image, matrix, (360, 240))
            polygon, points, _, _ = tracker.update(frame)
            self.assert_corners(polygon, cv2.perspectiveTransform(tracker.corners, matrix))
            self.assertGreaterEqual(len(points), 8)
            methods.add(tracker.method)
        self.assertTrue({"flow", "sift"}.issubset(methods))

    def test_loss_and_reacquisition(self):
        tracker = PlanarTracker(self.image, (0, 0))
        tracker.update(self.image)
        polygon, points, _, _ = tracker.update(np.zeros_like(self.image))
        self.assertIsNone(polygon)
        self.assertEqual(len(points), 0)
        matrix = np.array([[0.5, -0.1, 80.], [0.1, 0.5, 40.], [0, 0, 1.]])
        frame = cv2.warpPerspective(self.image, matrix, (360, 240))
        polygon, _, _, _ = tracker.update(frame)
        self.assert_corners(polygon, cv2.perspectiveTransform(tracker.corners, matrix))
        self.assertEqual(tracker.method, "sift")

    def test_roi_coordinates(self):
        frame = np.zeros((320, 480, 3), np.uint8)
        frame[40:280, 70:430] = self.image
        tracker = PlanarTracker(self.image, (70, 40))
        polygon, _, _, _ = tracker.update(frame)
        expected = tracker.corners + np.float32([70, 40])
        self.assert_corners(polygon, expected)
        polygon, _, _, _ = tracker.update(frame)
        self.assert_corners(polygon, expected)

    def test_featureless_template_is_rejected(self):
        with self.assertRaises(ValueError):
            PlanarTracker(np.zeros_like(self.image), (0, 0))

    def test_video_and_metrics_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.avi"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 25, (360, 240))
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(self.image)
            writer.release()
            output = root / "nested" / "result"
            stats = track(source, output, label="Test")
            with (output / "input_frames.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), stats["frames"])
            self.assertEqual(stats["detected"], 3)
            self.assertEqual(json.loads((output / "input_stats.json").read_text()), stats)
            capture = cv2.VideoCapture(str(output / "input_tracked.avi"))
            count = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                self.assertEqual(frame.shape, self.image.shape)
                count += 1
            capture.release()
            self.assertEqual(count, 3)


if __name__ == "__main__":
    unittest.main()
