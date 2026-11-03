"""Поиск плоского объекта по ключевым точкам первого кадра."""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np


class PlanarTracker:
    """Масштабные эталоны SIFT и оптический поток с проверкой геометрии."""

    def __init__(self, template, origin):
        self.sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.015)
        self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
        self.template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        h, w = self.template.shape
        self.corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
        points, descriptors = [], []
        for scale in (1.0, 0.65, 0.4, 0.25, 0.16, 0.1):
            if min(h, w) * scale < 24:
                continue
            small = cv2.resize(self.template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            keypoints, features = self.sift.detectAndCompute(small, None)
            if features is not None:
                points.extend([(p.pt[0] * w / small.shape[1], p.pt[1] * h / small.shape[0])
                               for p in keypoints])
                descriptors.append(features)
        if not points or len(points) < 8:
            raise ValueError("На первом кадре мало ключевых точек. Выберите текстурный объект.")
        self.reference = np.float32(points)
        self.descriptors = np.vstack(descriptors)
        self.matrix = np.array([[1., 0., origin[0]], [0., 1., origin[1]], [0., 0., 1.]])
        self.previous = None
        self.source = self.target = None
        self.frame_index = 0
        self.method = "initial"

    def estimate(self, source, target, shape):
        if len(source) < 8:
            return None
        matrix, mask = cv2.findHomography(source, target, cv2.USAC_MAGSAC, 3.0,
                                          maxIters=5000, confidence=0.999)
        if matrix is None or mask is None:
            return None
        keep = mask.ravel().astype(bool)
        if keep.sum() < 8 or keep.mean() < 0.4:
            return None
        polygon = cv2.perspectiveTransform(self.corners, matrix)
        h, w = shape
        area = cv2.contourArea(polygon)
        if (not np.isfinite(polygon).all() or np.abs(polygon).max() > 10 * max(h, w)
                or not cv2.isContourConvex(polygon) or not 100 < area < 2 * w * h):
            return None
        # Точки в одном крошечном участке не определяют надёжную рамку.
        if cv2.contourArea(cv2.convexHull(target[keep])) < 0.015 * area:
            return None
        return matrix, source[keep], target[keep]

    def detect(self, gray):
        keypoints, features = self.sift.detectAndCompute(gray, None)
        if features is None or len(features) < 2:
            return None, 0
        pairs = self.matcher.knnMatch(self.descriptors, features, k=2)
        matches = sorted((pair[0] for pair in pairs
                          if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance),
                         key=lambda m: m.distance)
        # Разные масштабы эталона не должны голосовать одной точкой кадра несколько раз.
        unique, used = [], set()
        for match in matches:
            if match.trainIdx not in used:
                used.add(match.trainIdx)
                unique.append(match)
        source = self.reference[[m.queryIdx for m in unique]]
        target = np.float32([keypoints[m.trainIdx].pt for m in unique]).reshape(-1, 2)
        return self.estimate(source, target, gray.shape), len(unique)

    def flow(self, gray):
        if self.previous is None or self.target is None or len(self.target) < 8:
            return None, 0
        options = dict(winSize=(21, 21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        points = self.target.reshape(-1, 1, 2)
        moved, forward, error = cv2.calcOpticalFlowPyrLK(self.previous, gray, points, None, **options)
        if moved is None:
            return None, 0
        back, backward, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, moved, None, **options)
        if back is None:
            return None, 0
        h, w = gray.shape
        xy = moved.reshape(-1, 2)
        keep = ((forward.ravel() == 1) & (backward.ravel() == 1)
                & (np.linalg.norm(back - points, axis=2).ravel() < 1.0)
                & (error.ravel() < 25) & np.isfinite(xy).all(axis=1)
                & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h))
        return self.estimate(self.source[keep], xy[keep], gray.shape), int(keep.sum())

    def seed_points(self, gray):
        polygon = cv2.perspectiveTransform(self.corners, self.matrix)
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, np.int32(polygon), 255)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
        for px, py in self.target:
            cv2.circle(mask, (round(float(px)), round(float(py))), 4, 0, -1)
        points = cv2.goodFeaturesToTrack(gray, maxCorners=250, qualityLevel=0.01,
                                        minDistance=4, mask=mask, blockSize=5)
        if points is not None:
            canonical = cv2.perspectiveTransform(points, np.linalg.inv(self.matrix)).reshape(-1, 2)
            # Новые точки остаются привязаны к координатам исходного объекта.
            self.source = np.vstack((self.source, canonical))
            self.target = np.vstack((self.target, points.reshape(-1, 2)))

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result, matches = self.flow(gray)
        method = "flow"
        if self.frame_index == 0:
            self.source = self.reference.copy()
            self.target = cv2.perspectiveTransform(self.source.reshape(-1, 1, 2), self.matrix).reshape(-1, 2)
            result = self.matrix, self.source, self.target
            matches, method = len(self.source), "initial"
        elif result is None or self.frame_index % 5 == 0:
            detection, count = self.detect(gray)
            if detection is not None:
                result, matches, method = detection, count, "sift"
        self.frame_index += 1
        self.previous = gray
        if result is None:
            self.source = self.target = None
            self.method = "lost"
            return None, np.empty((0, 2), np.float32), matches, 0
        self.matrix, self.source, self.target = result
        self.method = method
        inliers = len(self.target)
        visible_points = self.target.copy()
        if len(self.target) < 100:
            self.seed_points(gray)
        return cv2.perspectiveTransform(self.corners, self.matrix), visible_points, matches, inliers


def track(video, output_dir, roi=None, label=""):
    capture = cv2.VideoCapture(str(video))
    writer = None
    try:
        ok, first = capture.read()
        if not ok:
            raise ValueError(f"Не удалось прочитать видео: {video}")
        h, w = first.shape[:2]
        x, y, rw, rh = roi if roi else (0, 0, w, h)
        if min(x, y) < 0 or min(rw, rh) < 2 or x + rw > w or y + rh > h:
            raise ValueError(f"ROI должен находиться внутри кадра {w}x{h}")
        template = first[y:y + rh, x:x + rw]
        cv2.setRNGSeed(42)
        tracker = PlanarTracker(template, (x, y))
        output_dir.mkdir(parents=True, exist_ok=True)
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0
        output = output_dir / f"{video.stem}_tracked.avi"
        if output.resolve() == video.resolve():
            raise ValueError("Путь результата совпадает с исходным видео")
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
        if not writer.isOpened():
            raise ValueError(f"Не удалось открыть файл результата: {output}")
        frame, total, detected = first, 0, 0
        started = time.perf_counter()
        with (output_dir / f"{video.stem}_frames.csv").open("w", newline="", encoding="utf-8") as file:
            log = csv.writer(file)
            log.writerow(["frame", "time_seconds", "detected", "matches", "inliers", "method"])
            while True:
                polygon, points, matches, inliers = tracker.update(frame)
                found = polygon is not None
                log.writerow([total, round(total / fps, 3), int(found), matches, inliers, tracker.method])
                if found:
                    detected += 1
                    polygon = np.int32(polygon)
                    # Прореживаем только отрисовку: все точки участвуют в трекинге.
                    step = max(1, int(np.ceil(len(points) / 150)))
                    for px, py in points[::step]:
                        cv2.circle(frame, (round(float(px)), round(float(py))), 1, (0, 255, 255), -1, cv2.LINE_AA)
                    cv2.polylines(frame, [polygon], True, (0, 255, 0), 2, cv2.LINE_AA)
                    if label:
                        tx, ty = polygon[:, 0, :].min(axis=0)
                        cv2.putText(frame, label, (int(np.clip(tx, 0, w - 1)), int(np.clip(ty - 10, 20, h - 1))),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                if total in (0, 30, 100, 200, 300, 400, 500):
                    cv2.imwrite(str(output_dir / f"{video.stem}_{total:04d}.jpg"), frame)
                writer.write(frame)
                total += 1
                ok, frame = capture.read()
                if not ok:
                    break
        elapsed = time.perf_counter() - started
        stats = dict(video=str(video), frames=total, detected=detected,
                     detection_percent=round(100 * detected / total, 2),
                     seconds=round(elapsed, 2), processing_fps=round(total / elapsed, 2),
                     width=w, height=h, source_fps=fps, template_keypoints=len(tracker.reference))
        (output_dir / f"{video.stem}_stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return stats
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Видео или папка с видео")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                        help="Прямоугольник объекта на первом кадре; по умолчанию весь кадр")
    parser.add_argument("--label", default="", help="Необязательная подпись латиницей; по умолчанию без подписи")
    args = parser.parse_args()
    if args.video.is_dir():
        videos = sorted(p for p in args.video.iterdir()
                        if p.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv", ".m4v"})
    else:
        videos = [args.video]
    if not videos:
        parser.error("В папке нет видео")
    for video in videos:
        try:
            print(json.dumps(track(video, args.output, args.roi, args.label), ensure_ascii=False), flush=True)
        except ValueError as error:
            parser.exit(1, f"Ошибка: {error}\n")


if __name__ == "__main__":
    main()
