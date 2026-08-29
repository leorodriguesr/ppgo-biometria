import asyncio
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.modules.setdefault("insightface", MagicMock())
sys.modules.setdefault("insightface.app", MagicMock())

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from PIL import Image, ImageFilter
from main import (
    MIN_BLUR_VARIANCE,
    POSE_FRONT,
    POSE_LEFT,
    POSE_RIGHT,
    blur_variance,
    collect_pose_errors,
    cosine_similarity,
    crop_face_region,
)


class FaceMathTests(unittest.TestCase):
    def test_identical_embeddings_are_one(self):
        vec = np.array([0.2, 0.4, 0.8], dtype=np.float64)
        self.assertAlmostEqual(cosine_similarity(vec, vec), 1.0, places=6)

    def test_orthogonal_embeddings_are_zero(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        self.assertAlmostEqual(cosine_similarity(a, b), 0.0, places=6)

    def test_front_pose_accepts_small_yaw(self):
        self.assertEqual(collect_pose_errors(POSE_FRONT, 12.0), [])

    def test_front_pose_rejects_profile_yaw(self):
        self.assertEqual(collect_pose_errors(POSE_FRONT, 35.0), ["WRONG_POSE"])

    def test_left_profile_requires_negative_yaw(self):
        self.assertEqual(collect_pose_errors(POSE_LEFT, -40.0), [])
        self.assertEqual(collect_pose_errors(POSE_LEFT, 10.0), ["WRONG_POSE"])

    def test_right_profile_requires_positive_yaw(self):
        self.assertEqual(collect_pose_errors(POSE_RIGHT, 40.0), [])
        self.assertEqual(collect_pose_errors(POSE_RIGHT, -10.0), ["WRONG_POSE"])


class UploadGuardTests(unittest.TestCase):
    def test_rejects_non_image_extension(self):
        from fastapi import HTTPException
        from main import validate_image_upload

        file = MagicMock()
        file.filename = "face.txt"
        file.content_type = "text/plain"
        with self.assertRaises(HTTPException) as raised:
            validate_image_upload(file)
        self.assertEqual(raised.exception.detail["code"], "INVALID_FILE")

    def test_accepts_jpeg(self):
        from main import validate_image_upload

        file = MagicMock()
        file.filename = "front1.jpg"
        file.content_type = "image/jpeg"
        validate_image_upload(file)

    def test_rejects_oversized_upload(self):
        from fastapi import HTTPException
        from main import MAX_UPLOAD_BYTES, read_image_bytes

        async def _too_large():
            file = MagicMock()

            async def _read(_n):
                return b"x" * (MAX_UPLOAD_BYTES + 2)

            file.read = _read
            with self.assertRaises(HTTPException) as raised:
                await read_image_bytes(file)
            self.assertEqual(raised.exception.detail["code"], "INVALID_FILE")

        asyncio.run(_too_large())


class BlurMetricTests(unittest.TestCase):
    def _checkerboard(self, size: int, cell: int) -> np.ndarray:
        yy, xx = np.indices((size, size))
        pattern = ((xx // cell) + (yy // cell)) % 2
        gray = (pattern * 255).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)

    def _box_blur(self, img: np.ndarray, radius: int = 12) -> np.ndarray:
        return np.array(Image.fromarray(img).filter(ImageFilter.BoxBlur(radius)))

    def test_sharp_high_res_photo_passes_threshold(self):
        sharp = self._checkerboard(2400, 8)
        self.assertGreater(blur_variance(sharp), MIN_BLUR_VARIANCE)

    def test_heavily_blurred_photo_fails_threshold(self):
        sharp = self._checkerboard(800, 8)
        blurred = self._box_blur(sharp, radius=16)
        self.assertLess(blur_variance(blurred), MIN_BLUR_VARIANCE)

    def test_face_crop_ignores_smooth_background(self):
        canvas = np.full((2000, 1600, 3), 140, dtype=np.uint8)
        face = self._checkerboard(280, 6)
        canvas[200:480, 660:940] = face
        full = blur_variance(canvas)
        cropped = blur_variance(crop_face_region(canvas, [660, 200, 940, 480]))
        self.assertGreater(cropped, full)
        self.assertGreater(cropped, MIN_BLUR_VARIANCE)


class SsoAuthTests(unittest.TestCase):
    def setUp(self):
        from main import _sso_token_cache

        _sso_token_cache.clear()
        self._prev_require = os.environ.get("REQUIRE_SSO")
        os.environ["REQUIRE_SSO"] = "true"

    def tearDown(self):
        from main import _sso_token_cache

        _sso_token_cache.clear()
        if self._prev_require is None:
            os.environ.pop("REQUIRE_SSO", None)
        else:
            os.environ["REQUIRE_SSO"] = self._prev_require

    def test_extracts_bearer_and_token_header(self):
        from main import extract_request_token

        self.assertEqual(extract_request_token("Bearer abc", None), "abc")
        self.assertEqual(extract_request_token(None, "xyz"), "xyz")

    def test_missing_token_is_unauthorized(self):
        from fastapi import HTTPException
        from main import require_sso_token

        with self.assertRaises(HTTPException) as raised:
            asyncio.run(require_sso_token(None, None))
        self.assertEqual(raised.exception.status_code, 401)
        self.assertEqual(raised.exception.detail["code"], "UNAUTHORIZED")

    def test_invalid_sso_token_is_unauthorized(self):
        from unittest.mock import patch

        from fastapi import HTTPException
        from main import require_sso_token

        with patch("main.call_sso_validate", return_value=False):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(require_sso_token("Bearer invalido", None))
        self.assertEqual(raised.exception.status_code, 401)

    def test_valid_sso_token_passes(self):
        from unittest.mock import patch

        from main import require_sso_token

        with patch("main.call_sso_validate", return_value=True):
            asyncio.run(require_sso_token("Bearer valido", None))

    def test_can_disable_sso_for_local_dev(self):
        from main import require_sso_token

        os.environ["REQUIRE_SSO"] = "false"
        asyncio.run(require_sso_token(None, None))


if __name__ == "__main__":
    unittest.main()
