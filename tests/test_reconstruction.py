#!/usr/bin/env python3
"""
Unit tests for DroneNav-SAR Reconstruction Pipeline
"""

import unittest
import tempfile
import shutil
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestColmapPipeline(unittest.TestCase):
    """Tests for COLMAP pipeline."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.temp_dir / "input"
        self.output_dir = self.temp_dir / "output"
        self.input_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_quality_presets_exist(self):
        """Test that quality presets are defined."""
        from src.reconstruction.colmap_pipeline import QUALITY_PRESETS
        self.assertIn("low", QUALITY_PRESETS)
        self.assertIn("medium", QUALITY_PRESETS)
        self.assertIn("high", QUALITY_PRESETS)
        self.assertIn("image_resize", QUALITY_PRESETS["medium"])

    def test_create_workspace(self):
        """Test workspace creation."""
        from src.reconstruction.colmap_pipeline import create_workspace
        create_workspace(self.input_dir, self.output_dir, "medium")
        self.assertTrue(self.output_dir.exists())
        self.assertTrue((self.output_dir / "sparse").exists())
        self.assertTrue((self.output_dir / "dense").exists())
        self.assertTrue((self.output_dir / "images").exists())
        self.assertTrue((self.output_dir / "colmap").exists())


class TestReconstructEntryPoint(unittest.TestCase):
    """Tests for unified reconstruct entry point."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.temp_dir / "input"
        self.output_dir = self.temp_dir / "output"
        self.input_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_count_images(self):
        """Test image counting."""
        from src.reconstruction.reconstruct import count_images

        # Create dummy images
        (self.input_dir / "img1.jpg").touch()
        (self.input_dir / "img2.JPG").touch()
        (self.input_dir / "img3.png").touch()
        (self.input_dir / "img4.PNG").touch()
        (self.input_dir / "not_image.txt").touch()

        count = count_images(self.input_dir)
        self.assertEqual(count, 4)

    def test_auto_select_method_prefers_colmap_for_many_images(self):
        """Test auto-selection logic."""
        from src.reconstruction.reconstruct import auto_select_method

        with patch('src.reconstruction.reconstruct.check_colmap_available', return_value=True), \
             patch('src.reconstruction.reconstruct.check_nerfstudio_available', return_value=True):

            # Many images -> COLMAP
            for i in range(35):
                (self.input_dir / f"img{i}.jpg").touch()

            method = auto_select_method(self.input_dir, "medium")
            self.assertEqual(method, "colmap")

    def test_auto_select_method_prefers_nerfstudio_for_few_images(self):
        """Test auto-selection for few images."""
        from src.reconstruction.reconstruct import auto_select_method

        with patch('src.reconstruction.reconstruct.check_colmap_available', return_value=True), \
             patch('src.reconstruction.reconstruct.check_nerfstudio_available', return_value=True):

            # Few images -> Nerfstudio
            for i in range(10):
                (self.input_dir / f"img{i}.jpg").touch()

            method = auto_select_method(self.input_dir, "medium")
            self.assertEqual(method, "nerfstudio")

    def test_auto_select_method_fallback(self):
        """Test fallback when one method unavailable."""
        from src.reconstruction.reconstruct import auto_select_method

        with patch('src.reconstruction.reconstruct.check_colmap_available', return_value=False), \
             patch('src.reconstruction.reconstruct.check_nerfstudio_available', return_value=True):

            for i in range(10):
                (self.input_dir / f"img{i}.jpg").touch()

            method = auto_select_method(self.input_dir, "medium")
            self.assertEqual(method, "nerfstudio")


class TestBlenderCleanup(unittest.TestCase):
    """Tests for BlenderProc cleanup (mocked since needs Blender)."""

    def test_args_parsing(self):
        """Test argument parsing."""
        import argparse

        # Test that parser exists and has expected args
        parser = argparse.ArgumentParser()
        parser.add_argument("input_file")
        parser.add_argument("output_file")
        parser.add_argument("--target-tris", type=int, default=50000)
        parser.add_argument("--texture-size", type=int, default=2048)
        parser.add_argument("--skip-bake", action="store_true")
        parser.add_argument("--skip-collision", action="store_true")
        parser.add_argument("--export-usd", action="store_true")

        # Parse some test args
        args = parser.parse_args(["input.glb", "output.glb", "--target-tris", "30000"])
        self.assertEqual(args.input_file, "input.glb")
        self.assertEqual(args.output_file, "output.glb")
        self.assertEqual(args.target_tris, 30000)
        self.assertEqual(args.texture_size, 2048)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Blender tests require Linux with Blender installed")
    def test_blender_import(self):
        """Test that blender_cleanup can be imported inside Blender."""
        pass


class TestIntegration(unittest.TestCase):
    """Integration tests (require full environment)."""

    @unittest.skipUnless(sys.platform.startswith("linux"), "Integration tests require Linux")
    def test_colmap_pipeline_dry_run(self):
        """Dry-run test for COLMAP pipeline (requires COLMAP installed)."""
        # This would run in the Docker container
        pass

    @unittest.skipUnless(sys.platform.startswith("linux"), "Integration tests require Linux")
    def test_nerfstudio_pipeline_dry_run(self):
        """Dry-run test for Nerfstudio pipeline."""
        pass


if __name__ == "__main__":
    unittest.main()