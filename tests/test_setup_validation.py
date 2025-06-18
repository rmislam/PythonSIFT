import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


class TestSetupValidation:
    """Validation tests to ensure the testing infrastructure is properly set up."""

    def test_python_version(self):
        """Test that Python version is 3.8 or higher."""
        assert sys.version_info >= (3, 8), "Python 3.8 or higher is required"

    def test_numpy_import(self):
        """Test that numpy can be imported and is functional."""
        arr = np.array([1, 2, 3])
        assert arr.shape == (3,)
        assert np.sum(arr) == 6

    def test_opencv_import(self):
        """Test that OpenCV can be imported and is functional."""
        img = np.zeros((10, 10), dtype=np.uint8)
        result = cv2.resize(img, (5, 5))
        assert result.shape == (5, 5)

    def test_pysift_import(self):
        """Test that pysift module can be imported."""
        import pysift
        assert hasattr(pysift, 'computeKeypointsAndDescriptors')

    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit test marker works correctly."""
        assert True

    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration test marker works correctly."""
        assert True

    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow test marker works correctly."""
        import time
        time.sleep(0.1)
        assert True

    def test_temp_dir_fixture(self, temp_dir):
        """Test that temp_dir fixture works correctly."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
        assert test_file.read_text() == "test content"

    def test_sample_image_fixture(self, sample_image):
        """Test that sample_image fixture provides correct image."""
        assert isinstance(sample_image, np.ndarray)
        assert sample_image.shape == (100, 100, 3)
        assert sample_image.dtype == np.uint8

    def test_grayscale_image_fixture(self, grayscale_image):
        """Test that grayscale_image fixture provides correct image."""
        assert isinstance(grayscale_image, np.ndarray)
        assert grayscale_image.shape == (100, 100)
        assert grayscale_image.dtype == np.uint8

    def test_sample_keypoints_fixture(self, sample_keypoints):
        """Test that sample_keypoints fixture provides valid keypoints."""
        assert len(sample_keypoints) == 3
        for kp in sample_keypoints:
            assert isinstance(kp, cv2.KeyPoint)

    def test_sample_descriptors_fixture(self, sample_descriptors):
        """Test that sample_descriptors fixture provides valid descriptors."""
        assert isinstance(sample_descriptors, np.ndarray)
        assert sample_descriptors.shape == (3, 128)
        assert sample_descriptors.dtype == np.uint8

    def test_mock_config_fixture(self, mock_config):
        """Test that mock_config fixture provides expected configuration."""
        expected_keys = {
            'sigma', 'num_intervals', 'assumed_blur', 'image_border_width',
            'contrast_threshold', 'edge_threshold', 'gaussian_kernel_size'
        }
        assert set(mock_config.keys()) == expected_keys

    def test_test_image_path_fixture(self, test_image_path):
        """Test that test_image_path fixture creates a valid image file."""
        assert test_image_path.exists()
        assert test_image_path.suffix == ".png"
        img = cv2.imread(str(test_image_path))
        assert img is not None
        assert img.shape == (100, 100, 3)

    def test_coverage_is_enabled(self):
        """Test that coverage is properly configured."""
        try:
            import coverage
            assert True, "Coverage module is available"
        except ImportError:
            pytest.skip("Coverage module not installed yet")

    def test_project_structure(self):
        """Test that the expected project structure exists."""
        project_root = Path(__file__).parent.parent
        assert (project_root / "pyproject.toml").exists()
        assert (project_root / "tests").exists()
        assert (project_root / "tests" / "__init__.py").exists()
        assert (project_root / "tests" / "conftest.py").exists()
        assert (project_root / "tests" / "unit").exists()
        assert (project_root / "tests" / "integration").exists()
        assert (project_root / "pysift.py").exists()


@pytest.mark.performance
class TestPerformanceMarker:
    """Test class to validate performance marker."""
    
    def test_performance_marker_works(self):
        """Test that performance marker can be applied."""
        assert True