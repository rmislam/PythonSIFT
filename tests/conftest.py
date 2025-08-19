import os
import tempfile
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a simple test image."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (80, 80), (255, 255, 255), -1)
    cv2.circle(image, (50, 50), 20, (0, 0, 0), -1)
    return image


@pytest.fixture
def grayscale_image() -> np.ndarray:
    """Create a simple grayscale test image."""
    image = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (80, 80), 255, -1)
    cv2.circle(image, (50, 50), 20, 0, -1)
    return image


@pytest.fixture
def sample_keypoints():
    """Create sample keypoints for testing."""
    keypoints = [
        cv2.KeyPoint(x=10.0, y=10.0, size=5.0, angle=0.0),
        cv2.KeyPoint(x=50.0, y=50.0, size=10.0, angle=45.0),
        cv2.KeyPoint(x=90.0, y=90.0, size=15.0, angle=90.0),
    ]
    return keypoints


@pytest.fixture
def sample_descriptors():
    """Create sample descriptors for testing."""
    np.random.seed(42)
    descriptors = np.random.randint(0, 256, (3, 128), dtype=np.uint8)
    return descriptors


@pytest.fixture
def mock_config():
    """Provide a mock configuration for testing."""
    return {
        'sigma': 1.6,
        'num_intervals': 3,
        'assumed_blur': 0.5,
        'image_border_width': 5,
        'contrast_threshold': 0.04,
        'edge_threshold': 10,
        'gaussian_kernel_size': 16,
    }


@pytest.fixture(autouse=True)
def reset_cv2_modules():
    """Reset cv2 modules state before each test."""
    yield


@pytest.fixture
def capture_stdout(monkeypatch):
    """Capture stdout for testing print statements."""
    import io
    import sys
    
    captured_output = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', captured_output)
    return captured_output


@pytest.fixture
def test_image_path(temp_dir: Path, sample_image: np.ndarray) -> Path:
    """Save a test image to a temporary file and return its path."""
    image_path = temp_dir / "test_image.png"
    cv2.imwrite(str(image_path), sample_image)
    return image_path


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return the path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def mock_sift_detector(mocker):
    """Mock SIFT detector for testing."""
    mock_detector = mocker.Mock()
    mock_detector.detectAndCompute.return_value = ([], None)
    return mock_detector


def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "requires_display: mark test as requiring a display"
    )