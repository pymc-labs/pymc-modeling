"""Report the versions actually exercised, without importing optional backends."""

import platform
from importlib.metadata import PackageNotFoundError, version


def pytest_report_header():
    versions = [f"Python {platform.python_version()}"]
    for package in (
        "pytensor",
        "numpy",
        "scipy",
        "numba",
        "jax",
        "jaxlib",
        "torch",
        "mlx",
        "mlx-cpu",
    ):
        try:
            versions.append(f"{package} {version(package)}")
        except PackageNotFoundError:
            versions.append(f"{package} not installed")
    return "Example runtime: " + ", ".join(versions)
