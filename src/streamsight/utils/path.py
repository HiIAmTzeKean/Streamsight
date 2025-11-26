"""Path utilities for streamsight library.

This module provides functions to resolve paths relative to the repository root,
ensuring that data and logs are stored consistently regardless of where the
library is run from.
"""

import os
from pathlib import Path
from typing import Optional


_REPO_ROOT_CACHE: Optional[Path] = None


def get_repo_root(
    marker_files: tuple[str, ...] = (
        ".git",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "README.md",
        "requirements.txt",
    ),
) -> Path:
    """Get the repository root directory.

    Searches upward from the current file location to find the repository root
    by looking for common marker files. Results are cached for performance.

    :param marker_files: Tuple of filenames that indicate repo root
    :type marker_files: tuple[str, ...]
    :return: Path to repository root
    :rtype: Path
    :raises RuntimeError: If repo root cannot be found
    """
    global _REPO_ROOT_CACHE

    if _REPO_ROOT_CACHE is not None:
        return _REPO_ROOT_CACHE

    # Start from this file's location
    current = Path(__file__).resolve().parent
    max_depth = 10

    for _ in range(max_depth):
        # Check for marker files
        if any((current / marker).exists() for marker in marker_files):
            _REPO_ROOT_CACHE = current
            return current

        # Move up one directory
        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent

    # Fallback: try environment variable
    if "STREAMSIGHT_ROOT" in os.environ:
        root = Path(os.environ["STREAMSIGHT_ROOT"])
        if root.exists():
            _REPO_ROOT_CACHE = root
            return root

    raise RuntimeError(
        "Could not find repository root. Please set STREAMSIGHT_ROOT "
        "environment variable or ensure you're running from within the repo."
    )


def get_data_dir(subdir: str = "") -> Path:
    """Get the data directory at repo root.

    :param subdir: Optional subdirectory within data/
    :type subdir: str
    :return: Path to data directory
    :rtype: Path
    """
    data_dir = get_repo_root() / "data"
    if subdir:
        data_dir = data_dir / subdir
    return data_dir


def get_logs_dir(subdir: str = "") -> Path:
    """Get the logs directory at repo root.

    :param subdir: Optional subdirectory within logs/
    :type subdir: str
    :return: Path to logs directory
    :rtype: Path
    """
    logs_dir = get_repo_root() / "logs"
    if subdir:
        logs_dir = logs_dir / subdir
    return logs_dir


def get_cache_dir(subdir: str = "") -> Path:
    """Get the cache directory at repo root.

    :param subdir: Optional subdirectory within cache/
    :type subdir: str
    :return: Path to cache directory
    :rtype: Path
    """
    cache_dir = get_repo_root() / "cache"
    if subdir:
        cache_dir = cache_dir / subdir
    return cache_dir


def ensure_dir(path: Path | str) -> Path:
    """Ensure directory exists, create if needed.

    :param path: Directory path
    :type path: Path | str
    :return: Path object
    :rtype: Path
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path: str | Path, relative_to_root: bool = True) -> Path:
    """Resolve a path, optionally relative to repo root.

    :param path: Path to resolve
    :type path: str | Path
    :param relative_to_root: If True and path is relative, resolve to repo root
    :type relative_to_root: bool
    :return: Resolved absolute path
    :rtype: Path
    """
    path = Path(path)

    if path.is_absolute():
        return path

    if relative_to_root:
        return get_repo_root() / path

    return path.resolve()
