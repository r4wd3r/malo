"""Challenge loaders for different benchmark formats."""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml

from MaLO.challenges.models import (
    BenchmarkType,
    ChallengeData,
    Goal,
    GoalType,
)

logger = logging.getLogger(__name__)


class ChallengeLoader(ABC):
    """Abstract base class for challenge loaders."""

    @abstractmethod
    def load(self, path: str) -> ChallengeData:
        """Loads challenge data from a folder.

        Args:
            path: Path to the challenge folder.

        Returns:
            Loaded challenge data.
        """
        pass


class MaloChallengeLoader(ChallengeLoader):
    """Loader for MaLO benchmark challenges."""

    def load(self, path: str) -> ChallengeData:
        """Loads a MaLO benchmark challenge.

        Args:
            path: Path to the challenge folder.

        Returns:
            Loaded challenge data.
        """
        metadata_file = os.path.join(path, "challenge.yml")
        with open(metadata_file, "r", encoding="utf-8") as handle:
            metadata = yaml.safe_load(handle)

        # Use files listed in challenge.yml when present; else all under src/
        listed = metadata.get("files")
        if listed:
            files = list(listed)
        else:
            src_path = os.path.join(path, "src")
            if os.path.isdir(src_path):
                files = [
                    os.path.join("src", f) for f in os.listdir(src_path)
                ]
            else:
                files = []

        # Check for Docker configuration
        dockerfile = (
            os.path.join(path, "Dockerfile")
            if os.path.isfile(os.path.join(path, "Dockerfile"))
            else None
        )
        compose = (
            os.path.join(path, "docker-compose.yml")
            if os.path.isfile(os.path.join(path, "docker-compose.yml"))
            else None
        )

        # Parse goal
        goal = _parse_goal_from_metadata(metadata)

        return ChallengeData(
            id=metadata.get("ID", ""),
            path=path,
            name=metadata.get("name", ""),
            date=str(metadata.get("date", "")),
            origin=metadata.get("origin", ""),
            category=metadata.get("category", ""),
            description=metadata.get("description", ""),
            goal=goal,
            tags=metadata.get("tags", []),
            hints=metadata.get("hints", []),
            files=files,
            tools=metadata.get("tools", []),
            container=dockerfile,
            compose=compose,
            benchmark=BenchmarkType.MALO_BENCHMARK,
        )


class NYUCTFChallengeLoader(ChallengeLoader):
    """Loader for NYU CTF benchmark challenges."""

    def load(self, path: str) -> ChallengeData:
        """Loads a NYU CTF challenge.

        Args:
            path: Path to the challenge folder.

        Returns:
            Loaded challenge data.
        """
        metadata_file = os.path.join(path, "challenge.json")
        with open(metadata_file, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        # Extract path components
        path_parts = path.split(os.sep)
        challenge_year = path_parts[-4] if len(path_parts) >= 4 else ""
        challenge_origin = path_parts[-3] if len(path_parts) >= 3 else ""

        # Build challenge ID
        category = metadata.get("category", "")
        name = metadata.get("name", "").replace(" ", "-")
        challenge_id = f"NYU-{challenge_year}-{challenge_origin}-{category}-{name}"
        challenge_id = re.sub(r"[^a-zA-Z0-9-]", "", challenge_id)

        # Check for Docker configuration
        dockerfile = (
            os.path.join(path, "Dockerfile")
            if os.path.isfile(os.path.join(path, "Dockerfile"))
            else None
        )
        compose = (
            os.path.join(path, "docker-compose.yml")
            if os.path.isfile(os.path.join(path, "docker-compose.yml"))
            else None
        )

        # Parse goal
        goal = _parse_goal_from_metadata(metadata, is_nyu=True)

        # Calculate base path
        base_path = str(Path(*path_parts[:-4])) if len(path_parts) >= 4 else ""

        return ChallengeData(
            id=challenge_id,
            path=path,
            name=name,
            date=challenge_year,
            origin=challenge_origin,
            category=category,
            description=metadata.get("description", ""),
            goal=goal,
            files=metadata.get("files", []),
            container=dockerfile,
            compose=compose,
            benchmark=BenchmarkType.NYU_CTF,
            base_path=base_path,
        )


def _parse_goal_from_metadata(
    metadata: dict[str, Any],
    is_nyu: bool = False,
) -> Goal:
    """Parses goal from challenge metadata.

    If no explicit goal exists, migrates from legacy flags format.

    Args:
        metadata: Challenge metadata dictionary.
        is_nyu: Whether this is a NYU CTF challenge.

    Returns:
        Parsed Goal object.

    Raises:
        ValueError: If no goal or flags are found.
    """
    # Check for explicit goal definition
    goal_data = metadata.get("goal")
    if goal_data:
        return Goal(
            goal_type=GoalType(goal_data["goal_type"]),
            description=goal_data["description"],
            validation_data=goal_data["validation_data"],
            metadata=goal_data.get("metadata"),
        )

    # Migrate from legacy flags format
    if is_nyu:
        flag = metadata.get("flag", "")
        if flag:
            return Goal(
                goal_type=GoalType.EXACT_MATCH,
                description="Submit the correct flag",
                validation_data={"expected": flag},
                metadata={"auto_migrated": True, "source": "nyu_flag"},
            )
    else:
        flags = metadata.get("flags", [])
        if flags:
            expected = flags[0] if isinstance(flags, list) else str(flags)
            return Goal(
                goal_type=GoalType.EXACT_MATCH,
                description="Submit the correct flag",
                validation_data={"expected": expected},
                metadata={"auto_migrated": True, "original_flags": flags},
            )

    raise ValueError("Challenge must have either 'goal' or 'flags' defined")


def detect_benchmark_type(path: str) -> BenchmarkType | None:
    """Detects the benchmark type for a challenge folder.

    Args:
        path: Path to the challenge folder.

    Returns:
        Detected benchmark type, or None if unknown.
    """
    if _is_malo_challenge(path):
        return BenchmarkType.MALO_BENCHMARK
    if _is_nyu_challenge(path):
        return BenchmarkType.NYU_CTF
    return None


def _is_malo_challenge(path: str) -> bool:
    """Checks if a folder contains a MaLO benchmark challenge."""
    yml_path = os.path.join(path, "challenge.yml")
    src_path = os.path.join(path, "src")

    if not os.path.isfile(yml_path):
        return False
    if not os.path.isdir(src_path):
        return False

    with open(yml_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    required = ["ID", "name", "date", "origin", "category", "description"]
    for key in required:
        if key not in data:
            return False

    has_goal_or_flags = "goal" in data or "flags" in data
    return has_goal_or_flags


def _is_nyu_challenge(path: str) -> bool:
    """Checks if a folder contains a NYU CTF challenge."""
    json_path = os.path.join(path, "challenge.json")
    if not os.path.isfile(json_path):
        return False

    # Check path structure: <year>/<event>/<category>/<challenge>
    path_parts = path.split(os.sep)
    if len(path_parts) < 4:
        return False

    year = path_parts[-4]
    if not (year.isdigit() and len(year) == 4):
        return False

    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    required = ["name", "description", "category", "flag", "files"]
    for key in required:
        if key not in data:
            return False

    return True


def load_challenge(path: str) -> ChallengeData:
    """Loads a single challenge, auto-detecting the format.

    Args:
        path: Path to the challenge folder.

    Returns:
        Loaded challenge data.

    Raises:
        ValueError: If the challenge format is unknown.
    """
    benchmark_type = detect_benchmark_type(path)

    if benchmark_type == BenchmarkType.MALO_BENCHMARK:
        return MaloChallengeLoader().load(path)
    if benchmark_type == BenchmarkType.NYU_CTF:
        return NYUCTFChallengeLoader().load(path)

    raise ValueError(f"Unknown challenge format in {path}")


def load_challenges(path: str, batch: bool = False) -> list[ChallengeData]:
    """Loads one or more challenges from a path.

    Args:
        path: Path to challenge folder or parent directory.
        batch: If True, recursively load all challenges in the directory.

    Returns:
        List of loaded challenge data.
    """
    if not batch:
        return [load_challenge(path)]

    challenges: list[ChallengeData] = []
    for root, dirs, files in os.walk(path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        # Check for challenge markers
        if "challenge.yml" in files or "challenge.json" in files:
            try:
                challenge = load_challenge(root)
                challenges.append(challenge)
            except (ValueError, FileNotFoundError, KeyError) as err:
                # Log error but continue with other challenges
                logger.warning("Failed to load challenge at %s: %s", root, err)

    return challenges
