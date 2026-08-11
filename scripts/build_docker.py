#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path


IMAGE_TAG = "malo:latest"


def build_docker_image():
    """Build the Docker image for MaLO."""
    print(f"Building Docker image for MaLO... ({IMAGE_TAG})")

    # Path to the Dockerfile
    repo_root = Path(__file__).resolve().parent.parent
    dockerfile_path = repo_root / "docker" / "malo" / "Dockerfile"

    if not dockerfile_path.exists():
        print(f"ERROR: Dockerfile not found at {dockerfile_path}")
        return False

    # Build the Docker image
    # Use --platform linux/amd64 for ARM systems (macOS ARM) to run x64 containers
    try:
        subprocess.run(
            [
                "docker",
                "build",
                "--platform",
                "linux/amd64",
                "-t",
                IMAGE_TAG,
                "-f",
                str(dockerfile_path),
                str(dockerfile_path.parent),
            ],
            check=True,
        )
        print("Docker image built successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to build Docker image: {e}")
        return False


def create_docker_network():
    """Create Docker network for MaLO if it doesn't exist."""
    print("Checking Docker network...")

    # Check if network exists
    try:
        result = subprocess.run(
            [
                "docker",
                "network",
                "ls",
                "--filter",
                "name=ctfnet",
                "--format",
                "{{.Name}}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if "ctfnet" in result.stdout:
            print("Docker network 'ctfnet' already exists.")
            return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to check Docker network: {e}")
        return False

    # Create network
    try:
        subprocess.run(["docker", "network", "create", "ctfnet"], check=True)
        print("Docker network 'ctfnet' created successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to create Docker network: {e}")
        return False


if __name__ == "__main__":
    if build_docker_image() and create_docker_network():
        print("\nSetup complete! You can now run MaLO with:")
        print(
            "poetry run malo run --config "
            "config/single_react/architecture/single_react.yaml "
            "-c path/to/challenge"
        )
    else:
        print("\nSetup failed. Please check the errors above and try again.")
