"""Constants for Docker environment and challenge containers."""

from __future__ import annotations

# Docker network used to connect agent and challenge containers.
DEFAULT_DOCKER_NETWORK = "ctfnet"

# Prefix for challenge container names (suffix is sanitized challenge id).
CHALLENGE_CONTAINER_PREFIX = "malo_challenge_"

# Default port for challenge services (e.g. pwn binaries via socat).
DEFAULT_CHALLENGE_PORT = 1337

# Filenames used to detect challenge container setup.
DOCKERFILE_NAME = "Dockerfile"
DOCKER_COMPOSE_FILE = "docker-compose.yml"

# Image tag prefix when building from challenge Dockerfile.
CHALLENGE_IMAGE_PREFIX = "malo_challenge_"
