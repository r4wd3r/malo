"""Challenge container lifecycle for malo-benchmark (and extensible for NYU later)."""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Any

import docker

from MaLO.challenges.models import ChallengeData
from MaLO.environment.constants import (
    CHALLENGE_CONTAINER_PREFIX,
    CHALLENGE_IMAGE_PREFIX,
    DEFAULT_CHALLENGE_PORT,
    DOCKER_COMPOSE_FILE,
    DOCKERFILE_NAME,
)

logger = logging.getLogger(__name__)


def _run_docker_cmd(
    args: list[str],
    cwd: str | Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a Docker CLI command; on failure raise RuntimeError with stderr.

    Args:
        args: Command and arguments (e.g. ["docker", "build", "-t", "img", "."]).
        cwd: Optional working directory.

    Returns:
        CompletedProcess with stdout/stderr as strings.

    Raises:
        RuntimeError: If the command exits non-zero, message includes stderr.
    """
    try:
        return subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd is not None else None,
        )
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or str(e)).strip()
        raise RuntimeError(f"Docker failed: {err}") from e


def _sanitize_challenge_id(challenge_id: str) -> str:
    """Sanitize challenge id for use in Docker names and image tags.

    Args:
        challenge_id: Raw challenge id (e.g. MALO-0001-24).

    Returns:
        Lowercase alphanumeric and underscores only.
    """
    return re.sub(r"[^a-z0-9]+", "_", challenge_id.lower()).strip("_") or "challenge"


def ensure_network(network_name: str) -> None:
    """Create the Docker network if it does not exist.

    Args:
        network_name: Name of the network (e.g. ctfnet).

    Raises:
        docker.errors.APIError: If network creation fails.
    """
    client = docker.from_env()
    try:
        client.networks.get(network_name)
        logger.info("Docker network %s already exists", network_name)
    except docker.errors.NotFound:
        client.networks.create(network_name)
        logger.info("Created Docker network %s", network_name)


def needs_challenge_container(challenge: ChallengeData) -> bool:
    """Return True if the challenge defines a Docker container (compose or Dockerfile).

    Args:
        challenge: Loaded challenge data.

    Returns:
        True if challenge has compose or container path set.
    """
    return bool(challenge.compose or challenge.container)


def start_challenge_container(
    challenge: ChallengeData,
    network: str,
) -> tuple[str, int]:
    """Start the challenge container (compose or Dockerfile) and return endpoint.

    For malo-benchmark: runs docker-compose or builds and runs from Dockerfile.
    Design allows branching on challenge.benchmark for NYU later.

    Args:
        challenge: Loaded challenge data (must have compose or container set).
        network: Docker network name to attach the challenge container to.

    Returns:
        (hostname, port) that the agent should use to reach the challenge service.

    Raises:
        ValueError: If challenge has no container/compose.
        RuntimeError: If start fails.
    """
    if not needs_challenge_container(challenge):
        raise ValueError(
            "Challenge has no compose or Dockerfile; cannot start challenge container."
        )

    challenge_dir = Path(challenge.path).resolve()
    compose_path = challenge_dir / DOCKER_COMPOSE_FILE
    dockerfile_path = challenge_dir / DOCKERFILE_NAME
    sid = _sanitize_challenge_id(challenge.challenge_id)
    container_name = f"{CHALLENGE_CONTAINER_PREFIX}{sid}"

    if compose_path.exists():
        return _start_compose(
            challenge_dir, compose_path, container_name, network
        )
    if dockerfile_path.exists():
        return _start_dockerfile(
            challenge_dir, sid, container_name, network
        )
    raise RuntimeError(
        f"Challenge has container/compose set but neither "
        f"{DOCKER_COMPOSE_FILE} nor {DOCKERFILE_NAME} found under {challenge.path}"
    )


# Well-known container names used by malo-benchmark compose files; remove
# before starting so a leftover from a previous challenge does not block.
_COMPOSE_LEGACY_CONTAINER_NAMES = ("ctf-challenge",)


def _start_compose(
    challenge_dir: Path,
    compose_path: Path,
    container_name: str,
    network: str,
) -> tuple[str, int]:
    """Start compose and connect the service container to the given network."""
    service_name, service_port = _select_compose_service(compose_path)
    if service_port is None:
        logger.warning(
            "No port found in compose; defaulting to %s.",
            DEFAULT_CHALLENGE_PORT,
        )
        service_port = DEFAULT_CHALLENGE_PORT
    # Tear down this project and any well-known legacy names so start is
    # idempotent when multiple challenges share the same container_name.
    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(compose_path),
            "down",
            "--remove-orphans",
        ],
        capture_output=True,
        check=False,
        cwd=challenge_dir,
    )
    for name in _COMPOSE_LEGACY_CONTAINER_NAMES:
        subprocess.run(
            ["docker", "rm", "-f", name],
            capture_output=True,
            check=False,
        )
    _run_docker_cmd(
        [
            "docker",
            "compose",
            "-f",
            str(compose_path),
            "up",
            "-d",
            "--force-recreate",
        ],
        cwd=challenge_dir,
    )
    logger.info("Started docker-compose for %s", challenge_dir.name)

    # Get the first container id from the compose project (directory name).
    ps_cmd = ["docker", "compose", "-f", str(compose_path), "ps", "-q"]
    if service_name:
        ps_cmd.append(service_name)
    result = _run_docker_cmd(ps_cmd, cwd=challenge_dir)
    ids = result.stdout.strip().splitlines()
    if not ids:
        raise RuntimeError(
            "docker compose up did not start any container; "
            "cannot connect to shared network."
        )
    service_container_id = ids[0].strip()
    client = docker.from_env()
    network_obj = client.networks.get(network)
    container = client.containers.get(service_container_id)
    
    # Check if container is already connected to the network
    container_networks = container.attrs.get("NetworkSettings", {}).get("Networks", {})
    if network not in container_networks:
        network_obj.connect(service_container_id)
        logger.info("Connected compose container %s to network %s", service_container_id, network)
    else:
        logger.info("Compose container %s already connected to network %s", service_container_id, network)

    # Compose container name is typically <project>_<service>_<replica>.
    hostname = container.name
    return hostname, service_port


def _start_dockerfile(
    challenge_dir: Path,
    sanitized_id: str,
    container_name: str,
    network: str,
) -> tuple[str, int]:
    """Build image from Dockerfile and run container on the given network."""
    image_tag = f"{CHALLENGE_IMAGE_PREFIX}{sanitized_id}"
    build_ctx = challenge_dir.resolve()
    _run_docker_cmd(
        ["docker", "build", "-t", image_tag, str(build_ctx)],
    )
    logger.info("Built image %s from %s", image_tag, challenge_dir)

    dockerfile_path = challenge_dir / DOCKERFILE_NAME
    service_port = _extract_dockerfile_port(dockerfile_path)
    if service_port is None:
        logger.warning(
            "No EXPOSE port found in Dockerfile; defaulting to %s.",
            DEFAULT_CHALLENGE_PORT,
        )
        service_port = DEFAULT_CHALLENGE_PORT

    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        check=False,
    )
    run_cmd = [
        "docker",
        "run",
        "-d",
        "--rm",
        "--name",
        container_name,
        "--network",
        network,
        "--platform",
        "linux/amd64",
    ]
    if service_port is not None:
        run_cmd.extend(["-p", f"{service_port}:{service_port}"])
    run_cmd.append(image_tag)
    _run_docker_cmd(run_cmd)
    logger.info("Started challenge container %s on network %s", container_name, network)
    return container_name, service_port


def stop_challenge_container(challenge: ChallengeData) -> None:
    """Stop the challenge container (compose or single container).

    Args:
        challenge: Same challenge data used for start_challenge_container.
    """
    challenge_dir = Path(challenge.path).resolve()
    compose_path = challenge_dir / DOCKER_COMPOSE_FILE
    sid = _sanitize_challenge_id(challenge.challenge_id)
    container_name = f"{CHALLENGE_CONTAINER_PREFIX}{sid}"

    if compose_path.exists():
        try:
            subprocess.run(
                [
                    "docker",
                    "compose",
                    "-f",
                    str(compose_path),
                    "down",
                    "--volumes",
                ],
                check=True,
                capture_output=True,
                cwd=str(challenge_dir),
            )
            logger.info("Stopped docker-compose for %s", challenge_dir.name)
        except subprocess.CalledProcessError as e:
            logger.warning("Failed to stop docker-compose: %s", e)
    else:
        try:
            subprocess.run(
                ["docker", "stop", container_name],
                check=True,
                capture_output=True,
            )
            logger.info("Stopped container %s", container_name)
        except subprocess.CalledProcessError as e:
            logger.warning("Failed to stop container %s: %s", container_name, e)


def _extract_dockerfile_port(dockerfile_path: Path) -> int | None:
    """Extract the first exposed TCP port from a Dockerfile, if any."""
    try:
        content = dockerfile_path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or not stripped.upper().startswith("EXPOSE"):
            continue
        for token in stripped.split()[1:]:
            port_text = token.split("/")[0]
            if port_text.isdigit():
                return int(port_text)
    return None


def _select_compose_service(
    compose_path: Path,
) -> tuple[str | None, int | None]:
    """Return service name and first exposed port from docker-compose, if any."""
    try:
        import yaml
    except Exception:
        return None, None

    try:
        data = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    except OSError:
        return None, None

    services = (data or {}).get("services", {})
    if not isinstance(services, dict):
        return None, None

    first_service = next(iter(services.keys()), None)
    for name, service in services.items():
        port = _extract_service_port(service or {})
        if port is not None:
            return name, port
    return first_service, None


def _extract_service_port(service: dict[str, Any]) -> int | None:
    """Extract the first container port from a compose service definition."""
    for key in ("ports", "expose"):
        raw_ports = service.get(key, [])
        if raw_ports is None:
            continue
        if not isinstance(raw_ports, list):
            raw_ports = [raw_ports]
        for entry in raw_ports:
            if isinstance(entry, int):
                return entry
            if not isinstance(entry, str):
                continue
            # Formats: "8000:80", "127.0.0.1:8000:80", "80", "80/tcp"
            parts = entry.split(":")
            container_part = parts[-1]
            port_text = container_part.split("/")[0]
            if port_text.isdigit():
                return int(port_text)
    return None
