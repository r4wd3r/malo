"""Docker environment manager for MaLO LangGraph."""

from __future__ import annotations

import logging
import os
import tarfile
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import docker

from MaLO.config.models import EnvironmentConfig

logger = logging.getLogger(__name__)

# Default working directory inside the container for challenge files.
CTF_WORKDIR = "/ctf"


@dataclass(frozen=True)
class ExecutionResult:
    """Result of a command executed inside a container."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


class EnvironmentManager:
    """Manages the lifecycle of a Docker-based CTF environment."""

    def __init__(
        self,
        config: EnvironmentConfig,
        docker_client: docker.DockerClient | None = None,
    ) -> None:
        """Initializes the environment manager.

        Args:
            config: Environment configuration.
            docker_client: Optional Docker client instance.
        """
        self._config = config
        self._client = docker_client or docker.from_env()
        self._container: docker.models.containers.Container | None = None
        self._workdir = CTF_WORKDIR

    @property
    def workdir(self) -> str:
        """Returns the working directory inside the container."""
        return self._workdir

    def setup(self) -> None:
        """Starts the container and runs setup commands."""
        # Pull or use local image
        try:
            self._client.images.get(self._config.docker_image)
            logger.info("Using local image: %s", self._config.docker_image)
        except docker.errors.ImageNotFound:
            logger.info("Pulling image: %s", self._config.docker_image)
            self._client.images.pull(self._config.docker_image)

        run_kwargs: dict = {
            "detach": True,
            "tty": True,
            "stdin_open": True,
            "working_dir": self._workdir,
            "user": "root",
            "privileged": True,
        }
        if self._config.docker_network:
            run_kwargs["network"] = self._config.docker_network

        # Start container with elevated privileges
        self._container = self._client.containers.run(
            self._config.docker_image,
            **run_kwargs,
        )
        logger.info("Container started: %s", self._container.short_id)

        # Create working directory
        self.exec_run(f"mkdir -p {self._workdir}")

        # Run setup commands
        for command in self._config.setup_commands:
            self.exec_run(command)

    def copy_files(
        self,
        challenge_path: str,
        files: list[str],
    ) -> None:
        """Copies challenge files into the container.

        Args:
            challenge_path: Path to the challenge folder on host.
            files: List of relative file paths to copy.
        """
        if self._container is None:
            raise RuntimeError("Container is not running.")

        challenge_path = Path(challenge_path)

        # First, collect all unique parent directories and create them
        parent_dirs: set[str] = set()
        for file_rel in files:
            container_dest = f"{self._workdir}/{file_rel}"
            parent_dir = str(Path(container_dest).parent)
            parent_dirs.add(parent_dir)

        # Create all directories in one command
        if parent_dirs:
            dirs_str = " ".join(f'"{d}"' for d in sorted(parent_dirs))
            result = self.exec_run(f"mkdir -p {dirs_str}")
            if result.exit_code != 0:
                logger.warning("mkdir failed: %s", result.stderr)

        # Now copy all files
        copied = 0
        for file_rel in files:
            host_path = challenge_path / file_rel
            if not host_path.exists():
                logger.warning("File not found: %s", host_path)
                continue

            container_dest = f"{self._workdir}/{file_rel}"
            self._copy_file_to_container(host_path, container_dest)
            copied += 1

        logger.info("Copied %d files to container:%s", copied, self._workdir)

    def _copy_file_to_container(
        self,
        host_path: Path,
        container_path: str,
    ) -> None:
        """Copies a single file to the container using tar.

        Args:
            host_path: Path to file on host.
            container_path: Destination path in container.
        """
        if self._container is None:
            raise RuntimeError("Container is not running.")

        container_dir = str(Path(container_path).parent)
        file_name = Path(container_path).name

        # Create a tar archive in memory
        tar_stream = BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tar.add(str(host_path), arcname=file_name)
        tar_stream.seek(0)

        # Put the tar archive into the container
        self._container.put_archive(container_dir, tar_stream)

    def copy_directory(
        self,
        src_path: str,
        dest_name: str = "src",
    ) -> None:
        """Copies an entire directory into the container.

        Args:
            src_path: Path to directory on host.
            dest_name: Name of directory in container workdir.
        """
        if self._container is None:
            raise RuntimeError("Container is not running.")

        src_path = Path(src_path)
        if not src_path.is_dir():
            raise ValueError(f"Not a directory: {src_path}")

        container_dest = f"{self._workdir}/{dest_name}"
        self.exec_run(f"mkdir -p {container_dest}")

        # Create tar archive of directory
        tar_stream = BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            for root, _, files in os.walk(src_path):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(src_path)
                    tar.add(str(file_path), arcname=str(arcname))
        tar_stream.seek(0)

        # Put the tar archive into the container
        self._container.put_archive(container_dest, tar_stream)
        logger.info("Copied directory to container:%s", container_dest)

    def exec_run(
        self,
        command: str,
        timeout_seconds: int | None = None,
        workdir: str | None = None,
    ) -> ExecutionResult:
        """Executes a command inside the container.

        Args:
            command: Command to run.
            timeout_seconds: Optional override for command timeout.
            workdir: Optional working directory override.

        Returns:
            Execution result with stdout, stderr, and exit code.

        Raises:
            RuntimeError: If the container is not running.
        """
        if self._container is None:
            raise RuntimeError("Container is not running.")

        timeout = timeout_seconds or self._config.timeout_seconds
        work_dir = workdir or self._workdir

        def _run_command() -> tuple[int, tuple[bytes | None, bytes | None]]:
            return self._container.exec_run(
                command,
                demux=True,
                workdir=work_dir,
            )

        executor = ThreadPoolExecutor(max_workers=1)
        timed_out = False
        try:
            future = executor.submit(_run_command)
            exit_code, output = future.result(timeout=timeout)
        except FuturesTimeoutError:
            timed_out = True
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            return ExecutionResult(
                exit_code=124,
                stdout="",
                stderr="Command timed out.",
                timed_out=True,
            )
        finally:
            if not timed_out:
                executor.shutdown(wait=True)

        stdout_bytes, stderr_bytes = output
        stdout = (stdout_bytes or b"").decode("utf-8", errors="replace")
        stderr = (stderr_bytes or b"").decode("utf-8", errors="replace")
        return ExecutionResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
        )

    def teardown(self) -> None:
        """Stops and removes the container."""
        if self._container is None:
            return
        try:
            self._container.kill()
        except docker.errors.APIError:
            pass  # Container may already be stopped
        finally:
            try:
                self._container.remove(force=True)
            except docker.errors.APIError:
                pass
            self._container = None
            logger.info("Container stopped and removed.")
