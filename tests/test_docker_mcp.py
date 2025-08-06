"""Tests for Docker MCP Server."""

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from src.mcp_servers.docker.server import DockerMCPServer
from src.mcp_servers.docker.launcher import DockerMCPLauncher


@pytest.fixture
def temp_project_root():
    """Create a temporary project root for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_docker_client():
    """Mock Docker client for testing."""
    with patch("src.mcp_servers.docker.server.docker.from_env") as mock_docker:
        client = Mock()
        client.ping.return_value = True
        mock_docker.return_value = client
        yield client


class TestDockerMCPServer:
    """Test cases for DockerMCPServer."""

    def test_init_success(self, mock_docker_client, temp_project_root):
        """Test successful server initialization."""
        server = DockerMCPServer(project_root=temp_project_root)

        assert server.project_root == Path(temp_project_root).resolve()
        assert server.docker_client == mock_docker_client
        assert len(server.managed_containers) == 0
        mock_docker_client.ping.assert_called_once()

    def test_init_docker_connection_failure(self, temp_project_root):
        """Test server initialization with Docker connection failure."""
        with patch("src.mcp_servers.docker.server.docker.from_env") as mock_docker:
            mock_docker.side_effect = Exception("Docker not available")

            with pytest.raises(RuntimeError, match="Failed to connect to Docker"):
                DockerMCPServer(project_root=temp_project_root)

    def test_validate_path_within_project(self, mock_docker_client, temp_project_root):
        """Test path validation within project boundaries."""
        server = DockerMCPServer(project_root=temp_project_root)

        # Test relative path
        result = server._validate_path("subdir/file.txt")
        expected = Path(temp_project_root) / "subdir/file.txt"
        assert result == expected.resolve()

        # Test absolute path within project
        abs_path = Path(temp_project_root) / "test.txt"
        result = server._validate_path(str(abs_path))
        assert result == abs_path.resolve()

    def test_validate_path_outside_project(self, mock_docker_client, temp_project_root):
        """Test path validation outside project boundaries."""
        server = DockerMCPServer(project_root=temp_project_root)

        with pytest.raises(ValueError, match="outside project boundaries"):
            server._validate_path("/etc/passwd")

    @pytest.mark.asyncio
    async def test_build_image_success(self, mock_docker_client, temp_project_root):
        """Test successful image build."""
        server = DockerMCPServer(project_root=temp_project_root)

        # Create test Dockerfile
        dockerfile_path = Path(temp_project_root) / "Dockerfile"
        dockerfile_path.write_text("FROM python:3.13-slim")

        # Mock successful build
        mock_image = Mock()
        mock_image.id = "sha256:test123"
        build_logs = [{"stream": "Step 1/1 : FROM python:3.13-slim\n"}]

        mock_docker_client.images.build.return_value = (mock_image, build_logs)

        result = await server._build_image(path=temp_project_root, tag="test:latest")

        assert len(result) == 1
        result_data = json.loads(result[0].text)
        assert result_data["status"] == "success"
        assert result_data["image_id"] == "sha256:test123"
        assert result_data["tag"] == "test:latest"
        assert "Step 1/1" in result_data["build_logs"][0]

    @pytest.mark.asyncio
    async def test_build_image_build_context_not_found(
        self, mock_docker_client, temp_project_root
    ):
        """Test image build with missing build context."""
        server = DockerMCPServer(project_root=temp_project_root)

        result = await server._build_image(path="nonexistent", tag="test:latest")

        assert len(result) == 1
        assert "Build context not found" in result[0].text

    @pytest.mark.asyncio
    async def test_build_image_dockerfile_not_found(
        self, mock_docker_client, temp_project_root
    ):
        """Test image build with missing Dockerfile."""
        server = DockerMCPServer(project_root=temp_project_root)

        result = await server._build_image(path=temp_project_root, tag="test:latest")

        assert len(result) == 1
        assert "Dockerfile not found" in result[0].text

    @pytest.mark.asyncio
    async def test_run_container_success(self, mock_docker_client, temp_project_root):
        """Test successful container run."""
        server = DockerMCPServer(project_root=temp_project_root)

        # Create test volume mount directory
        volume_dir = Path(temp_project_root) / "data"
        volume_dir.mkdir()

        # Mock container
        mock_container = Mock()
        mock_container.id = "container123"
        mock_container.name = "test_container"
        mock_container.logs.return_value = b"Hello World"
        mock_container.wait.return_value = {"StatusCode": 0}

        mock_docker_client.containers.run.return_value = mock_container

        result = await server._run_container(
            image="python:3.13-slim",
            command="echo 'Hello World'",
            volumes={str(volume_dir): "/data"},
            detach=False,
        )

        assert len(result) == 1
        result_data = json.loads(result[0].text)
        assert result_data["status"] == "success"
        assert result_data["container_id"] == "container123"
        assert result_data["output"] == "Hello World"
        assert result_data["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_run_container_detached(self, mock_docker_client, temp_project_root):
        """Test detached container run."""
        server = DockerMCPServer(project_root=temp_project_root)

        mock_container = Mock()
        mock_container.id = "container123"
        mock_container.name = "test_container"

        mock_docker_client.containers.run.return_value = mock_container

        result = await server._run_container(image="python:3.13-slim", detach=True)

        assert len(result) == 1
        result_data = json.loads(result[0].text)
        assert result_data["status"] == "success"
        assert result_data["detached"] is True
        assert "container123" in server.managed_containers

    @pytest.mark.asyncio
    async def test_run_container_volume_not_found(
        self, mock_docker_client, temp_project_root
    ):
        """Test container run with non-existent volume mount."""
        server = DockerMCPServer(project_root=temp_project_root)

        result = await server._run_container(
            image="python:3.13-slim", volumes={"nonexistent": "/data"}
        )

        assert len(result) == 1
        assert "Host path not found" in result[0].text

    @pytest.mark.asyncio
    async def test_stop_container_success(self, mock_docker_client, temp_project_root):
        """Test successful container stop."""
        server = DockerMCPServer(project_root=temp_project_root)

        # Add container to managed containers
        mock_container = Mock()
        server.managed_containers["container123"] = mock_container

        mock_docker_client.containers.get.return_value = mock_container

        result = await server._stop_container("container123")

        assert len(result) == 1
        result_data = json.loads(result[0].text)
        assert result_data["status"] == "success"
        assert result_data["action"] == "stopped"
        assert "container123" not in server.managed_containers
        mock_container.stop.assert_called_once_with(timeout=10)

    @pytest.mark.asyncio
    async def test_remove_container_success(
        self, mock_docker_client, temp_project_root
    ):
        """Test successful container removal."""
        server = DockerMCPServer(project_root=temp_project_root)

        mock_container = Mock()
        server.managed_containers["container123"] = mock_container

        mock_docker_client.containers.get.return_value = mock_container

        result = await server._remove_container("container123", force=True)

        assert len(result) == 1
        result_data = json.loads(result[0].text)
        assert result_data["status"] == "success"
        assert result_data["action"] == "removed"
        assert "container123" not in server.managed_containers
        mock_container.remove.assert_called_once_with(force=True)

    @pytest.mark.asyncio
    async def test_get_container_logs_success(
        self, mock_docker_client, temp_project_root
    ):
        """Test successful log retrieval."""
        server = DockerMCPServer(project_root=temp_project_root)

        mock_container = Mock()
        mock_container.logs.return_value = b"Log line 1\nLog line 2\nLog line 3"

        mock_docker_client.containers.get.return_value = mock_container

        result = await server._get_container_logs("container123", tail=100)

        assert len(result) == 1
        result_data = json.loads(result[0].text)
        assert result_data["container_id"] == "container123"
        assert result_data["streaming"] is False
        assert len(result_data["logs"]) == 3

    @pytest.mark.asyncio
    async def test_list_containers_success(self, mock_docker_client, temp_project_root):
        """Test successful container listing."""
        server = DockerMCPServer(project_root=temp_project_root)

        # Add managed container
        mock_managed = Mock()
        server.managed_containers["managed123"] = mock_managed

        # Mock container list
        mock_container = Mock()
        mock_container.id = "container123"
        mock_container.name = "test_container"
        mock_container.status = "running"
        mock_container.image.tags = ["python:3.13-slim"]
        mock_container.attrs = {
            "Created": "2024-01-01T00:00:00Z",
            "NetworkSettings": {"Ports": {"80/tcp": [{"HostPort": "8080"}]}},
        }

        mock_docker_client.containers.list.return_value = [mock_container]

        result = await server._list_containers(all_containers=True)

        assert len(result) == 1
        result_data = json.loads(result[0].text)
        assert result_data["count"] == 1
        assert len(result_data["managed_containers"]) == 1
        assert "managed123" in result_data["managed_containers"]

        container_info = result_data["containers"][0]
        assert container_info["id"] == "container123"
        assert container_info["name"] == "test_container"
        assert container_info["status"] == "running"

    @pytest.mark.asyncio
    async def test_list_images_success(self, mock_docker_client, temp_project_root):
        """Test successful image listing."""
        server = DockerMCPServer(project_root=temp_project_root)

        mock_image = Mock()
        mock_image.id = "sha256:abc123"
        mock_image.tags = ["python:3.13-slim", "python:latest"]
        mock_image.attrs = {"Size": 1024000, "Created": "2024-01-01T00:00:00Z"}

        mock_docker_client.images.list.return_value = [mock_image]

        result = await server._list_images()

        assert len(result) == 1
        result_data = json.loads(result[0].text)
        assert result_data["count"] == 1

        image_info = result_data["images"][0]
        assert image_info["id"] == "sha256:abc123"
        assert len(image_info["tags"]) == 2
        assert image_info["size"] == 1024000

    @pytest.mark.asyncio
    async def test_remove_image_success(self, mock_docker_client, temp_project_root):
        """Test successful image removal."""
        server = DockerMCPServer(project_root=temp_project_root)

        result = await server._remove_image("test:latest", force=True)

        assert len(result) == 1
        result_data = json.loads(result[0].text)
        assert result_data["status"] == "success"
        assert result_data["action"] == "removed"
        mock_docker_client.images.remove.assert_called_once_with(
            image="test:latest", force=True
        )

    @pytest.mark.asyncio
    async def test_cleanup_resources_containers(
        self, mock_docker_client, temp_project_root
    ):
        """Test cleanup of managed containers."""
        server = DockerMCPServer(project_root=temp_project_root)

        # Add managed containers
        mock_container1 = Mock()
        mock_container2 = Mock()
        server.managed_containers["container1"] = mock_container1
        server.managed_containers["container2"] = mock_container2

        result = await server._cleanup_resources(containers=True)

        assert len(result) == 1
        result_data = json.loads(result[0].text)
        assert result_data["status"] == "success"
        assert result_data["cleanup_results"]["containers"]["count"] == 2
        assert len(server.managed_containers) == 0

        mock_container1.remove.assert_called_once_with(force=True)
        mock_container2.remove.assert_called_once_with(force=True)

    @pytest.mark.asyncio
    async def test_cleanup_resources_images(
        self, mock_docker_client, temp_project_root
    ):
        """Test cleanup of dangling images."""
        server = DockerMCPServer(project_root=temp_project_root)

        mock_docker_client.images.prune.return_value = {
            "SpaceReclaimed": 5242880,
            "ImagesDeleted": [{"Deleted": "sha256:abc123"}],
        }

        result = await server._cleanup_resources(images=True)

        assert len(result) == 1
        result_data = json.loads(result[0].text)
        assert result_data["cleanup_results"]["images"]["reclaimed_space"] == 5242880
        assert result_data["cleanup_results"]["images"]["deleted_images"] == 1

    @pytest.mark.asyncio
    async def test_cleanup_resources_volumes(
        self, mock_docker_client, temp_project_root
    ):
        """Test cleanup of unused volumes."""
        server = DockerMCPServer(project_root=temp_project_root)

        mock_docker_client.volumes.prune.return_value = {
            "SpaceReclaimed": 1024000,
            "VolumesDeleted": [{"Name": "volume1"}],
        }

        result = await server._cleanup_resources(volumes=True)

        assert len(result) == 1
        result_data = json.loads(result[0].text)
        assert result_data["cleanup_results"]["volumes"]["reclaimed_space"] == 1024000
        assert result_data["cleanup_results"]["volumes"]["deleted_volumes"] == 1


class TestDockerMCPLauncher:
    """Test cases for DockerMCPLauncher."""

    def test_init(self, temp_project_root):
        """Test launcher initialization."""
        launcher = DockerMCPLauncher(project_root=temp_project_root)

        assert launcher.project_root == temp_project_root
        assert launcher.process is None

    def test_init_default_project_root(self):
        """Test launcher initialization with default project root."""
        launcher = DockerMCPLauncher()

        assert launcher.project_root == str(Path.cwd())

    @patch("subprocess.Popen")
    def test_start(self, mock_popen, temp_project_root):
        """Test launcher start."""
        mock_process = Mock()
        mock_popen.return_value = mock_process

        launcher = DockerMCPLauncher(project_root=temp_project_root)
        result = launcher.start()

        assert result == mock_process
        assert launcher.process == mock_process

        # Verify subprocess call
        mock_popen.assert_called_once()
        args, kwargs = mock_popen.call_args
        assert temp_project_root in args[0]
        assert kwargs["stdin"] == subprocess.PIPE
        assert kwargs["stdout"] == subprocess.PIPE
        assert kwargs["stderr"] == subprocess.PIPE

    def test_stop_running_process(self, temp_project_root):
        """Test stopping a running process."""
        launcher = DockerMCPLauncher(project_root=temp_project_root)

        mock_process = Mock()
        mock_process.poll.return_value = None  # Still running
        launcher.process = mock_process

        launcher.stop()

        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once_with(timeout=5)
        assert launcher.process is None

    def test_stop_timeout_kill(self, temp_project_root):
        """Test stopping process with timeout and kill."""
        launcher = DockerMCPLauncher(project_root=temp_project_root)

        mock_process = Mock()
        mock_process.wait.side_effect = [subprocess.TimeoutExpired("test", 5), None]
        launcher.process = mock_process

        launcher.stop()

        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()
        assert launcher.process is None

    def test_is_running_true(self, temp_project_root):
        """Test is_running when process is active."""
        launcher = DockerMCPLauncher(project_root=temp_project_root)

        mock_process = Mock()
        mock_process.poll.return_value = None
        launcher.process = mock_process

        assert launcher.is_running() is True

    def test_is_running_false_no_process(self, temp_project_root):
        """Test is_running when no process."""
        launcher = DockerMCPLauncher(project_root=temp_project_root)

        assert launcher.is_running() is False

    def test_is_running_false_stopped_process(self, temp_project_root):
        """Test is_running when process has stopped."""
        launcher = DockerMCPLauncher(project_root=temp_project_root)

        mock_process = Mock()
        mock_process.poll.return_value = 0  # Exited
        launcher.process = mock_process

        assert launcher.is_running() is False

    def test_context_manager(self, temp_project_root):
        """Test launcher as context manager."""
        with patch.object(DockerMCPLauncher, "start") as mock_start:
            with patch.object(DockerMCPLauncher, "stop") as mock_stop:
                mock_start.return_value = Mock()

                with DockerMCPLauncher(project_root=temp_project_root) as launcher:
                    assert launcher is not None

                mock_start.assert_called_once()
                mock_stop.assert_called_once()


class TestDockerMCPIntegration:
    """Integration tests for Docker MCP Server components."""

    def test_server_tool_registration(self, mock_docker_client, temp_project_root):
        """Test that all expected tools are registered."""
        server = DockerMCPServer(project_root=temp_project_root)

        # This tests that the server can be initialized and tools are registered
        # without errors (actual tool list testing would require async setup)
        assert server.server is not None
        assert hasattr(server, "_register_tools")

    @pytest.mark.asyncio
    async def test_error_handling_in_tool_calls(
        self, mock_docker_client, temp_project_root
    ):
        """Test error handling in tool calls."""
        server = DockerMCPServer(project_root=temp_project_root)

        # Mock an exception in Docker operations
        mock_docker_client.containers.get.side_effect = Exception("Docker error")

        result = await server._stop_container("nonexistent")

        assert len(result) == 1
        assert "Stop container error" in result[0].text
