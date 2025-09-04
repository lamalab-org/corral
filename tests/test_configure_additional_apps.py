from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from corral.backend.env import Environment
from corral.backend.server import create_benchmark_server
from corral.router.routes import CorralRouter


class DummyEnv(Environment):
    def get_task_prompt(self) -> str | list[dict]:
        return "dummy prompt"

    def score(self) -> float:
        return 1.0

    def configure_additional_apps(self):
        return "configured ok"


def create_app_with_env(task_id: str = "task_a") -> TestClient:
    env = DummyEnv(task_id=task_id, base_work_dir="", fs_manager=None)
    app = create_benchmark_server({task_id: env})
    return TestClient(app)


class TestConfigureAdditionalAppsEndpoint:
    def test_successful_configuration_returns_status_and_ids(self):
        client = create_app_with_env("task_a")
        response = client.post("/tasks/task_a/configure")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "configured ok"
        assert data["task_id"] == "task_a"
        # trial_id starts at "0" on first reset_state during env init
        assert data["trial_id"] == "0"

    def test_unknown_task_returns_404(self):
        client = create_app_with_env("task_a")
        response = client.post("/tasks/unknown/configure")
        assert response.status_code == 404


class TestCorralRouterConfigureAdditionalApps:
    def test_router_calls_configure_endpoint_and_returns_json(self):
        router = CorralRouter(base_url="http://example.com")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "configured ok",
            "task_id": "task_a",
            "trial_id": "3",
        }

        with patch(
            "corral.router.routes.requests.post", return_value=mock_response
        ) as mock_post:
            data = router.configure_additional_apps("task_a")

        assert data["status"] == "configured ok"
        assert data["task_id"] == "task_a"
        assert data["trial_id"] == "3"

        mock_post.assert_called_once_with("http://example.com/tasks/task_a/configure")
