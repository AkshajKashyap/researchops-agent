from pathlib import Path


def test_dockerfile_exists() -> None:
    assert Path("Dockerfile").exists()


def test_docker_compose_exists() -> None:
    assert Path("docker-compose.yml").exists()


def test_dockerignore_exists() -> None:
    assert Path(".dockerignore").exists()
