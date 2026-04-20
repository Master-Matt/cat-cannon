from pathlib import Path

from cat_cannon.app.deploy_jetson import (
    JetsonDeployConfig,
    build_bootstrap_command,
    build_rsync_command,
    build_ssh_command,
)


def test_build_bootstrap_command_installs_repo_with_extras() -> None:
    config = JetsonDeployConfig(
        host="192.168.55.1",
        user="mdev",
        password=None,
        remote_dir="/home/mdev/cat_cannon",
        extras=("dev", "vision"),
        restart_service=False,
        install_service=False,
        service_name="cat-cannon",
        skip_system_packages=False,
    )

    command = build_bootstrap_command(config)

    assert 'python3 -m venv "$tmpdir/.venv-check" >/dev/null 2>&1' in command
    assert "DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv" in command
    assert "apt-get update || true" in command
    assert "mkdir -p /home/mdev/cat_cannon" in command
    assert "cd /home/mdev/cat_cannon" in command
    assert "python3 -m venv .venv" in command
    assert ".venv/bin/python -m pip install -e '.[dev,vision]'" in command


def test_build_bootstrap_command_can_restart_and_install_service() -> None:
    config = JetsonDeployConfig(
        host="192.168.55.1",
        user="mdev",
        password="nvidia",
        remote_dir="/opt/cat-cannon",
        extras=("vision",),
        restart_service=True,
        install_service=True,
        service_name="cat-cannon",
        skip_system_packages=False,
    )

    command = build_bootstrap_command(config)

    assert "printf '%s\\n' nvidia | sudo -S -p '' sh -lc 'tmpdir=$(mktemp -d)" in command
    assert "DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv" in command
    assert "apt-get update || true" in command
    assert "printf '%s\\n' nvidia | sudo -S -p '' sh -lc 'cp /opt/cat-cannon/systemd/cat-cannon.service /etc/systemd/system/cat-cannon.service'" in command
    assert "printf '%s\\n' nvidia | sudo -S -p '' sh -lc 'systemctl daemon-reload'" in command
    assert "printf '%s\\n' nvidia | sudo -S -p '' sh -lc 'systemctl enable --now cat-cannon.service'" in command
    assert "printf '%s\\n' nvidia | sudo -S -p '' sh -lc 'systemctl restart cat-cannon.service'" in command


def test_build_bootstrap_command_can_skip_system_package_bootstrap() -> None:
    config = JetsonDeployConfig(
        host="192.168.55.1",
        user="mdev",
        password=None,
        remote_dir="/home/mdev/cat_cannon",
        extras=("dev",),
        restart_service=False,
        install_service=False,
        service_name="cat-cannon",
        skip_system_packages=True,
    )

    command = build_bootstrap_command(config)

    assert "apt-get install -y python3-venv" not in command
    assert "python3 -m venv .venv" in command


def test_build_ssh_command_uses_sshpass_prefix_when_password_present() -> None:
    config = JetsonDeployConfig(
        host="192.168.55.1",
        user="mdev",
        password="nvidia",
        remote_dir="/home/mdev/cat_cannon",
        extras=("dev",),
        restart_service=False,
        install_service=False,
        service_name="cat-cannon",
        skip_system_packages=False,
    )

    command = build_ssh_command(config, control_path="/tmp/cat-cannon-ctrl", remote_command="echo ok")

    assert command[:3] == ["sshpass", "-p", "nvidia"]
    assert "mdev@192.168.55.1" in command
    assert "echo ok" in command


def test_build_rsync_command_excludes_local_virtualenv_and_cache() -> None:
    config = JetsonDeployConfig(
        host="192.168.55.1",
        user="mdev",
        password=None,
        remote_dir="/home/mdev/cat_cannon",
        extras=("dev",),
        restart_service=False,
        install_service=False,
        service_name="cat-cannon",
        skip_system_packages=False,
    )

    command = build_rsync_command(
        config,
        control_path="/tmp/cat-cannon-ctrl",
        root_dir=Path("/workspace/cat_cannon"),
    )

    assert "--exclude=.venv/" in command
    assert "--exclude=.pytest_cache/" in command
    assert "--exclude=everything-claude-code/" in command
    assert command[-2:] == ["/workspace/cat_cannon/", "mdev@192.168.55.1:/home/mdev/cat_cannon/"]
