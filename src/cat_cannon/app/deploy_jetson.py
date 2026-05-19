from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


DEFAULT_HOST = "192.168.0.218"
DEFAULT_USER = "mdev"
DEFAULT_SERVICE_NAME = "cat-cannon"


class JetsonDeployError(RuntimeError):
    """Raised when the Jetson deployment workflow cannot proceed."""


@dataclass(frozen=True)
class JetsonDeployConfig:
    host: str
    user: str
    password: str | None
    remote_dir: str
    extras: tuple[str, ...]
    restart_service: bool
    install_service: bool
    service_name: str
    skip_system_packages: bool


@dataclass(frozen=True)
class DeployStep:
    name: str
    command: list[str]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy Cat Cannon to a Jetson over SSH.")
    parser.add_argument("--host", default=os.environ.get("JETSON_HOST", DEFAULT_HOST), help="Jetson hostname or IP")
    parser.add_argument("--user", default=os.environ.get("JETSON_USER", DEFAULT_USER), help="Jetson SSH username")
    parser.add_argument(
        "--password",
        default=os.environ.get("JETSON_PASSWORD"),
        help="Optional SSH password; prefers sshpass if installed",
    )
    parser.add_argument(
        "--remote-dir",
        default=os.environ.get("JETSON_REMOTE_DIR", f"/home/{DEFAULT_USER}/cat_cannon"),
        help="Remote deployment directory",
    )
    parser.add_argument(
        "--extras",
        default="dev,bench,vision",
        help="Comma-separated extras to install remotely",
    )
    parser.add_argument("--restart-service", action="store_true", help="Restart the systemd service after deploy")
    parser.add_argument(
        "--install-service",
        action="store_true",
        help="Install/update the bundled systemd unit before restart",
    )
    parser.add_argument(
        "--skip-system-packages",
        action="store_true",
        help="Skip apt-based bootstrap steps such as python3-venv installation",
    )
    parser.add_argument("--service-name", default=DEFAULT_SERVICE_NAME, help="systemd service base name")
    return parser.parse_args(argv)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ssh_prefix(config: JetsonDeployConfig) -> list[str]:
    if config.password and shutil.which("sshpass"):
        return ["sshpass", "-p", config.password]
    return []


def _ssh_options(control_path: str) -> list[str]:
    return [
        "-o",
        "ControlMaster=auto",
        "-o",
        "ControlPersist=60",
        "-o",
        f"ControlPath={control_path}",
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]


def _sudo_shell(config: JetsonDeployConfig, command: str) -> str:
    quoted_command = shlex.quote(command)
    if config.password:
        password = shlex.quote(config.password)
        return f"printf '%s\\n' {password} | sudo -S -p '' sh -lc {quoted_command}"
    return f"sudo sh -lc {quoted_command}"


def build_ssh_command(
    config: JetsonDeployConfig,
    *,
    control_path: str,
    remote_command: str,
) -> list[str]:
    return [
        *_ssh_prefix(config),
        "ssh",
        *_ssh_options(control_path),
        f"{config.user}@{config.host}",
        remote_command,
    ]


def build_rsync_command(
    config: JetsonDeployConfig,
    *,
    control_path: str,
    root_dir: Path,
) -> list[str]:
    ssh_transport = shlex.join(["ssh", *_ssh_options(control_path)])
    excludes = [
        ".git/",
        ".venv/",
        ".pytest_cache/",
        "__pycache__/",
        ".ruff_cache/",
        ".mypy_cache/",
        "*.pyc",
        "*.engine*",
        "everything-claude-code/",
        "data/",
        # Preserve user-edited configs on the remote
        "configs/zones.yaml",
        "configs/app.yaml",
    ]
    command = [
        *_ssh_prefix(config),
        "rsync",
        "-az",
        "--delete",
    ]
    for exclude in excludes:
        command.append(f"--exclude={exclude}")
    command.extend(
        [
            "-e",
            ssh_transport,
            f"{root_dir}/",
            f"{config.user}@{config.host}:{config.remote_dir}/",
        ]
    )
    return command


_NV_TORCH_WHEEL = (
    "https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/"
    "torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
)


def _without_vision(extras_suffix: str) -> str:
    """Remove the 'vision' extra from an extras suffix like '[dev,bench,vision]'.

    The vision extra pulls ultralytics→torch→torchvision which conflicts
    with the NVIDIA torch wheel.  We install those in jetson-gpu-setup instead.
    """
    if not extras_suffix:
        return ""
    inner = extras_suffix.strip("[]")
    parts = [p.strip() for p in inner.split(",") if p.strip().lower() != "vision"]
    if not parts:
        return ""
    return f"[{','.join(parts)}]"


def build_jetson_gpu_setup_command(config: JetsonDeployConfig) -> str:
    """Build command to install NVIDIA torch stack and patch torchvision for Jetson."""
    remote_dir = shlex.quote(config.remote_dir)
    site_packages = f"{config.remote_dir}/.venv/lib/python3.10/site-packages"
    patch_script = f"{config.remote_dir}/scripts/patch_torchvision_jetson.py"
    commands = [
        f"cd {remote_dir}",
        # Install NVIDIA torch wheel (replaces PyPI torch)
        f".venv/bin/pip install --no-cache '{_NV_TORCH_WHEEL}'",
        # Pin numpy < 2 (NV torch compiled against numpy 1.x)
        ".venv/bin/pip install --no-cache 'numpy<2'",
        # Install torchvision 0.19.0 without deps (ABI-compat with NV torch 2.5)
        ".venv/bin/pip install --no-cache --no-deps 'torchvision==0.19.0'",
        # Install ultralytics without deps (avoids torch resolution from PyPI)
        ".venv/bin/pip install --no-cache --no-deps 'ultralytics>=8.4'",
        # Install ultralytics' non-torch runtime deps
        ".venv/bin/pip install --no-cache matplotlib pillow requests scipy psutil polars ultralytics-thop",
        # Install onnx for engine export
        ".venv/bin/pip install --no-cache onnx onnxslim",
        # Run the torchvision patch script (rsynced from repo)
        f".venv/bin/python {shlex.quote(patch_script)} {shlex.quote(site_packages)}",
    ]
    return " && ".join(commands)


def build_udev_install_command(config: JetsonDeployConfig) -> str:
    """Build command to install udev rules for persistent camera symlinks."""
    rules_src = f"{config.remote_dir}/systemd/99-cat-cannon-cameras.rules"
    rules_dst = "/etc/udev/rules.d/99-cat-cannon-cameras.rules"
    commands = [
        _sudo_shell(config, f"cp {shlex.quote(rules_src)} {rules_dst}"),
        _sudo_shell(config, "udevadm control --reload-rules"),
        _sudo_shell(config, "udevadm trigger"),
    ]
    return " && ".join(commands)


def build_bootstrap_command(config: JetsonDeployConfig) -> str:
    extras_suffix = f"[{','.join(config.extras)}]" if config.extras else ""
    remote_dir = shlex.quote(config.remote_dir)
    commands = []
    if not config.skip_system_packages:
        ensure_venv_command = (
            "tmpdir=$(mktemp -d) && "
            "if python3 -m venv \"$tmpdir/.venv-check\" >/dev/null 2>&1; then "
            "rm -rf \"$tmpdir\"; "
            "else "
            "rm -rf \"$tmpdir\"; "
            "DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv || "
            "(apt-get update || true) && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv; "
            "fi"
        )
        commands.append(_sudo_shell(config, ensure_venv_command))
    commands.extend(
        [
            f"mkdir -p {remote_dir}",
            f"cd {remote_dir}",
            "python3 -m venv .venv",
            ".venv/bin/python -m pip install --upgrade pip",
            # Install WITHOUT 'vision' extra — torch/torchvision are handled
            # by jetson-gpu-setup to avoid pip pulling incompatible PyPI wheels.
            f".venv/bin/python -m pip install -e '.{_without_vision(extras_suffix)}'",
        ]
    )
    if config.install_service:
        service_file = f"{config.service_name}.service"
        commands.extend(
            [
                _sudo_shell(
                    config,
                    f"cp {shlex.quote(config.remote_dir)}/systemd/{service_file} /etc/systemd/system/{service_file}",
                ),
                _sudo_shell(config, "systemctl daemon-reload"),
                _sudo_shell(config, f"systemctl enable --now {service_file}"),
            ]
        )
    if config.restart_service:
        commands.append(_sudo_shell(config, f"systemctl restart {config.service_name}.service"))
    return " && ".join(commands)


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def build_deploy_steps(
    config: JetsonDeployConfig,
    *,
    control_path: str,
) -> tuple[list[DeployStep], list[str]]:
    root_dir = _repo_root()

    bootstrap_connect = build_ssh_command(
        config,
        control_path=control_path,
        remote_command=f"mkdir -p {shlex.quote(config.remote_dir)}",
    )
    sync_command = build_rsync_command(config, control_path=control_path, root_dir=root_dir)
    bootstrap_command = build_bootstrap_command(config)
    remote_bootstrap = build_ssh_command(
        config,
        control_path=control_path,
        remote_command=bootstrap_command,
    )
    gpu_setup_command = build_jetson_gpu_setup_command(config)
    remote_gpu_setup = build_ssh_command(
        config,
        control_path=control_path,
        remote_command=gpu_setup_command,
    )
    udev_command = build_udev_install_command(config)
    remote_udev = build_ssh_command(
        config,
        control_path=control_path,
        remote_command=udev_command,
    )
    # Seed config files from examples if they don't already exist on the remote
    seed_configs_cmd = (
        f"cd {shlex.quote(config.remote_dir)} && "
        "cp -n configs/app.example.yaml configs/app.yaml 2>/dev/null; "
        "cp -n configs/zones.example.yaml configs/zones.yaml 2>/dev/null; "
        "true"
    )
    remote_seed_configs = build_ssh_command(
        config,
        control_path=control_path,
        remote_command=seed_configs_cmd,
    )
    # Install desktop shortcut
    desktop_cmd = (
        f"mkdir -p /home/{config.user}/Desktop && "
        f"cp {shlex.quote(config.remote_dir)}/systemd/cat-cannon.desktop "
        f"/home/{config.user}/Desktop/cat-cannon.desktop && "
        f"chmod +x /home/{config.user}/Desktop/cat-cannon.desktop && "
        f"gio set /home/{config.user}/Desktop/cat-cannon.desktop metadata::trusted true 2>/dev/null; true"
    )
    remote_desktop = build_ssh_command(
        config,
        control_path=control_path,
        remote_command=desktop_cmd,
    )
    close_command = build_ssh_command(
        config,
        control_path=control_path,
        remote_command="true",
    )
    close_command = [*close_command[:-2], "-O", "exit", close_command[-2], close_command[-1]]

    steps = [
        DeployStep(name="ssh-bootstrap", command=bootstrap_connect),
        DeployStep(name="rsync", command=sync_command),
        DeployStep(name="seed-configs", command=remote_seed_configs),
        DeployStep(name="remote-bootstrap", command=remote_bootstrap),
        DeployStep(name="jetson-gpu-setup", command=remote_gpu_setup),
        DeployStep(name="udev-install", command=remote_udev),
        DeployStep(name="desktop-shortcut", command=remote_desktop),
    ]
    return steps, close_command


def deploy(config: JetsonDeployConfig) -> None:
    for binary in ("ssh", "rsync"):
        if shutil.which(binary) is None:
            raise JetsonDeployError(f"Required command not found: {binary}")
    if config.password and shutil.which("sshpass") is None:
        print("[jetson] sshpass not found; SSH will prompt for the password interactively.")
    with tempfile.TemporaryDirectory(prefix="cat-cannon-ssh-") as temp_dir:
        control_path = str(Path(temp_dir) / "control")
        steps, close_command = build_deploy_steps(config, control_path=control_path)
        try:
            for step in steps:
                print(f"[jetson] running {step.name}: {shlex.join(step.command)}")
                _run(step.command)
        except subprocess.CalledProcessError as exc:
            raise JetsonDeployError(
                f"step '{step.name}' failed with exit code {exc.returncode}\n"
                f"command: {shlex.join(step.command)}"
            ) from exc
        finally:
            subprocess.run(close_command, check=False)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    extras = tuple(part.strip() for part in args.extras.split(",") if part.strip())
    config = JetsonDeployConfig(
        host=args.host,
        user=args.user,
        password=args.password,
        remote_dir=args.remote_dir,
        extras=extras,
        restart_service=bool(args.restart_service),
        install_service=bool(args.install_service),
        service_name=args.service_name,
        skip_system_packages=bool(args.skip_system_packages),
    )
    try:
        deploy(config)
    except JetsonDeployError as exc:
        print(f"[jetson] deployment failed: {exc}")
        return 1

    print(f"[jetson] deployment complete: {config.user}@{config.host}:{config.remote_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
