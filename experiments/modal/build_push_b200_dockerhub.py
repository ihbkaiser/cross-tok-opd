"""Build and push the portable B200 image from a Modal VM Sandbox.

The registry credentials are supplied only through a named Modal Secret with
REGISTRY_USERNAME and REGISTRY_PASSWORD fields. The script never receives raw
credential values as arguments and never writes them into the build context.
"""

from __future__ import annotations

import json
import re
import shlex

import modal

from experiments.modal.b200_docker_contract import (
    LOCAL_ROOT,
    audit_local_context,
    ignore_local_path,
    normalize_image_ref,
    validate_secret_name,
)


APP_NAME = "vdt-simct-b200-docker-builder"
DOCKERFILE = "/build/context/docker/Dockerfile.b200-cu130"
CONTEXT = "/build/context"
MODAL_SDK_MINIMUM = "1.4.0"


def sandbox_image() -> modal.Image:
    return (
        modal.Image.from_registry("ubuntu:24.04")
        .env({"DEBIAN_FRONTEND": "noninteractive"})
        .apt_install("ca-certificates", "docker.io", "docker-buildx")
        .run_commands("mkdir -p /build/context")
        .add_local_dir(
            LOCAL_ROOT,
            CONTEXT,
            copy=True,
            ignore=ignore_local_path,
        )
    )


def wait_checked(process: modal.container_process.ContainerProcess, label: str) -> None:
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        detail = process.stderr.read().strip()
        if detail:
            print(detail)
        raise RuntimeError(f"{label} failed with exit code {process.returncode}")


app = modal.App(APP_NAME)


@app.local_entrypoint()
def main(
    image: str,
    registry_secret: str,
    repo_commit: str,
    timeout_seconds: int = 10_800,
) -> None:
    image_ref = normalize_image_ref(image)
    secret_name = validate_secret_name(registry_secret)
    if not re.fullmatch(r"[0-9a-f]{40}", repo_commit):
        raise ValueError("repo commit must be an exact 40-character lowercase SHA")
    if not 600 <= timeout_seconds <= 21_600:
        raise ValueError("timeout must be between 600 and 21600 seconds")

    audit = audit_local_context()
    print("SIMCT_DOCKER_CONTEXT_JSON=" + json.dumps(audit, sort_keys=True))

    registry_secret = modal.Secret.from_name(secret_name)
    sandbox: modal.Sandbox | None = None
    logged_in = False
    try:
        with modal.enable_output():
            sandbox = modal.Sandbox.create(
                "/usr/bin/dockerd",
                "-D",
                app=app,
                image=sandbox_image(),
                cpu=8,
                memory=32_768,
                timeout=timeout_seconds,
                secrets=[registry_secret],
                experimental_options={"vm_runtime": True},
            )

        ready = sandbox.exec(
            "sh",
            "-lc",
            "for i in $(seq 1 180); do "
            "if [ -S /var/run/docker.sock ] && docker info >/dev/null 2>&1; then "
            "echo SIMCT_DOCKER_DAEMON=ready; exit 0; fi; sleep 1; done; "
            "echo SIMCT_DOCKER_DAEMON=timeout >&2; exit 1",
        )
        wait_checked(ready, "Docker daemon readiness")

        login = sandbox.exec(
            "sh",
            "-lc",
            "test -n \"${REGISTRY_USERNAME:-}\" "
            "&& test -n \"${REGISTRY_PASSWORD:-}\" "
            "&& printf %s \"$REGISTRY_PASSWORD\" "
            "| docker login docker.io --username \"$REGISTRY_USERNAME\" "
            "--password-stdin >/dev/null 2>&1",
        )
        login.wait()
        if login.returncode != 0:
            raise RuntimeError("Docker Hub authentication failed")
        logged_in = True
        print("SIMCT_DOCKER_REGISTRY_AUTH=pass")

        builder = sandbox.exec(
            "sh",
            "-lc",
            "docker buildx create --name simct-b200-builder --driver docker-container --use "
            ">/dev/null 2>&1 || docker buildx use simct-b200-builder; "
            "docker buildx inspect --bootstrap >/dev/null 2>&1; "
            "echo SIMCT_BUILDX=ready",
        )
        wait_checked(builder, "buildx bootstrap")

        quoted_ref = shlex.quote(image_ref)
        quoted_commit = shlex.quote(repo_commit)
        build_command = (
            "set -o pipefail; "
            "docker buildx build "
            "--platform linux/amd64 "
            f"--file {shlex.quote(DOCKERFILE)} "
            f"--tag {quoted_ref} "
            f"--build-arg REPO_COMMIT={quoted_commit} "
            "--provenance=mode=max --sbom=true "
            "--output type=registry,compression=estargz,force-compression=true,oci-mediatypes=true "
            "--metadata-file /tmp/simct-build-metadata.json "
            f"{shlex.quote(CONTEXT)} 2>&1"
        )
        build = sandbox.exec("bash", "-lc", build_command)
        wait_checked(build, "B200 image build and push")

        inspect_process = sandbox.exec(
            "sh",
            "-lc",
            f"docker buildx imagetools inspect {quoted_ref} "
            "--format '{{json .Manifest}}' 2>/dev/null",
        )
        inspect_process.wait()
        if inspect_process.returncode != 0:
            raise RuntimeError("pushed image inspection failed")
        manifest = inspect_process.stdout.read().strip()
        print(
            "SIMCT_DOCKER_PUSH_JSON="
            + json.dumps(
                {
                    "status": "pushed",
                    "image": image_ref,
                    "repo_commit": repo_commit,
                    "manifest": json.loads(manifest),
                },
                sort_keys=True,
            )
        )
    finally:
        if sandbox is not None:
            if logged_in:
                logout = sandbox.exec(
                    "sh", "-lc", "docker logout docker.io >/dev/null 2>&1 || true"
                )
                logout.wait()
            sandbox.terminate(wait=True)
