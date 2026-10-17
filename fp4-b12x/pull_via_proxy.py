#!/usr/bin/env python3
"""Pull a Docker Hub image through an HTTP proxy without touching dockerd, then docker-load it.

Why not `docker pull`: the pull is performed by dockerd, so only a daemon-level proxy setting reaches
Docker Hub. Setting one means editing a systemd drop-in and restarting dockerd, which would restart
seven containers on this box (homeassistant up two weeks, hermes seven days, the multica stack, and
lp-ear). Docker CLI's own `proxies` config injects proxy variables into containers, not into the
daemon's registry calls. So this does the registry protocol in userspace, where HTTPS_PROXY is
honoured, writes an OCI-style docker archive, and hands that to `docker load`.

Registry steps: token -> manifest -> config blob -> layer blobs -> assemble. Layers are decompressed
to plain tars because that is what docker load expects. Already-downloaded layers are reused, so a
run can be resumed. Uncompressed media types are copied through unchanged.

Usage:  HTTPS_PROXY=http://127.0.0.1:7890 python3 pull_via_proxy.py eugr/spark-vllm-b12x latest
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

REGISTRY = "https://registry-1.docker.io"
AUTH = "https://auth.docker.io/token"
WORK = Path("/home/xujie/dl/userpull")
PROXY = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or ""

GZIP_TYPES = ("application/vnd.docker.image.rootfs.diff.tar.gzip",
              "application/vnd.oci.image.layer.v1.tar+gzip")
RAW_TYPES = ("application/vnd.docker.image.rootfs.diff.tar",
             "application/vnd.oci.image.layer.v1.tar",
             "application/vnd.oci.image.layer.nondistributable.v1.tar")


def opener() -> urllib.request.OpenerDirector:
    handlers = []
    if PROXY:
        handlers.append(urllib.request.ProxyHandler({"http": PROXY, "https": PROXY}))
    else:
        handlers.append(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener(*handlers)


def get(url: str, headers: dict[str, str] | None = None, stream: bool = False):
    req = urllib.request.Request(url, headers=headers or {})
    return opener().open(req, timeout=180)


def token(repo: str) -> str:
    url = "%s?service=registry.docker.io&scope=repository:%s:pull" % (AUTH, repo)
    with get(url) as r:
        return json.load(r)["token"]


def manifest(repo: str, ref: str, tok: str) -> dict:
    accept = ", ".join([
        "application/vnd.docker.distribution.manifest.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.index.v1+json",
    ])
    with get("%s/v2/%s/manifests/%s" % (REGISTRY, repo, ref),
             {"Authorization": "Bearer " + tok, "Accept": accept}) as r:
        doc = json.load(r)
    if "manifests" in doc:                      # a list: pick arm64 for this hardware
        want = next((m for m in doc["manifests"]
                     if (m.get("platform") or {}).get("architecture") == "arm64"), None)
        want = want or doc["manifests"][0]
        print("  list -> %s/%s %s" % ((want.get("platform") or {}).get("os"),
                                      (want.get("platform") or {}).get("architecture"),
                                      want["digest"][:16]))
        return manifest(repo, want["digest"], tok)
    return doc


def fetch_blob(repo: str, digest: str, dest: Path, tok: str) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with get("%s/v2/%s/blobs/%s" % (REGISTRY, repo, digest),
             {"Authorization": "Bearer " + tok}, stream=True) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 22)
    tmp.rename(dest)


def main() -> int:
    repo = sys.argv[1] if len(sys.argv) > 1 else "eugr/spark-vllm-b12x"
    ref = sys.argv[2] if len(sys.argv) > 2 else "latest"
    tag = "%s:%s" % (repo, ref)

    if not PROXY:
        print("no HTTPS_PROXY set; Docker Hub is unreachable from this box without it")
        return 1
    print("proxy  %s" % PROXY)
    print("image  %s" % tag)

    WORK.mkdir(parents=True, exist_ok=True)
    layers_dir = WORK / "layers"
    layers_dir.mkdir(exist_ok=True)

    tok = token(repo)
    man = manifest(repo, ref, tok)
    cfg_digest = man["config"]["digest"]
    layers = man["layers"]
    total = sum(l["size"] for l in layers)
    print("config %s" % cfg_digest[:20])
    print("layers %d, %.1f GiB compressed" % (len(layers), total / 2 ** 30))

    cfg_path = WORK / (cfg_digest.split(":")[1] + ".json")
    if not cfg_path.exists():
        fetch_blob(repo, cfg_digest, cfg_path, tok)
    cfg = json.loads(cfg_path.read_text())
    print("arch   %s/%s" % (cfg.get("os"), cfg.get("architecture")))
    if cfg.get("architecture") != "arm64":
        print("REFUSING: this image is %s, and this box is arm64" % cfg.get("architecture"))
        return 2

    entries: list[str] = []
    done_bytes = 0
    for i, layer in enumerate(layers):
        digest = layer["digest"]
        media = layer.get("mediaType", "")
        out = layers_dir / ("%02d_%s.tar" % (i, digest.split(":")[1][:12]))
        entries.append(out.name)
        if out.exists():
            done_bytes += layer["size"]
            continue
        blob = layers_dir / (digest.split(":")[1][:12] + ".blob")
        if not blob.exists():
            fetch_blob(repo, digest, blob, tok)
        if media in GZIP_TYPES:
            with gzip.open(blob, "rb") as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 22)
        elif media in RAW_TYPES:
            blob.rename(out)
        else:
            print("  unknown media type %r on layer %d" % (media, i))
            return 3
        if media in GZIP_TYPES and blob.exists():
            blob.unlink()
        done_bytes += layer["size"]
        pct = 100.0 * done_bytes / max(1, total)
        print("  [%2d/%2d] %5.1f%%  %s" % (i + 1, len(layers), pct, out.name), flush=True)

    man_obj = [{"Config": cfg_path.name, "RepoTags": [tag], "Layers": entries}]
    (WORK / "manifest.json").write_text(json.dumps(man_obj))

    archive = WORK / (repo.split("/")[-1] + "_" + ref + ".tar")
    print("packing %s" % archive)
    with tarfile.open(archive, "w") as tf:
        tf.add(WORK / "manifest.json", arcname="manifest.json")
        tf.add(cfg_path, arcname=cfg_path.name)
        for name in entries:
            tf.add(layers_dir / name, arcname=name)
    print("archive %.1f GiB" % (archive.stat().st_size / 2 ** 30))

    sha = hashlib.sha256()
    with open(archive, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            sha.update(chunk)
    print("sha256 %s" % sha.hexdigest()[:32])
    print()
    print("now:  docker load -i %s" % archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
