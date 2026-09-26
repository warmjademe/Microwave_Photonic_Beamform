"""下载本仓库的冻结数据附件，核对 SHA-256，按原目录解压。无需运行仿真。"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import urllib.request


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_members(root: Path, index: Path) -> int:
    """只核对文件完整性；不训练模型、不计算新的实验结果。"""
    opener = gzip.open if index.suffix == ".gz" else open
    with opener(index, "rt", encoding="utf-8") as stream:
        records = json.load(stream)
    for record in records:
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("文件索引路径越界")
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise ValueError("文件缺失或长度不同：" + record["path"])
        if sha256(path) != record["sha256"]:
            raise ValueError("文件校验失败：" + record["path"])
    return len(records)


def extract_data(bundle: tarfile.TarFile, root: Path) -> None:
    """仅接收数据目录内的普通文件/硬链接，兼容 Python 3.9；拒绝软链接和设备文件。"""
    root = root.resolve()
    def checked(name: str) -> Path:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("dataset_simulation",):
            raise ValueError("附件包含越界路径")
        target = root / relative
        if target.is_symlink() or not target.resolve().is_relative_to(root):
            raise ValueError("解压路径包含越界链接")
        return target
    for member in bundle:
        path = checked(member.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if member.isdir():
            path.mkdir(exist_ok=True)
        elif member.isfile():
            content = bundle.extractfile(member)
            if content is None:
                raise ValueError("普通文件缺少内容")
            with content, path.open("wb") as out:
                shutil.copyfileobj(content, out, length=4 * 1024 * 1024)
        elif member.islnk():
            source = checked(member.linkname)
            if not source.is_file():
                raise ValueError("硬链接目标缺失")
            if path.exists():
                if os.path.samefile(path, source):
                    continue
                path.unlink()
            os.link(source, path)
        else:
            raise ValueError("附件包含不支持的特殊文件")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--only", nargs="+", help="仅处理指定分组；默认全部")
    parser.add_argument("--list", action="store_true", help="列出附件，不下载")
    parser.add_argument("--extract", action="store_true", help="校验后恢复 dataset_simulation 目录")
    parser.add_argument("--verify-only", action="store_true", help="核对已恢复文件，不联网")
    args = parser.parse_args()
    root = args.root.resolve()
    descriptor = root / "dataset_simulation/RELEASE_DATA.json"
    manifest = json.loads(descriptor.read_text(encoding="utf-8"))
    assets = manifest["assets"]
    if args.only:
        unknown = set(args.only) - {item["group"] for item in assets}
        if unknown:
            parser.error("未知分组：" + ", ".join(sorted(unknown)))
        assets = [item for item in assets if item["group"] in args.only]
    for item in assets:
        print(f'{item["group"]}: {item["bytes"] / 2**20:.1f} MiB — {item["name"]}', flush=True)
        if args.list:
            continue
        index = root / item["file_index"]
        if sha256(index) != item["file_index_sha256"]:
            raise ValueError("附件文件索引改变")
        if args.verify_only:
            print("完整性核对通过：", verify_members(root, index), "个文件", flush=True)
            continue
        cache = root / ".downloads"
        cache.mkdir(exist_ok=True)
        archive = cache / item["name"]
        if not archive.exists() or sha256(archive) != item["sha256"]:
            temporary = archive.with_suffix(archive.suffix + ".part")
            request = urllib.request.Request(item["url"], headers={"User-Agent": "Microwave-Photonic-Beamform"})
            digest = hashlib.sha256()
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
                for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                    out.write(block)
                    digest.update(block)
            if temporary.stat().st_size != item["bytes"] or digest.hexdigest() != item["sha256"]:
                raise ValueError("附件校验失败，保留 .part 供诊断：" + item["name"])
            temporary.replace(archive)
        if args.extract:
            with tarfile.open(archive, "r:gz") as bundle:
                extract_data(bundle, root)
            print("解压并核对通过：", verify_members(root, index), "个文件", flush=True)
    print("完成", flush=True)


if __name__ == "__main__":
    main()
