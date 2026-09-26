"""仅访问 train 环境；不使用验证或测试数据选模型。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline_common.data import Dataset, features
from baseline_mlp.method import fit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("输出路径已存在，拒绝覆盖检查点。")
    if min(args.epochs, args.batch_size, args.hidden) <= 0:
        parser.error("轮数、批量和隐层维数必须为正。")
    dataset = Dataset(args.dataset)
    environments = dataset.environments("train")
    if not environments:
        parser.error("数据集中没有训练环境。")
    rows, targets, environment_ids = [], [], []
    for environment in environments:
        # path 是公共数据目录标识，既不打开环境真值，也不读取测试记录。
        identifier = environment.get("id", environment.get("environment_id", environment["path"]))
        environment_ids.append(str(identifier))
        for carrier_index in range(len(dataset.carriers)):
            observation = dataset.observations(environment, carrier_index)
            label = dataset.labels(environment, carrier_index)
            if "control" not in label:
                raise ValueError("训练标签缺少归一化 control 字段。")
            rows.append(features(observation))
            targets.append(np.asarray(label["control"], np.float64))
    def report(record):
        print(json.dumps(dict(split="train", **record), ensure_ascii=False), flush=True)
    model = fit(np.stack(rows), np.stack(targets), epochs=args.epochs,
                batch_size=args.batch_size, hidden=args.hidden, seed=args.seed, callback=report)
    root = Path(__file__).resolve().parents[1]
    source_files = [Path(__file__), Path(__file__).with_name("method.py"),
                    root / "baseline_common/data.py", root / "baseline_common/controls.py"]
    manifest = dataset.manifest
    metadata = dict(
        training_environment_ids=environment_ids,
        training_environment_count=len(environment_ids),
        training_sample_count=len(rows),
        train_carriers_ghz=list(dataset.carriers),
        dataset_manifest_sha256=hashlib.sha256((args.dataset / "manifest.json").read_bytes()).hexdigest(),
        dataset_manifest_summary={key: manifest[key] for key in
                                  ("schema", "status", "master_seed", "signal_config", "carriers_ghz")
                                  if key in manifest},
        model_selection="Fixed final epoch; no validation/test data loaded or scored.",
        source_sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in source_files})
    model.save(args.output, metadata=metadata)
    print(json.dumps(dict(status="complete", checkpoint=str(args.output),
                          training_rows=len(rows), final_training_mse=model.history[-1]["mse"]),
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
