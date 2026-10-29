"""仅重绘已核验的 A/F 汇总结果，不训练、不推理、不重新计算统计量。"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def render(results, output, zh_font):
    verification = json.loads((results / "verification.json").read_text())
    assert verification["status"] == "passed"
    assert verification["conditions"] == 14688
    assert verification["old_joint_correction_used"] is False
    for name in ("summary.json", "comparisons.json"):
        actual = hashlib.sha256((results / name).read_bytes()).hexdigest()
        assert actual == verification["files_sha256"][name], name
    rows = json.loads((results / "summary.json").read_text())
    comparisons = json.loads((results / "comparisons.json").read_text())
    ber = {r["id"]: r for r in comparisons if r["metric"] == "ber"}
    keys = ["A_without_F", "A_with_F", "F_without_A", "F_with_A", "AF_vs_fixed_scan64"]
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42})
    for language in ("en", "zh"):
        # CFF 中文 OTF 使用 Type 3 轮廓嵌入，避免 XeLaTeX 合并时的 Type 42 字形映射错误。
        plt.rcParams["pdf.fonttype"] = 3 if language == "zh" else 42
        fp = FontProperties(fname=str(zh_font), size=11) if language == "zh" else FontProperties(size=11)
        cn = language == "zh"
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.4, 7.1))
        values = [r["ber"] * 100 for r in rows]
        lower = [r["ci95"]["ber"][0] * 100 for r in rows]
        upper = [r["ci95"]["ber"][1] * 100 for r in rows]
        errors = [[v-lo for v,lo in zip(values,lower)], [hi-v for v,hi in zip(values,upper)]]
        top.bar(range(5), values, yerr=errors, capsize=4, width=.6,
                color=["#999999", "#4477AA", "#CCBB44", "#228833", "#AA3377"])
        labels = (["传统估计\n16 次", "A\n16 次", "传统估计+F\n64 次", "A+F\n64 次", "固定扫描\n64 次"]
                  if cn else ["Traditional\n16 probes", "A\n16 probes", "Traditional+F\n64 probes", "A+F\n64 probes", "Fixed scan\n64 probes"])
        top.set_xticks(range(5), labels, fontproperties=fp)
        for i, value in enumerate(values):
            top.text(i, upper[i]+.17, f"{value:.4f}", ha="center", fontsize=10)
        top.set_ylim(0, max(upper)+1.9)
        top.set_ylabel("BER (%)", fontproperties=fp)
        top.set_title("(a) 接收质量与 95% 区间" if cn else "(a) Receive quality and 95% intervals", fontproperties=fp)
        top.set_axisbelow(True)
        top.grid(axis="y", alpha=.2)
        labels = (["加入 A，无 F（16 次）", "加入 A，有 F（64 次）",
                   "加入 F，无 A（16→64 次）", "加入 F，有 A（16→64 次）",
                   "A+F 对比固定扫描（64 次）"] if cn else
                  ["Add A; no F (16 probes)", "Add A; with F (64 probes)",
                   "Add F; no A (16 to 64)", "Add F; with A (16 to 64)",
                   "A+F vs. fixed scan (64)"])
        for i, key in enumerate(keys):
            r = ber[key]
            gain, lo, hi = -100*r["estimate"], -100*r["ci95"][1], -100*r["ci95"][0]
            bottom.errorbar(gain, i, xerr=[[gain-lo], [hi-gain]], fmt="o", capsize=4, color="#4477AA")
        bottom.set_yticks(range(5), labels, fontproperties=fp)
        bottom.invert_yaxis()
        bottom.axvline(0, color="black", linewidth=.8, linestyle="--")
        bottom.grid(axis="x", alpha=.2)
        bottom.set_xlabel("BER 降低值（百分点）" if cn else "BER reduction (percentage points)", fontproperties=fp)
        bottom.set_title("(b) 配对效应与 95% 区间" if cn else "(b) Paired effects and 95% intervals", fontproperties=fp)
        for ax in (top, bottom):
            ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(h_pad=2.4)
        fig.savefig(output / f"rq2_af_ablation_{language}.pdf")
        fig.savefig(output / f"rq2_af_ablation_{language}.png", dpi=150)
        plt.close(fig)
    provenance = {
        "operation": "layout-and-labels-only",
        "statistical_values_recomputed": False,
        "source_verification_sha256": hashlib.sha256((results / "verification.json").read_bytes()).hexdigest(),
        "source_files": {name: verification["files_sha256"][name] for name in ("summary.json", "comparisons.json")},
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("rq2_af_ablation_*.*")},
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zh-font", type=Path, required=True)
    args = parser.parse_args()
    render(args.results, args.output, args.zh_font)
