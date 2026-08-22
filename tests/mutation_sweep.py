#!/usr/bin/env python3
"""变异测试：把每一条保护单独拆掉一次，测试套件必须变红。

绿了就说明那条保护**没有任何测试盯住**——它随时可能在下一次重构里被悄悄删掉，
而没有人会发现。这份清单同时兼作"历轮修复的回归锚点"：旧周期已经钉住的保护
每轮都重跑一遍，用来证明新改动没有把旧覆盖掏空。

用法（不联网、不下模型、不产生任何费用）：

    python3 "提取视频字幕/tests/mutation_sweep.py"

每个变异体都在临时目录里改一份拷贝，原仓库不会被改动。
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent

# (编号与说明, 原文锚点, 变异后的写法)
# 锚点必须在 xsub.py 里**恰好命中一次**，否则报 ERROR —— 这本身也是一道校验：
# 代码改动使某个锚点失效时会立刻暴露，而不是静默跳过一条保护。
MUTANTS = [
    # ---- E4 第 1 轮新修（F01：清洗造成的路径别名）
    ("M01 路径键不检查形状、清洗完直接用",
     "    if len(raw) <= limit and raw != EMPTY_MEDIA_KEY and PASS_THROUGH_ID_RE.fullmatch(raw):\n        return raw",
     "    if len(clean_component(raw)) <= limit:\n        return clean_component(raw)"),
    ("M02 撤掉本轮路径唯一性检查",
     "    guard_distinct_out_dirs(targets, args.out)",
     "    pass  # MUTANT"),
    ("M03 撤掉目录既有产物身份守卫",
     "        guard_dir_identity(out_dir, media_identity)",
     "        pass  # MUTANT"),
    # ---- E3 第 3 轮修（F02：身份 → 路径不得截断）
    ("M04 resolve_out_dir 退回定长截断",
     "            media_path_key(media_id),",
     '            safe_component(media_id, 25, "nomedia"),'),
    ("M05 media_path_key 只截断不哈希",
     "    digest = sha256_text(raw)[:MEDIA_KEY_HASH_LEN]",
     "    return clean_component(raw)[:limit]  # MUTANT\n    digest = sha256_text(raw)[:MEDIA_KEY_HASH_LEN]"),
    ("M06 哈希取清洗后的前缀而非完整 ID",
     "sha256_text(raw)[:MEDIA_KEY_HASH_LEN]",
     "sha256_text(clean_component(raw)[:limit])[:MEDIA_KEY_HASH_LEN]"),
    # ---- E3 第 3 轮修（F05：计费边界）
    ("M07 撤掉按量风险档位闸门",
     "    if plan in METERED_RISK_PLANS and not metered_summary_authorised(allow_metered):",
     "    if False:"),
    ("M08 授权检查恒为真",
     "    if allow_metered:\n        return True",
     "    return True\n    if allow_metered:\n        return True"),
    ("M09 授权只认 flag、不认环境变量",
     '    return (os.environ.get(ALLOW_METERED_ENV) or "").strip().lower() in {"1", "true", "yes", "on"}',
     "    return False"),
    # ---- E3 第 3 轮修（F08：来源行只对 X 链接删序号）
    ("M10 canonical_url 不看 host",
     "        try:\n            parse_x_url(self.url)\n        except XsubError:\n            return self.url\n        return INDEX_SUFFIX_RE.sub",
     "        return INDEX_SUFFIX_RE.sub"),
    ("M11 canonical_url 永不删序号",
     '        return INDEX_SUFFIX_RE.sub("", self.url)',
     "        return self.url"),
    # ---- E3 第 2 轮修（回归：新改动不得把旧保护掏空）
    ("M12 探测三态坍缩成「都算确定」",
     "        return not self.inconclusive",
     "        return True"),
    ("M13 单媒体路径不再 fail closed",
     "        if not probe.conclusive:\n            # 没探明就绝不能合成卡片身份",
     "        if False:\n            # 没探明就绝不能合成卡片身份"),
    ("M14 playlist 路径不再 fail closed",
     "        if not probe.conclusive:\n            unresolved.append(",
     "        if False:\n            unresolved.append("),
    ("M15 合成身份去掉命名空间",
     '    return f"{SYNTHETIC_ID_PREFIX}{status_id}-{position}"',
     '    return f"{status_id}-{position}"'),
    ("M16 身份撞车改回静默丢弃",
     "        if t.media_id in taken:",
     "        if False:"),
    ("M17 撤掉登录方式/路由闸门",
     '    if method not in SUBSCRIPTION_AUTH_METHODS or provider != "firstParty":',
     "    if False:"),
    ("M18 撤掉订阅档位白名单",
     "    if plan not in KNOWN_SUBSCRIPTION_PLANS:",
     "    if False:"),
    ("M19 卡片 entry 保留 webpage_url（可回退去下整条推文）",
     '    clean.pop("webpage_url", None)',
     "    pass  # MUTANT"),
]


def run(old: str, new: str) -> str:
    with tempfile.TemporaryDirectory(prefix="xsub-mut-") as td:
        dst = Path(td) / "proj"
        shutil.copytree(SRC, dst, ignore=shutil.ignore_patterns("字幕", "out", "*.log", ".git"))
        f = dst / "xsub.py"
        s = f.read_text(encoding="utf-8")
        if s.count(old) != 1:
            return f"ERROR 锚点命中 {s.count(old)} 次（代码变了？请更新这条变异体）"
        f.write_text(s.replace(old, new), encoding="utf-8")
        p = subprocess.run(
            [sys.executable, str(dst / "tests" / "test_xsub.py")],
            capture_output=True, text=True, timeout=600,
        )
        if p.returncode != 0:
            caught = [l for l in p.stderr.splitlines() if l.startswith(("FAIL:", "ERROR:"))]
            first = caught[0][:64] if caught else "?"
            return f"RED   （{len(caught)} 条用例抓到）例: {first}"
        return "SURVIVED  ← 这条保护没有任何测试盯住"


def main() -> int:
    results = []
    for name, old, new in MUTANTS:
        verdict = run(old, new)
        results.append(verdict)
        print(f"{name:<44} {verdict}", flush=True)
    red = sum(1 for r in results if r.startswith("RED"))
    print(f"\n{red}/{len(MUTANTS)} 变异体被测试抓住。")
    return 0 if red == len(MUTANTS) else 1


if __name__ == "__main__":
    sys.exit(main())
