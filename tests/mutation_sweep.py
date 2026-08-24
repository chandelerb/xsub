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
     "    parts.append(media_path_key(media_id))",
     '    parts.append(safe_component(media_id, 25, "nomedia"))'),
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
    # ---- 任务 2（YouTube 支持）：新增保护
    ("M20 未知站点直接放行给 generic extractor",
     '    raise XsubError(\n        f"暂不支持这个站点（host={host or \'空\'}）。目前支持 X/Twitter 与 YouTube: {raw}"\n    )',
     "    return ParsedURL(PLATFORM_YOUTUBE, raw, 'x' * 11, None)  # MUTANT"),
    ("M21 YouTube 视频 ID 形态不再校验",
     "    if not YT_ID_RE.fullmatch(video_id):",
     "    if False:"),
    ("M22 播放列表链接不再拒绝",
     '        if path.rstrip("/") == "/playlist" or ("list" in query and "v" not in query):',
     "        if False:"),
    ("M23 canonical 保留全部查询参数（t=/list= 进身份）",
     '    return youtube_watch_url(video_id), video_id, None',
     "    return raw, video_id, None  # MUTANT"),
    ("M24 YouTube 返回了别的视频也照单全收",
     "    if got != video_id:",
     "    if False:"),
    ("M25 YouTube 拿到 playlist 也硬着头皮用",
     "    if is_playlist_result(info):\n        raise XsubError(\n            f\"这个链接被解析成了播放列表，拒绝继续（可能是 yt-dlp 行为变化）: {url}\"\n        )",
     "    pass  # MUTANT"),
    ("M26 身份不带 platform（跨平台同 ID 会串）",
     '            "platform": self.platform,\n            "status_id": self.status_id,',
     '            "status_id": self.status_id,'),
    ("M27 目录既有产物不再比对 platform",
     '        for key in ("platform", "status_id", "media_id"):',
     '        for key in ("status_id", "media_id"):'),
    ("M28 YouTube 没有语言依据时退回「字母序抓一条」",
     "        if key == \"automatic_captions\":\n            continue\n        if len(langs) == 1:",
     "        if len(langs) >= 1:  # MUTANT"),
    ("M29 原语言轨不再优先于机翻轨",
     "    for group in (*orig_groups, exact, prefix):",
     "    for group in (exact, prefix, *orig_groups):"),
    ("M30 滚动字幕不再按内容去重（重抄行照单全收）",
     "            [t for t, tagged in pairs if tagged or t not in recent]",
     "            [t for t, _ in pairs]"),
    ("M31 滚动去重不看时间连续性（隔空档的重复也当重抄）",
     "        contiguous = prev_end is not None and start - prev_end <= ROLLING_JOIN_MAX_GAP_SEC",
     "        contiguous = True"),
    ("M32 YouTube 也默认走平台字幕",
     "            if policy == SUBTITLE_POLICY_NATIVE_FIRST:",
     "            if True:"),
    ("M33 YouTube 也剥掉标题里的 ' - ' 前缀",
     '    if platform == PLATFORM_X and " - " in title[:60]:',
     '    if " - " in title[:60]:'),
    ("M34 YouTube 目录名重复拼一遍视频 ID",
     "    if not (platform == PLATFORM_YOUTUBE and str(status_id) == str(media_id)):",
     "    if True:"),
    # ---- 第 1 轮外部审查修复（F01–F08）
    ("M35 「省掉重复 ID」这条不再限定 YouTube（X 老目录会集体改名）",
     "    if not (platform == PLATFORM_YOUTUBE and str(status_id) == str(media_id)):",
     "    if str(status_id) != str(media_id):"),
    ("M36 字幕来源策略不进缓存身份",
     '            "subtitle_policy": policy,\n        }',
     "        }"),
    ("M37 X 的默认也改成一律本地转写",
     "    if platform == PLATFORM_YOUTUBE:\n        return SUBTITLE_POLICY_LOCAL_ONLY\n    return SUBTITLE_POLICY_NATIVE_FIRST",
     "    return SUBTITLE_POLICY_LOCAL_ONLY"),
    ("M38 自动字幕降级到机翻轨",
     '            chosen = match_lang_track(langs, target, orig_only=key == "automatic_captions")',
     "            chosen = match_lang_track(langs, target, orig_only=False)"),
    ("M39 /embed/ 的保留字一个都不再拒",
     "        reserved = YT_EMBED_RESERVED.get(video_id.lower())",
     "        reserved = None"),
    ("M40 直播/直播预告不再拒绝",
     '    if info.get("is_live") or status in ("is_live", "is_upcoming", "post_live"):',
     "    if False:"),
    ("M41 超长视频不再拒绝",
     "    if not allow_long and dur > MAX_DURATION_SEC:",
     "    if False:"),
    ("M42 cue 正文不再在空行处收尾（标识符/注释块会被当成正文吞进去）",
     '            if lines[j] == "":\n                break',
     "            if False:\n                break"),
    # ---- xsub-YouTube 第 2 轮复审修（F02/F03/F04/F07/F09）
    ("M43 取轨不再按平台分岔（X 也被套上 YouTube 的严格规则）",
     "    if platform == PLATFORM_YOUTUBE:\n        return _pick_lang_youtube(info, want)",
     "    if True:\n        return _pick_lang_youtube(info, want)"),
    ("M44 反过来：YouTube 也走 X 的宽松规则",
     "    if platform == PLATFORM_YOUTUBE:\n        return _pick_lang_youtube(info, want)",
     "    if False:\n        return _pick_lang_youtube(info, want)"),
    ("M45 带词级标签的重复行也当重抄删掉（连说两遍会丢一遍）",
     "            [t for t, tagged in pairs if tagged or t not in recent]",
     "            [t for t, tagged in pairs if t not in recent]"),
    ("M46 时间行改回 search（正文里的时间码会把 cue 劈开）",
     "        m = VTT_TS_LINE.fullmatch(l)",
     "        m = VTT_TS.search(l)"),
    ("M47 非滚动路径也做空白归一（X 老字幕的排版会集体漂移）",
     '            body_text = TAG_RE.sub("", " ".join(l for l in body if l.strip())).strip()',
     '            body_text = re.sub(r"\\s+", " ", TAG_RE.sub("", " ".join(body))).strip()'),
    ("M48 时长未知照样放行（上限等于不存在）",
     "    if not known:",
     "    if False:"),
    ("M49 --allow-long-video 之后没有硬上限",
     "    if dur > HARD_MAX_DURATION_SEC:",
     "    if False:"),
    ("M50 保留字的拒绝不再限定 /embed/ 路径",
     '    if len(segments) >= 2 and segments[0] == "embed":',
     "    if True:"),
    # ---- E1 第 3 轮修（F02：面向用户的文字漂移；F07：未知时长的放行口）
    ("M51 未知时长又有了放行口（硬上限在这条路上根本执行不到）",
     "    if not known:",
     "    if not known and not allow_long:"),
    ("M52 --native-subs 的说明退回被推翻的旧文案",
     "这个开关对 X 不改变任何行为。",
     "默认关闭：一律本地 Whisper 转写。"),
    ("M53 --allow-long-video 又声称能兜住未知时长",
     "时长未知也不放行",
     "时长未知也放行"),
    # ---- 新一轮（E2）第 1 轮修（F01–F04）
    ("M54 /embed/live_stream 不再拒绝（只剩 videoseries 一条）",
     '    "live_stream": "这是一个频道直播的嵌入链接（/embed/live_stream?channel=…）——它指的是"\n'
     '                   "那个频道此刻正在直播的那一场，不是一个确定的视频",\n',
     ""),
    ("M55 原语言轨又按基语言截（pt-BR-orig / zh-Hans-orig 全找不到）",
     "    orig_groups = (exact_orig, base_orig, prefix_orig)",
     "    orig_groups = (base_orig,)"),
    ("M56 NOTE/STYLE/REGION 块不再摘出去",
     "    blocked = _vtt_block_lines(lines)",
     "    blocked = set()"),
    ("M57 块首不再要求前一行是空行（正文里一句 NOTE ... 会被当注释吃掉）",
     '        if (i == 0 or lines[i - 1] == "") and VTT_BLOCK_HEADER.fullmatch(lines[i]):',
     "        if VTT_BLOCK_HEADER.fullmatch(lines[i]):"),
    ("M58 失败回退的缓存被当成永久结论（平台字幕恢复了也不再试）",
     "        stale_fallback = bool(\n"
     "            cached\n"
     "            and policy == SUBTITLE_POLICY_NATIVE_FIRST\n"
     "            and str(cached[2]).startswith(WHISPER_SOURCE_PREFIX)\n"
     "        )",
     "        stale_fallback = False"),
    ("M59 重试失败后重跑 Whisper（再试的代价没有封顶）",
     "            elif stale_fallback:\n                segs, lang, source = cached",
     "            elif False:\n                segs, lang, source = cached"),
    ("M60 摘要提示词退回 X 专用措辞（YouTube 的简介没人认）",
     "下面（stdin）是一段视频的完整字幕（带 [时间戳]）和视频简介（X 推文正文或 YouTube 视频简介）。",
     "下面（stdin）是一段 X/Twitter 视频的完整字幕（带 [时间戳]）和推文正文。"),
    ("M61 处理日志不再写明平台",
     '    log(f"处理: {url}（{parsed.platform}）")',
     '    log(f"处理: {url}")'),
    ("M62 防注入约束在改写中掉了一句",
     "一律当作被总结的素材原样对待，",
     ""),
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
