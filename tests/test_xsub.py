#!/usr/bin/env python3
"""xsub 回归测试。

每个 test class 对应一条评审 finding（F01–F07），命名里带编号，方便复审时逐条核对。
只用标准库；xsub.py 把 yt_dlp / mlx_whisper 放在函数内部 import，所以这里导入模块本身
不需要装任何第三方依赖。

跑法：  python3 tests/test_xsub.py            （或 python3 -m unittest discover tests）
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import re
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("xsub", ROOT / "xsub.py")
xsub = importlib.util.module_from_spec(spec)
sys.modules["xsub"] = xsub
spec.loader.exec_module(xsub)


class Args:
    """process() 用到的参数集合的最小替身。"""

    def __init__(self, out: Path, **kw):
        self.out = out
        self.model = xsub.DEFAULT_MODEL
        self.lang = None
        self.force = False
        self.cookies = False
        self.no_summary = True
        self.native_subs = False  # 与生产默认一致（X 走平台字幕优先，YouTube 走本地转写）
        self.open = False
        self.allow_long_video = False
        for k, v in kw.items():
            setattr(self, k, v)


# --------------------------------------------------------------- F01 时间码
class TestF01Timestamps(unittest.TestCase):
    def test_millisecond_rollover_never_emits_1000(self):
        """59.9996s 旧版会输出非法的 00:00:59,1000，必须进位成 00:01:00,000。"""
        self.assertEqual(xsub.fmt_ts(59.9996, srt=True), "00:01:00,000")
        self.assertEqual(xsub.fmt_ts(59.9996), "01:00")

    def test_hour_and_minute_rollover(self):
        self.assertEqual(xsub.fmt_ts(3599.9999, srt=True), "01:00:00,000")
        self.assertEqual(xsub.fmt_ts(3599.9999), "1:00:00")

    def test_no_srt_timestamp_has_ms_out_of_range(self):
        """全量扫一遍边界值，毫秒位必须恒 < 1000、秒/分 < 60。"""
        vals = [i / 10000 for i in range(0, 40000)] + [59.9994, 59.9995, 119.99951, 7199.9999]
        for v in vals:
            ts = xsub.fmt_ts(v, srt=True)
            h, m, rest = ts.split(":")
            s, ms = rest.split(",")
            self.assertLess(int(ms), 1000, f"{v} -> {ts}")
            self.assertLess(int(s), 60, f"{v} -> {ts}")
            self.assertLess(int(m), 60, f"{v} -> {ts}")

    def test_negative_clamped(self):
        self.assertEqual(xsub.fmt_ts(-5, srt=True), "00:00:00,000")

    def test_srt_render_roundtrip(self):
        srt = xsub.render_srt([{"start": 59.9996, "end": 60.4, "text": "hi"}])
        self.assertIn("00:01:00,000 --> 00:01:00,400", srt)


# ------------------------------------------------- F02 claude 子进程隔离/认证
class TestF02SummarySubprocessIsolation(unittest.TestCase):
    def test_argv_disables_tools_mcp_settings_and_session(self):
        argv = xsub.summary_argv("/usr/bin/claude")
        pairs = list(zip(argv, argv[1:]))
        self.assertIn(("--tools", ""), pairs)
        self.assertIn(("--mcp-config", '{"mcpServers":{}}'), pairs)
        self.assertIn(("--setting-sources", ""), pairs)
        self.assertIn(("--max-turns", "1"), pairs)
        for flag in ("--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"):
            self.assertIn(flag, argv)

    def test_argv_does_not_use_bare_mode(self):
        """--bare 会强制走 ANTHROPIC_API_KEY/apiKeyHelper 认证，与"不用 API key"冲突。"""
        self.assertNotIn("--bare", xsub.summary_argv("/usr/bin/claude"))

    def test_system_prompt_frames_transcript_as_untrusted(self):
        argv = xsub.summary_argv("/usr/bin/claude")
        sysprompt = argv[argv.index("--system-prompt") + 1]
        self.assertIn("不可信", sysprompt)
        self.assertIn("绝不执行", sysprompt)

    def test_env_drops_api_key_and_gateway_overrides(self):
        dirty = {
            "PATH": "/usr/bin",
            "HOME": "/Users/x",
            "ANTHROPIC_API_KEY": "sk-should-not-leak",
            "ANTHROPIC_AUTH_TOKEN": "t",
            "ANTHROPIC_BASE_URL": "https://evil.example",
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "AWS_SECRET_ACCESS_KEY": "s",
            "CLAUDECODE": "1",
            "LC_ALL": "en_US.UTF-8",
        }
        with mock.patch.dict(os.environ, dirty, clear=True):
            env = xsub.summary_env()
        for leaked in (
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_BASE_URL",
            "CLAUDE_CODE_USE_BEDROCK",
            "AWS_SECRET_ACCESS_KEY",
            "CLAUDECODE",
        ):
            self.assertNotIn(leaked, env)
        self.assertEqual(env["PATH"], "/usr/bin")
        self.assertEqual(env["LC_ALL"], "en_US.UTF-8")

    def test_transcript_goes_through_stdin_not_argv(self):
        """字幕内容不得出现在命令行参数里（会进 ps / shell history）。"""
        argv = xsub.summary_argv("/usr/bin/claude")
        self.assertFalse(any("SECRET-TRANSCRIPT" in a for a in argv))

    def test_summarize_failure_does_not_touch_transcript(self):
        with TempDir() as d:
            md = d / "transcript.md"
            md.write_text("hello", encoding="utf-8")
            fake = mock.Mock(returncode=1, stdout=b"", stderr=b"boom")
            with mock.patch.object(xsub.shutil, "which", return_value="/usr/bin/claude"), mock.patch.object(
                xsub.subprocess, "run", return_value=fake
            ), mock.patch.object(
                xsub, "claude_auth_kind", return_value=(True, "authMethod=claude.ai")
            ):
                state = xsub.summarize(md, d / "summary.md", d / "summary.meta.json")
            self.assertEqual(state, "failed")
            self.assertEqual(md.read_text(encoding="utf-8"), "hello")
            self.assertFalse((d / "summary.md").exists())

    def test_summarize_timeout_and_oserror_are_caught(self):
        with TempDir() as d:
            md = d / "transcript.md"
            md.write_text("hello", encoding="utf-8")
            for exc in (
                xsub.subprocess.TimeoutExpired(cmd="claude", timeout=600),
                OSError("no such file"),
            ):
                with mock.patch.object(
                    xsub.shutil, "which", return_value="/usr/bin/claude"
                ), mock.patch.object(xsub.subprocess, "run", side_effect=exc), mock.patch.object(
                xsub, "claude_auth_kind", return_value=(True, "authMethod=claude.ai")
            ):
                    self.assertEqual(
                        xsub.summarize(md, d / "summary.md", d / "summary.meta.json"), "failed"
                    )

    def test_summarize_success_binds_meta_and_runs_in_temp_cwd(self):
        with TempDir() as d:
            md = d / "transcript.md"
            md.write_text("hello", encoding="utf-8")
            seen = {}

            def fake_run(argv, **kw):
                seen.update(kw)
                return mock.Mock(returncode=0, stdout=b"# summary", stderr=b"")

            with mock.patch.object(xsub.shutil, "which", return_value="/usr/bin/claude"), mock.patch.object(
                xsub.subprocess, "run", side_effect=fake_run
            ), mock.patch.object(
                xsub, "claude_auth_kind", return_value=(True, "authMethod=claude.ai")
            ):
                state = xsub.summarize(md, d / "summary.md", d / "summary.meta.json")
            self.assertEqual(state, "generated")
            self.assertEqual(seen["input"], b"hello")
            self.assertNotEqual(Path(seen["cwd"]).resolve(), d.resolve())
            meta = json.loads((d / "summary.meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["transcript_sha256"], xsub.sha256_text("hello"))

    def test_summarize_skipped_when_claude_missing(self):
        with TempDir() as d:
            md = d / "transcript.md"
            md.write_text("hello", encoding="utf-8")
            with mock.patch.object(xsub.shutil, "which", return_value=None):
                self.assertEqual(
                    xsub.summarize(md, d / "summary.md", d / "summary.meta.json"), "skipped"
                )


# --------------------------------------------------- F03 批处理不被 sys.exit 掐断
class TestF03BatchContinuesAndExitCode(unittest.TestCase):
    def test_one_bad_url_does_not_abort_the_rest(self):
        calls = []

        def fake_process(u, args):
            calls.append(u)
            if "bad" in u:
                raise xsub.XsubError("模拟失败")
            return [xsub.RunResult(Path("/tmp/xsub-fake"))]

        with TempDir() as d, mock.patch.object(xsub, "process", side_effect=fake_process):
            code = xsub.main(
                ["https://x.com/a/status/1bad", "https://x.com/a/status/2", "--out", str(d), "--no-summary"]
            )
        self.assertEqual(len(calls), 2, "第一条失败后必须继续处理第二条")
        self.assertEqual(code, 1)

    def test_partial_failure_exits_nonzero(self):
        """旧版 `1 if failed and not done else 0`：只要有一条成功就返回 0，会骗过自动化脚本。"""

        def fake_process(u, args):
            if "bad" in u:
                raise xsub.XsubError("模拟失败")
            return [xsub.RunResult(Path("/tmp/xsub-fake"))]

        with TempDir() as d, mock.patch.object(xsub, "process", side_effect=fake_process):
            code = xsub.main(
                ["https://x.com/a/status/1", "https://x.com/a/status/2bad", "--out", str(d), "--no-summary"]
            )
        self.assertEqual(code, 1)

    def test_all_success_exits_zero(self):
        with TempDir() as d, mock.patch.object(xsub, "process", return_value=[xsub.RunResult(d)]):
            self.assertEqual(xsub.main(["https://x.com/a/status/1", "--out", str(d), "--no-summary"]), 0)

    def test_no_sys_exit_inside_library_code(self):
        """除 __main__ 出口外，模块内不得再有 sys.exit() 调用（按 AST 判，不数注释）。"""
        import ast

        tree = ast.parse((ROOT / "xsub.py").read_text(encoding="utf-8"))
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "exit"
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "sys"
        ]
        self.assertEqual(len(calls), 1, "sys.exit 只允许出现在 __main__ 入口")
        # 且必须在 `if __name__ == "__main__":` 里，而不是任何函数体内
        in_func = any(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(c in ast.walk(n) for c in calls)
            for n in ast.walk(tree)
        )
        self.assertFalse(in_func, "sys.exit 不得出现在任何函数体内")

    def test_keyboard_interrupt_returns_130(self):
        with TempDir() as d, mock.patch.object(xsub, "process", side_effect=KeyboardInterrupt):
            self.assertEqual(xsub.main(["https://x.com/a/status/1", "--out", str(d)]), 130)


# --------------------------------------------------------- F04 缓存身份与失效
AUDIO_IDENTITY = {"schema": xsub.CACHE_SCHEMA, "status_id": "123", "media_id": "m1"}


class TestF04CacheIdentity(unittest.TestCase):
    IDENTITY = {
        "schema": xsub.CACHE_SCHEMA,
        "status_id": "123",
        "model": "mlx-community/whisper-large-v3-turbo",
        "lang_request": None,
    }

    def _write(self, p: Path, payload):
        p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def test_matching_cache_is_used(self):
        with TempDir() as d:
            self._write(
                d / "segments.json",
                {**self.IDENTITY, "language": "en", "source": "x", "segments": [{"start": 0, "end": 1, "text": "hi"}]},
            )
            got = xsub.load_segments_cache(d / "segments.json", self.IDENTITY)
            self.assertIsNotNone(got)
            self.assertEqual(got[1], "en")

    def test_cache_from_a_different_tweet_is_rejected(self):
        with TempDir() as d:
            self._write(
                d / "segments.json",
                {**self.IDENTITY, "status_id": "999", "segments": [{"start": 0, "end": 1, "text": "hi"}]},
            )
            self.assertIsNone(xsub.load_segments_cache(d / "segments.json", self.IDENTITY))

    def test_cache_from_a_different_model_or_lang_is_rejected(self):
        with TempDir() as d:
            for override in ({"model": "tiny"}, {"lang_request": "zh"}, {"schema": 0}):
                self._write(
                    d / "segments.json",
                    {**self.IDENTITY, **override, "segments": [{"start": 0, "end": 1, "text": "hi"}]},
                )
                self.assertIsNone(
                    xsub.load_segments_cache(d / "segments.json", self.IDENTITY), f"{override} 应判为 miss"
                )

    def test_corrupt_cache_is_a_miss_not_a_crash(self):
        with TempDir() as d:
            p = d / "segments.json"
            for junk in ("{not json", "[]", "null", json.dumps({**IDENT(self), "segments": "nope"})):
                p.write_text(junk, encoding="utf-8")
                self.assertIsNone(xsub.load_segments_cache(p, self.IDENTITY))

    def test_segment_with_missing_keys_is_a_miss(self):
        with TempDir() as d:
            self._write(d / "segments.json", {**self.IDENTITY, "segments": [{"start": 0}]})
            self.assertIsNone(xsub.load_segments_cache(d / "segments.json", self.IDENTITY))

    def test_audio_manifest_rejects_size_mismatch(self):
        with TempDir() as d:
            (d / "audio.m4a").write_bytes(b"1234")
            self._write(
                d / "audio.json",
                {**AUDIO_IDENTITY, "file": "audio.m4a", "size": 999},
            )
            retry = xsub.CookieRetry(False)
            with mock.patch.object(xsub.shutil, "which", return_value=None):
                # size 对不上 → 不走缓存 → 需要 ffmpeg → 抛 XsubError（证明没吃到脏缓存）
                with self.assertRaises(xsub.XsubError):
                    xsub.download_audio("https://x.com/a/status/123", d, AUDIO_IDENTITY, retry)

    def test_audio_manifest_hit_avoids_download(self):
        with TempDir() as d:
            (d / "audio.m4a").write_bytes(b"1234")
            self._write(
                d / "audio.json",
                {**AUDIO_IDENTITY, "file": "audio.m4a", "size": 4},
            )
            got = xsub.download_audio("https://x.com/a/status/123", d, AUDIO_IDENTITY, xsub.CookieRetry(False))
            self.assertEqual(got.name, "audio.m4a")

    def test_ffmpeg_is_not_required_when_audio_is_not_needed(self):
        """旧版在 main() 开头就检查 ffmpeg，缓存命中/平台字幕路径也被无谓拦下。"""
        src = (ROOT / "xsub.py").read_text(encoding="utf-8")
        head = src.split("def download_audio")[0]
        self.assertNotIn('which("ffmpeg")', head, "ffmpeg 检查必须在真正需要下载音频时才做")

    def test_force_purges_audio_native_and_segments(self):
        with TempDir() as root:
            info = {"id": "m1", "display_id": "123", "upload_date": "20260101", "uploader_id": "bob", "title": "t"}
            out_dir = xsub.resolve_out_dir(root, info, "123", "m1")
            out_dir.mkdir(parents=True)
            for name in ("audio.m4a", "audio.json", "native.en.vtt", "segments.json"):
                (out_dir / name).write_text("stale", encoding="utf-8")
            args = Args(root, force=True, native_subs=True)
            with mock.patch.object(xsub, "parse_x_url", return_value=("https://x.com/bob/status/123", "123", None)), \
                mock.patch.object(xsub, "fetch_info", return_value=info), \
                mock.patch.object(xsub, "fetch_native_subs", return_value=([{"start": 0, "end": 1, "text": "hi"}], "en", "平台字幕:en")):
                res = xsub.process("https://x.com/bob/status/123", args)[0]
            self.assertEqual(res.artifacts, ["transcript.md", "transcript.srt"])
            self.assertEqual(res.summary, "not_requested")
            for name in ("audio.m4a", "audio.json", "native.en.vtt"):
                self.assertFalse((out_dir / name).exists(), f"--force 应删除 {name}")
            cache = json.loads((out_dir / "segments.json").read_text(encoding="utf-8"))
            self.assertEqual(cache["status_id"], "123")
            self.assertEqual(cache["segments"][0]["text"], "hi")

    def test_writes_are_atomic(self):
        """写失败不得留下半截文件。"""
        with TempDir() as d:
            target = d / "x.md"
            target.write_text("old", encoding="utf-8")
            with mock.patch.object(xsub.os, "replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    xsub.write_atomic(target, "new")
            self.assertEqual(target.read_text(encoding="utf-8"), "old")
            self.assertEqual(list(d.glob(".x.md.*")), [], "临时文件必须清理干净")


def IDENT(case):
    return case.IDENTITY


# ------------------------------------------------------ F05 登录判定的误判
class TestF05AuthDetection(unittest.TestCase):
    def test_webpage_is_not_treated_as_age_restriction(self):
        """旧版裸子串 "age" 命中 "webpage"，普通网络错误会白读一次 Chrome Cookie。"""
        for msg in (
            "ERROR: Unable to download webpage: <urlopen error timed out>",
            "Unable to download API page",
            "The page could not be loaded",
            "Failed to parse JSON message",
        ):
            self.assertFalse(xsub.looks_like_auth(Exception(msg)), msg)

    def test_real_auth_errors_are_detected(self):
        for msg in (
            "ERROR: HTTP Error 401: Unauthorized",
            "HTTP Error 403: Forbidden",
            "This tweet is from a private account. Log in to view.",
            "NSFW tweet requires authentication",
            "Age-restricted video",
            "Sign in to confirm you are not a bot",
            "This account is protected",
            "Use --cookies-from-browser to pass cookies",
        ):
            self.assertTrue(xsub.looks_like_auth(Exception(msg)), msg)

    def test_cookie_retry_happens_once_and_covers_the_stage(self):
        attempts = []

        def flaky(cookies):
            attempts.append(cookies)
            if not cookies:
                raise RuntimeError("HTTP Error 403: Forbidden")
            return "ok"

        retry = xsub.CookieRetry(False)
        self.assertEqual(retry.call(flaky, "下载音频"), "ok")
        self.assertEqual(attempts, [False, True])
        # 重试成功后状态粘住，后续阶段直接带 cookie，不再重复弹钥匙串
        self.assertTrue(retry.cookies)
        attempts.clear()
        retry.call(flaky, "下载字幕")
        self.assertEqual(attempts, [True])

    def test_retry_is_not_attempted_twice(self):
        attempts = []

        def always_403(cookies):
            attempts.append(cookies)
            raise RuntimeError("HTTP Error 403: Forbidden")

        retry = xsub.CookieRetry(False)
        with self.assertRaises(xsub.XsubError) as ctx:
            retry.call(always_403, "读取推文信息")
        self.assertEqual(attempts, [False, True])
        # 两次错误都要保留，便于诊断到底是没登录还是登录了也没权限
        self.assertIn("匿名", str(ctx.exception))
        self.assertIn("Chrome 登录态", str(ctx.exception))
        with self.assertRaises(RuntimeError):
            retry.call(always_403, "下载音频")

    def test_non_auth_error_is_not_retried(self):
        attempts = []

        def boom(cookies):
            attempts.append(cookies)
            raise RuntimeError("Unable to download webpage: connection reset")

        with self.assertRaises(RuntimeError):
            xsub.CookieRetry(False).call(boom, "读取推文信息")
        self.assertEqual(attempts, [False])


# ------------------------------------------- F06 URL 校验与输出路径不可逃逸
class TestF06UrlAndPathSafety(unittest.TestCase):
    def test_canonicalizes_twitter_and_strips_extras(self):
        for raw in (
            "http://www.x.com/gregisenberg/status/2090176521335959721?s=20&t=abc",
            "x.com/gregisenberg/status/2090176521335959721",
            "  https://mobile.twitter.com/gregisenberg/statuses/2090176521335959721  ",
        ):
            url, sid, _idx = xsub.parse_x_url(raw)
            self.assertEqual(url, "https://x.com/gregisenberg/status/2090176521335959721")
            self.assertEqual(sid, "2090176521335959721")

    def test_rejects_non_x_and_malformed_urls(self):
        for raw in (
            "https://evil.example/x.com/status/1",
            "https://x.com.evil.example/a/status/1",
            "https://youtube.com/watch?v=abc",
            "file:///etc/passwd",
            "https://x.com/gregisenberg",
            "https://x.com/a/status/notanumber",
            "",
            "   ",
        ):
            with self.assertRaises(xsub.XsubError, msg=raw):
                xsub.parse_x_url(raw)

    def test_metadata_cannot_escape_the_output_root(self):
        with TempDir() as root:
            hostile = {
                "upload_date": "../../../../etc",
                "uploader_id": "../../root",
                "title": "../../../../../../tmp/pwned",
            }
            out_dir = xsub.resolve_out_dir(root, hostile, "123", "m1")
            self.assertIn(root.resolve(), out_dir.parents)
            self.assertNotIn("..", out_dir.name)

    def test_slash_and_null_in_metadata_are_neutralised(self):
        self.assertNotIn("/", xsub.safe_component("a/b/c"))
        self.assertNotIn("\x00", xsub.safe_component("a\x00b"))
        self.assertEqual(xsub.safe_component(".."), "unknown")
        self.assertEqual(xsub.safe_component(None, fallback="nodate"), "nodate")

    def test_folder_includes_status_id_so_distinct_tweets_never_collide(self):
        with TempDir() as root:
            info = {"upload_date": "20260819", "uploader_id": "gregisenberg", "title": "same title"}
            a = xsub.resolve_out_dir(root, info, "111", "m1")
            b = xsub.resolve_out_dir(root, info, "222", "m1")
            self.assertNotEqual(a, b)
            self.assertIn("_111_", a.name)


# ------------------------------------------- F07 摘要与字幕的绑定 / VTT 解析
class TestF07SummaryFreshness(unittest.TestCase):
    def _setup(self, d: Path, transcript: str, summary: str | None, sha: str | None):
        (d / "transcript.md").write_text(transcript, encoding="utf-8")
        if summary is not None:
            (d / "summary.md").write_text(summary, encoding="utf-8")
        if sha is not None:
            (d / "summary.meta.json").write_text(
                json.dumps({"schema": xsub.CACHE_SCHEMA, "transcript_sha256": sha}), encoding="utf-8"
            )

    def test_no_summary_reports_none(self):
        with TempDir() as d:
            self._setup(d, "abc", None, None)
            self.assertIsNone(xsub.summary_state(d))

    def test_summary_matching_current_transcript_is_current(self):
        with TempDir() as d:
            self._setup(d, "abc", "s", xsub.sha256_text("abc"))
            self.assertEqual(xsub.summary_state(d), "current")

    def test_summary_from_a_previous_transcript_is_stale(self):
        """--no-summary 或摘要失败时，旧 summary.md 不得被列成本轮产物。"""
        with TempDir() as d:
            self._setup(d, "NEW transcript", "old summary", xsub.sha256_text("OLD transcript"))
            self.assertEqual(xsub.summary_state(d), "stale")

    def test_missing_or_corrupt_meta_is_stale(self):
        with TempDir() as d:
            self._setup(d, "abc", "s", None)
            self.assertEqual(xsub.summary_state(d), "stale")
            (d / "summary.meta.json").write_text("{broken", encoding="utf-8")
            self.assertEqual(xsub.summary_state(d), "stale")


class TestVttParsing(unittest.TestCase):
    def test_parses_timestamps_without_hour_component(self):
        """WebVTT 允许省略小时位；旧正则强制 hh:mm:ss，这类文件会解析出 0 段。"""
        vtt = "WEBVTT\n\n00:01.000 --> 00:04.500\nHello world\n\n01:02.250 --> 01:05.000\nSecond line\n"
        segs = xsub.parse_vtt(vtt)
        self.assertEqual(len(segs), 2)
        self.assertAlmostEqual(segs[0]["start"], 1.0)
        self.assertAlmostEqual(segs[0]["end"], 4.5)
        self.assertAlmostEqual(segs[1]["start"], 62.25)

    def test_parses_full_hour_timestamps_and_strips_tags(self):
        vtt = "WEBVTT\n\n1\n01:02:03.400 --> 01:02:05.000\n<v Greg>Hi <b>there</b>\n"
        segs = xsub.parse_vtt(vtt)
        self.assertEqual(len(segs), 1)
        self.assertAlmostEqual(segs[0]["start"], 3723.4)
        self.assertEqual(segs[0]["text"], "Hi there")

    def test_short_millisecond_field_is_padded(self):
        segs = xsub.parse_vtt("WEBVTT\n\n00:00.5 --> 00:01.25\nx\n")
        self.assertAlmostEqual(segs[0]["start"], 0.5)
        self.assertAlmostEqual(segs[0]["end"], 1.25)

    def test_native_lang_selection_prefers_manual_then_honours_lang_flag(self):
        info = {"subtitles": {"ja": [{}]}, "automatic_captions": {"en": [{}], "en-US": [{}]}}
        self.assertEqual(xsub.pick_native_lang(info, None), ("ja", "平台字幕"))
        self.assertEqual(xsub.pick_native_lang(info, "en"), ("en", "平台自动字幕"))
        self.assertIsNone(xsub.pick_native_lang(info, "fr"), "语言不匹配时必须退回本地转写")
        self.assertIsNone(xsub.pick_native_lang({"subtitles": {}}, None))

    def test_lang_selection_refuses_to_gamble_without_evidence(self):
        """YouTube 上没有任何语言依据时宁可退回本地转写，也不按字母序抓一条。

        automatic_captions 有 160 条轨，字母序第一条是 aa（阿法尔语机翻），
        一条英文视频会配上一份阿法尔语字幕还报成功。
        这条严格规则**只对 YouTube**；X 的取轨行为另有专门用例锁定。
        """
        yt = xsub.PLATFORM_YOUTUBE
        many = {"subtitles": {"zh": [{}], "en": [{}], "ar": [{}]}}
        self.assertIsNone(xsub.pick_native_lang(many, None, yt), "多条人工轨又说不出要哪条 → 不赌")
        auto_only = {"automatic_captions": {"aa": [{}], "en": [{}], "zh": [{}]}}
        self.assertIsNone(
            xsub.pick_native_lang(auto_only, None, yt), "自动字幕表没有依据时一律不选"
        )

        # 有语言依据就必须挑得出来，且多次调用结果稳定
        with_lang = {**many, "language": "zh"}
        for _ in range(3):
            self.assertEqual(xsub.pick_native_lang(with_lang, None, yt), ("zh", "平台字幕"))
        # 只有一条人工字幕轨：创作者上传的就这一份，不存在挑错的余地
        self.assertEqual(
            xsub.pick_native_lang({"subtitles": {"ar": [{}]}}, None, yt), ("ar", "平台字幕")
        )

    def test_original_language_track_beats_machine_translated_ones(self):
        """YouTube 的 `<语言>-orig` 是 ASR 原件，同名无后缀轨可能是机翻，必须优先原件。"""
        yt = xsub.PLATFORM_YOUTUBE
        info = {"language": "en", "automatic_captions": {"aa": [{}], "en": [{}], "en-orig": [{}], "ja": [{}]}}
        self.assertEqual(xsub.pick_native_lang(info, None, yt), ("en-orig", "平台自动字幕"))
        self.assertEqual(xsub.pick_native_lang(info, "en", yt), ("en-orig", "平台自动字幕"))

    def test_machine_translated_track_is_not_used_for_a_foreign_video(self):
        """日语视频里那条 `en` 是机翻。要 en 就只能给 en，绝不能把 ja-orig 当成 en。"""
        yt = xsub.PLATFORM_YOUTUBE
        info = {"language": "ja", "automatic_captions": {"en": [{}], "ja-orig": [{}]}}
        self.assertEqual(xsub.pick_native_lang(info, None, yt), ("ja-orig", "平台自动字幕"))
        self.assertEqual(xsub.pick_native_lang(info, "fr", yt), None)


class TestRunResultReporting(unittest.TestCase):
    """F03.5 / F03.7：最终清单只报本轮真实产出，不用 exists() 反推。"""

    def test_artifacts_come_from_this_run_only(self):
        with TempDir() as root:
            info = {"id": "m1", "display_id": "123", "upload_date": "20260101", "uploader_id": "bob", "title": "t"}
            out_dir = xsub.resolve_out_dir(root, info, "123", "m1")
            out_dir.mkdir(parents=True)
            (out_dir / "summary.md").write_text("上一轮留下的摘要", encoding="utf-8")
            args = Args(root, no_summary=True, native_subs=True)
            with mock.patch.object(xsub, "parse_x_url", return_value=("https://x.com/bob/status/123", "123", None)), \
                mock.patch.object(xsub, "fetch_info", return_value=info), \
                mock.patch.object(xsub, "fetch_native_subs", return_value=([{"start": 0, "end": 1, "text": "hi"}], "en", "平台字幕:en")):
                res = xsub.process("https://x.com/bob/status/123", args)[0]
            self.assertNotIn("summary.md", res.artifacts, "--no-summary 时旧摘要不得算作本轮产物")
            self.assertEqual(res.summary, "not_requested")

    def test_failed_summary_is_not_listed_as_artifact(self):
        with TempDir() as root:
            info = {"id": "m1", "display_id": "123", "upload_date": "20260101", "uploader_id": "bob", "title": "t"}
            args = Args(root, no_summary=False, native_subs=True)
            with mock.patch.object(xsub, "parse_x_url", return_value=("https://x.com/bob/status/123", "123", None)), \
                mock.patch.object(xsub, "fetch_info", return_value=info), \
                mock.patch.object(xsub, "fetch_native_subs", return_value=([{"start": 0, "end": 1, "text": "hi"}], "en", "平台字幕:en")), \
                mock.patch.object(xsub, "summarize", return_value="failed"):
                res = xsub.process("https://x.com/bob/status/123", args)[0]
            self.assertEqual(res.summary, "failed")
            self.assertNotIn("summary.md", res.artifacts)
            self.assertIn("transcript.md", res.artifacts)


class TestStructuredAuthStatus(unittest.TestCase):
    """F05.1：优先用结构化 HTTP 状态码，而不是猜字符串。"""

    def test_status_401_403_detected_through_exception_chain(self):
        class HttpErr(Exception):
            def __init__(self, status):
                super().__init__("something went wrong")
                self.status = status

        for code in (401, 403):
            try:
                try:
                    raise HttpErr(code)
                except HttpErr as inner:
                    raise RuntimeError("ERROR: unable to download") from inner
            except RuntimeError as e:
                self.assertTrue(xsub.looks_like_auth(e), code)

    def test_other_http_status_never_triggers_cookie_read(self):
        class HttpErr(Exception):
            def __init__(self, status):
                super().__init__("private page not found")  # 文本里带 private，但状态码是 404
                self.status = status

        self.assertFalse(xsub.looks_like_auth(HttpErr(404)))
        self.assertFalse(xsub.looks_like_auth(HttpErr(500)))


class TestVttCueSettings(unittest.TestCase):
    def test_cue_settings_after_timestamp_are_ignored(self):
        vtt = (
            "WEBVTT\n\n"
            "00:00:01.000 --> 00:00:03.000 align:start position:0%\n"
            "First\n\n"
            "00:04.000 --> 00:06.000 line:90% align:middle\n"
            "Second\n"
        )
        segs = xsub.parse_vtt(vtt)
        self.assertEqual([s["text"] for s in segs], ["First", "Second"])
        self.assertAlmostEqual(segs[1]["start"], 4.0)

    def test_arrow_with_varied_spacing(self):
        self.assertEqual(len(xsub.parse_vtt("WEBVTT\n\n00:01.000-->00:02.000\nx\n")), 1)


class TestDirLockIsCrossProcess(unittest.TestCase):
    """F01.6：同一输出目录必须跨进程互斥。"""

    def test_second_process_blocks_until_first_releases(self):
        import subprocess as sp
        import textwrap

        with TempDir() as d:
            holder = textwrap.dedent(f"""
                import importlib.util, sys, time
                from pathlib import Path
                spec = importlib.util.spec_from_file_location("xsub", r"{ROOT / 'xsub.py'}")
                m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
                with m.dir_lock(Path(r"{d}")):
                    print("LOCKED", flush=True)
                    time.sleep(1.5)
            """)
            proc = sp.Popen([sys.executable, "-c", holder], stdout=sp.PIPE, text=True)
            self.assertEqual(proc.stdout.readline().strip(), "LOCKED")
            t0 = time.monotonic()
            with xsub.dir_lock(d):  # 必须等到对方释放
                waited = time.monotonic() - t0
            proc.wait(timeout=10)
            self.assertGreater(waited, 0.5, "第二个进程应被阻塞，而不是直接拿到锁")


class TempDir:
    def __init__(self):
        import tempfile

        self._t = tempfile.TemporaryDirectory(prefix="xsub-test-")

    def __enter__(self) -> Path:
        return Path(self._t.name)

    def __exit__(self, *a):
        self._t.cleanup()


# ------------------------------------------- F01 媒体身份（同一条推文的多个视频）
class FakeTwitter:
    """按 yt-dlp TwitterIE 的真实语义模拟一条挂了 3 个媒体的推文。

    上游 twitter.py 的关键事实（本 fixture 就是照它写的）：
      - _VALID_URL 末尾有 (?:/(?:video|photo)/(?P<index>\d+))? ，序号是媒体选择器；
      - _yes_playlist(twid, index) 在没有 noplaylist=True 时**忽略**该序号，返回整条 playlist；
      - 有 noplaylist=True 时按 extended_entities.media[index-1] 精确选中，
        越界报 "Video #N is unavailable"，选到图片报 "Media #N is not a video"；
      - 单条媒体的 info["id"] 是 media_id，info["display_id"] 才是推文 ID。
    """

    STATUS = "555"
    BASE = f"https://x.com/multi/status/{STATUS}"
    # 序号 1 = 视频，2 = 图片，3 = 视频；照抄 X 允许图文视频混排的情况
    MEDIA = [("video", "m-aaa", "第一个视频"), ("photo", None, None), ("video", "m-bbb", "第二个视频")]
    # 卡片视频：上游 extract_from_card_info() 产出的 data，会被 {**info, **data} 合成进 entries，
    # 但**不在** extended_entities.media 里，因此 /video/<n> 永远定位不到它们。
    CARDS: list = []
    # 外链播放器地址 → 由别的提取器返回的结果
    EXTERNAL: dict = {}
    # 额外并进每个视频 entry 的字段（如 subtitles / language）。默认空 = 其余用例逐字不变。
    ENTRY_EXTRA: dict = {}

    def __init__(self):
        self.extract_calls = []
        self.download_calls = []
        self.info_file_downloads = []
        self.fail_anonymous = False
        self.fail_media: set = set()  # 注入"这个媒体下载失败"，用来逼出上游的回退行为

    # ---- 供测试注入 sys.modules["yt_dlp"] 的假模块
    def as_module(self):
        import types

        mod = types.ModuleType("yt_dlp")
        outer = self

        class DownloadError(Exception):
            pass

        class YoutubeDL:
            def __init__(self, opts):
                self.opts = opts

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def extract_info(self, url, download=False):
                return outer._extract(url, self.opts)

            def download(self, urls):
                for u in urls:
                    got = outer._extract(u, self.opts)  # 下载走同一条选择逻辑
                    outer.download_calls.append((u, dict(self.opts)))
                    # 上游按同一个 outtmpl 逐个 entry 落盘：一条 playlist 里的每个视频
                    # 都写到同一个文件名上互相覆盖 —— 这正是"字幕属于错误视频"的机理
                    for one in (got.get("entries") if got.get("entries") else [got]):
                        self._write_media(one)

            def download_with_info_file(self, path):
                """对应 yt-dlp --load-info-json：卡片视频没有 URL 可定位，只能喂 info 字典。

                失败时的回退**照抄上游 YoutubeDL.download_with_info_file**：
                info 里还留着 webpage_url 就改成下载 webpage_url（= 整条推文），
                没有才 raise。fixture 必须复刻这一段，去掉 webpage_url 的保护才检验得出来。
                """
                with open(path, encoding="utf-8") as f:
                    info = json.load(f)
                outer.info_file_downloads.append(info)
                try:
                    self._write_media(info)
                except outer.DownloadError:
                    webpage_url = info.get("webpage_url")
                    if webpage_url is None:
                        raise
                    self.download([webpage_url])

            @staticmethod
            def sanitize_info(info, *a, **kw):
                return dict(info)

            def _write_media(self, info):
                """按 outtmpl 落盘一个音频或一份 vtt，内容里带媒体标记，串了一眼看得出来。"""
                tag = str(info.get("xsub_tag") or info.get("id") or "?")
                if tag in outer.fail_media:
                    raise outer.DownloadError(f"HTTP Error 404: Not Found ({tag})")
                tmpl = self.opts.get("outtmpl")
                if not tmpl:
                    return
                if self.opts.get("writesubtitles") or self.opts.get("writeautomaticsub"):
                    lang = (self.opts.get("subtitleslangs") or ["en"])[0]
                    Path(tmpl.replace("%(ext)s", f"{lang}.vtt")).write_text(
                        f"WEBVTT\n\n00:00.000 --> 00:01.000\n字幕属于 {tag}\n", encoding="utf-8"
                    )
                else:
                    ext = str(info.get("xsub_ext") or "m4a")
                    Path(tmpl.replace("%(ext)s", ext)).write_bytes(f"audio:{tag}".encode())

        mod.DownloadError = DownloadError
        mod.YoutubeDL = YoutubeDL
        outer.DownloadError = DownloadError
        return mod

    # ---- 内部：模拟 extractor
    def _tweet_base(self):
        """推文级字段。上游 entries 是 {**info, **data, "display_id": twid} 合成的，
        所以没有自带 id 的卡片视频会**继承推文 ID**。"""
        return {
            "id": self.STATUS,
            "display_id": self.STATUS,
            "webpage_url": self.BASE,  # 上游 entry 真会带上它 —— 卡片 entry 的回退退路
            "title": "整条推文",
            "description": "推文正文",
            "uploader": "Multi",
            "uploader_id": "multi",
            "upload_date": "20260101",
        }

    def _card_entries(self):
        return [{**self._tweet_base(), **data, "display_id": self.STATUS} for data in self.CARDS]

    def _entry(self, media_id, title):
        return {
            "id": media_id,
            "display_id": self.STATUS,
            "webpage_url": self.BASE,
            "title": title,
            "description": f"{title} 的推文正文",
            "uploader": "Multi",
            "uploader_id": "multi",
            "upload_date": "20260101",
            "duration": 12.0,
            "formats": [{"url": f"https://video.example/{media_id}.m4a"}],
            **self.ENTRY_EXTRA,
        }

    def _extract(self, url, opts):
        self.extract_calls.append((url, dict(opts)))
        if self.fail_anonymous and not opts.get("cookiesfrombrowser"):
            raise self.DownloadError("HTTP Error 403: Forbidden")
        if url in self.EXTERNAL:  # 外链播放器：交给对应的提取器，不再是 TwitterIE
            return dict(self.EXTERNAL[url])
        if not url.startswith(self.BASE):  # 没人认领的外链：上游会直接报错
            raise self.DownloadError(f"Unsupported URL: {url}")
        m = re.search(r"/video/(\d+)$", url)
        index = int(m.group(1)) if m else None
        videos = [self._entry(mid, t) for kind, mid, t in self.MEDIA if kind == "video"]
        entries = videos + self._card_entries()
        # 这一行就是上游 _yes_playlist 的语义：没开 noplaylist 时 URL 里的序号会被忽略
        if index is None or not opts.get("noplaylist"):
            if len(entries) == 1:  # 上游：len(entries) == 1 时直接返回 entry 本身，不是 playlist
                return entries[0]
            return {"_type": "playlist", "id": self.STATUS, "title": "整条推文", "entries": entries}
        # 注意：只查 MEDIA（= extended_entities.media）。卡片视频不在这里，
        # 所以逐序号探测永远找不到它们 —— 这就是 F08 的根因。
        if not 1 <= index <= len(self.MEDIA):
            raise self.DownloadError(f"Video #{index} is unavailable")
        kind, mid, title = self.MEDIA[index - 1]
        if kind != "video":
            raise self.DownloadError(f"Media #{index} is not a video")
        return self._entry(mid, title)


class FakeTwitterCase(unittest.TestCase):
    """挂载假 yt_dlp，并把本地转写/平台字幕换成确定性替身。"""

    def setUp(self):
        self.tw = FakeTwitter()
        self._saved = sys.modules.get("yt_dlp")
        sys.modules["yt_dlp"] = self.tw.as_module()

    def tearDown(self):
        if self._saved is None:
            sys.modules.pop("yt_dlp", None)
        else:
            sys.modules["yt_dlp"] = self._saved

    @staticmethod
    def _subs_per_media(info, url, out_dir, want_lang, retry, download_info=None,
                        platform=xsub.PLATFORM_X):
        """平台字幕替身：内容里带 media_id，串了就一眼看得出来。"""
        return [{"start": 0.0, "end": 1.0, "text": f"字幕属于 {info['id']}"}], "en", "平台字幕:en"


class TestF08CardVideos(FakeTwitterCase):
    """F08：卡片视频（外链播放器 / amplify / unified_card）不在 extended_entities.media 里，
    逐个 /video/<n> 探测永远找不到它们。方案 A：以 playlist entry 清单为准。"""

    # 内嵌媒体流、没有自己的 id —— 合成后会继承推文 ID
    INLINE_CARD = {"formats": [{"url": "https://vmap.example/card.m4a"}], "duration": 9.0}
    PLAYER_URL = "https://player.example/watch?v=xyz"
    PLAYER_CARD = {"_type": "url", "url": PLAYER_URL}
    PLAYER_RESULT = {
        "id": "ext-999", "display_id": "ext-999", "title": "外站视频",
        "uploader": "Player", "upload_date": "20260102",
        "formats": [{"url": "https://player.example/a.m4a"}],
    }

    def _run(self, root, raw_url):
        args = Args(root, no_summary=True, native_subs=True)
        with mock.patch.object(xsub, "fetch_native_subs", side_effect=self._subs_per_media):
            return xsub.process(raw_url, args)

    def _run_downloading(self, root, raw_url):
        """不走平台字幕替身，逼真实走 download_audio → 才能验证卡片视频到底怎么下载。"""
        args = Args(root, no_summary=True)
        with mock.patch.object(xsub, "fetch_native_subs", return_value=None), \
             mock.patch.object(xsub.shutil, "which", return_value="/usr/bin/ffmpeg"), \
             mock.patch.object(
                 xsub, "transcribe",
                 side_effect=lambda a, m, l, v: ([{"start": 0.0, "end": 1.0, "text": "转写"}], "en"),
             ):
            return xsub.process(raw_url, args)

    def test_index_probing_alone_cannot_find_a_card_video(self):
        """先自证 fixture：卡片视频确实无法用 /video/<n> 拿到，否则后面的测试是空转。"""
        self.tw.MEDIA = [("video", "m-aaa", "原生视频")]
        self.tw.CARDS = [self.INLINE_CARD]
        probed = xsub.probe_indexed_media(FakeTwitter.BASE, "555", 2, xsub.CookieRetry(False))
        self.assertEqual(set(probed.found), {"m-aaa"}, "序号探测本就只能找到原生附件视频")
        self.assertTrue(probed.conclusive, "这次探测每一步都得到了确定回答，不该有'没探明'")

    def test_card_only_tweet_is_still_processed(self):
        self.tw.MEDIA = []
        self.tw.CARDS = [self.INLINE_CARD]
        with TempDir() as root:
            results = self._run(root, FakeTwitter.BASE)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].media_id, "xsubcard-555-1", "卡片视频必须有自己的身份")

    def test_card_media_id_never_equals_the_status_id(self):
        """否则卡片视频会和推文级结果共用目录和缓存。"""
        self.tw.MEDIA = []
        self.tw.CARDS = [self.INLINE_CARD]
        targets = xsub.resolve_targets(FakeTwitter.BASE, "555", None, xsub.CookieRetry(False))
        self.assertNotEqual(targets[0].media_id, "555")
        self.assertEqual(targets[0].media_id, "xsubcard-555-1")
        # 整条推文只有一个视频时按链接下载本来就取不错，所以这里刻意**不判断**它是原生还是卡片；
        # source 记录的是"xsub 怎么定位到它的"，不是对上游分类的猜测。
        self.assertEqual(targets[0].source, "url")

    def test_native_plus_card_yields_one_output_each(self):
        self.tw.MEDIA = [("video", "m-aaa", "原生视频")]
        self.tw.CARDS = [self.INLINE_CARD]
        with TempDir() as root:
            results = self._run(root, FakeTwitter.BASE)
            self.assertEqual(len(results), 2, "原生视频和卡片视频都要处理")
            self.assertEqual({r.media_id for r in results}, {"m-aaa", "xsubcard-555-2"})
            self.assertEqual(len({r.out_dir for r in results}), 2, "两者不得共用目录")
            for r in results:
                cache = json.loads((r.out_dir / "segments.json").read_text(encoding="utf-8"))
                self.assertEqual(cache["media_id"], r.media_id, "manifest 与 media ID 必须一致")

    def test_external_player_card_is_reextracted_from_its_own_url(self):
        self.tw.MEDIA = []
        self.tw.CARDS = [self.PLAYER_CARD]
        self.tw.EXTERNAL = {self.PLAYER_URL: self.PLAYER_RESULT}
        targets = xsub.resolve_targets(FakeTwitter.BASE, "555", None, xsub.CookieRetry(False))
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].media_id, "ext-999", "外链卡片应拿到外站的真实媒体身份")
        self.assertEqual(targets[0].url, self.PLAYER_URL, "下载应指向外链，而不是推文链接")

    def test_unified_card_videos_keep_their_own_ids(self):
        """unified_card 走 extract_from_video_info，entry 自带 id，不该被合成身份覆盖。"""
        self.tw.MEDIA = []
        self.tw.CARDS = [
            {"id": "m-uni-1", "title": "统一卡片 1", "formats": [{"url": "https://u.example/1.m4a"}]},
            {"id": "m-uni-2", "title": "统一卡片 2", "formats": [{"url": "https://u.example/2.m4a"}]},
        ]
        targets = xsub.resolve_targets(FakeTwitter.BASE, "555", None, xsub.CookieRetry(False))
        self.assertEqual([t.media_id for t in targets], ["m-uni-1", "m-uni-2"])

    def test_card_video_downloads_through_the_info_dict(self):
        """同一条推文里既有原生又有卡片时，卡片没有任何 URL 能单独定位它。

        此时按推文 URL 下载会把整条推文的视频都下到同一个输出模板上，最后落盘的
        很可能是另一个视频，所以卡片必须走 --load-info-json 那条路。
        """
        self.tw.MEDIA = [("video", "m-aaa", "原生视频")]
        self.tw.CARDS = [self.INLINE_CARD]
        with TempDir() as root:
            results = self._run_downloading(root, FakeTwitter.BASE)
        self.assertEqual(len(results), 2)
        self.assertEqual(len(self.tw.info_file_downloads), 1, "卡片应走 download_with_info_file")
        fed = self.tw.info_file_downloads[0]
        self.assertEqual(fed["formats"], self.INLINE_CARD["formats"])
        self.assertNotIn("webpage_url", fed, "留着 webpage_url 上游会回退去下整条推文")
        self.assertEqual(
            [u for u, _ in self.tw.download_calls],
            [f"{FakeTwitter.BASE}/video/1"],
            "原生视频仍按带序号的链接下载，绝不能退回整条推文 URL",
        )

    def test_native_video_still_downloads_by_url(self):
        """方案 A 不能把原生视频也改成 info-dict 下载 —— 那条路已被真实链路验证过。"""
        self.tw.MEDIA = [("video", "m-aaa", "原生视频")]
        self.tw.CARDS = []
        with TempDir() as root:
            self._run_downloading(root, FakeTwitter.BASE)
        self.assertEqual(self.tw.info_file_downloads, [])
        self.assertEqual([u for u, _ in self.tw.download_calls], [FakeTwitter.BASE])

    def test_an_unresolvable_entry_fails_the_whole_input(self):
        """少处理了几个绝不能以退出码 0 的样子蒙混过去。"""
        self.tw.MEDIA = [("video", "m-aaa", "原生视频")]
        self.tw.CARDS = [{"title": "既没流也没外链的卡片"}]
        with TempDir() as root:
            with self.assertRaises(xsub.XsubError) as cm:
                self._run(root, FakeTwitter.BASE)
        self.assertIn("无法解析", str(cm.exception))

    def test_probed_media_missing_from_entries_is_not_dropped(self):
        """上游若改了 entry 的 id 形态，探测到的视频也不能被静默丢弃。"""
        self.tw.MEDIA = [("video", "m-aaa", "原生视频"), ("video", "m-bbb", "另一个")]
        self.tw.CARDS = [self.INLINE_CARD]
        real = xsub.playlist_entries

        def drop_first(info):
            return real(info)[1:]  # 假装 entry 清单里少了 m-aaa

        with mock.patch.object(xsub, "playlist_entries", side_effect=drop_first):
            targets = xsub.resolve_targets(FakeTwitter.BASE, "555", None, xsub.CookieRetry(False))
        self.assertIn("m-aaa", {t.media_id for t in targets}, "探测到却不在清单里的视频必须保留")


class TestE3EntryLevelIsolation(FakeTwitterCase):
    """xsub-E3 第 1 轮：卡片视频的音频与平台字幕必须**条目级**隔离。

    卡片视频的 target.url 是整条推文链接。只要下载还从这个 URL 走，yt-dlp 就会把
    推文里每个视频写进同一个 outtmpl 互相覆盖，最后落盘的很可能是别的视频，
    而 manifest 还按当前 media_id 记账 —— 这就是"字幕属于错误视频"。
    """

    CARD_A = {"xsub_tag": "card-A", "duration": 9.0,
              "formats": [{"url": "https://vmap.example/a.m4a"}]}
    CARD_B = {"xsub_tag": "card-B", "duration": 8.0,
              "formats": [{"url": "https://vmap.example/b.m4a"}]}
    SUBBED_A = {**CARD_A, "subtitles": {"en": [{"ext": "vtt"}]}}
    SUBBED_B = {**CARD_B, "subtitles": {"en": [{"ext": "vtt"}]}}
    LONE_PLAYER = {"_type": "url", "url": "https://nobody.example/watch?v=zzz"}

    def _run_audio(self, root):
        """真实走 download_audio，并让"转写结果"直接复述音频文件的内容。

        这样 transcript 里写的是哪个视频的音频，一眼可见；串了就红。
        """
        args = Args(root, no_summary=True)
        with mock.patch.object(xsub, "fetch_native_subs", return_value=None), \
             mock.patch.object(xsub.shutil, "which", return_value="/usr/bin/ffmpeg"), \
             mock.patch.object(
                 xsub, "transcribe",
                 side_effect=lambda a, m, l, v: (
                     [{"start": 0.0, "end": 1.0,
                       "text": a.read_text(encoding="utf-8", errors="replace")}], "en",
                 ),
             ):
            return xsub.process(FakeTwitter.BASE, args)

    def _run_native_subs(self, root, patch_subs=None):
        """真实走 fetch_native_subs（entry 自带 subtitles），转写路径应当用不上。"""
        args = Args(root, no_summary=True, native_subs=True)
        stack = [
            mock.patch.object(xsub.shutil, "which", return_value="/usr/bin/ffmpeg"),
            mock.patch.object(
                xsub, "transcribe",
                side_effect=lambda a, m, l, v: (_ for _ in ()).throw(
                    AssertionError("不该退回本地转写：平台字幕本就该下得到")
                ),
            ),
        ]
        if patch_subs is not None:
            stack.append(mock.patch.object(xsub, "fetch_native_subs", side_effect=patch_subs))
        with contextlib.ExitStack() as es:
            for cm in stack:
                es.enter_context(cm)
            return xsub.process(FakeTwitter.BASE, args)

    @staticmethod
    def _text_of(result) -> str:
        return (result.out_dir / "transcript.md").read_text(encoding="utf-8")

    def _dir_of(self, root, media_id) -> Path:
        # 按 media_path_key() 定位，而不是按 media_id 的字面量：身份进目录名时
        # 会被编码（含 `-`/`.` 等字符的 ID 走哈希分支），字面量匹配会认不出来。
        key = xsub.media_path_key(media_id)
        hits = [d for d in root.iterdir() if d.is_dir() and d.name.endswith(key)]
        self.assertEqual(len(hits), 1, f"没找到 {media_id}（键 {key}）的输出目录：{[d.name for d in root.iterdir()]}")
        return hits[0]

    # ---- E3-F01：音频
    def test_two_card_videos_each_get_their_own_audio(self):
        self.tw.MEDIA = []
        self.tw.CARDS = [self.CARD_A, self.CARD_B]
        with TempDir() as root:
            results = self._run_audio(root)
            self.assertEqual([r.media_id for r in results], ["xsubcard-555-1", "xsubcard-555-2"])
            self.assertIn("audio:card-A", self._text_of(results[0]))
            self.assertIn("audio:card-B", self._text_of(results[1]))
            self.assertNotIn("audio:card-B", self._text_of(results[0]), "两个卡片的音频串了")
        self.assertEqual(self.tw.download_calls, [], "卡片视频一次都不该按推文 URL 下载")
        self.assertEqual(len(self.tw.info_file_downloads), 2)
        for fed in self.tw.info_file_downloads:
            self.assertNotIn("webpage_url", fed, "留着 webpage_url 就等于留着回退整条推文的退路")

    def test_a_failing_card_never_falls_back_to_the_whole_tweet(self):
        """第二个卡片下载失败时，宁可整条判失败，也不能回退去下整条推文。"""
        self.tw.MEDIA = []
        self.tw.CARDS = [self.CARD_A, self.CARD_B]
        self.tw.fail_media = {"card-B"}
        with TempDir() as root:
            with self.assertRaises(xsub.MediaGroupError) as cm:
                self._run_audio(root)
            self.assertEqual([r.media_id for r in cm.exception.results], ["xsubcard-555-1"],
                             "已成功的卡片仍要保留")
            leaked = self._dir_of(root, "xsubcard-555-2") / "audio.m4a"
            self.assertFalse(leaked.exists(), "失败的卡片目录里不该出现任何音频，更不能是别人的")
        self.assertEqual(self.tw.download_calls, [], "绝不能退回按推文 URL 下载")

    def test_control_keeping_webpage_url_really_does_leak(self):
        """对照组：把保护（去掉 webpage_url）撤掉，上面那条测试必须变红。

        撤掉后上游会自动改下整条推文，把 card-A 的音频写进 card-B 的目录 ——
        证明上面的断言不是空转。
        """
        self.tw.MEDIA = []
        self.tw.CARDS = [self.CARD_A, self.CARD_B]
        self.tw.fail_media = {"card-B"}
        unprotected = lambda entry: dict(entry)  # noqa: E731  故意保留 webpage_url
        with TempDir() as root, mock.patch.object(xsub, "entry_download_info", unprotected):
            with self.assertRaises(xsub.MediaGroupError):
                self._run_audio(root)
            leaked = self._dir_of(root, "xsubcard-555-2") / "audio.m4a"
            self.assertTrue(leaked.exists(), "对照组本就该泄漏；没泄漏说明 fixture 没复刻上游回退")
            self.assertEqual(leaked.read_text(encoding="utf-8"), "audio:card-A",
                             "对照组应当把 card-A 的音频写进 card-B 的目录")
        self.assertIn(FakeTwitter.BASE, [u for u, _ in self.tw.download_calls])

    # ---- E3-F01：平台字幕
    def test_two_card_videos_each_get_their_own_native_subtitles(self):
        self.tw.MEDIA = []
        self.tw.CARDS = [self.SUBBED_A, self.SUBBED_B]
        with TempDir() as root:
            results = self._run_native_subs(root)
            self.assertEqual(len(results), 2)
            a, b = self._text_of(results[0]), self._text_of(results[1])
        self.assertIn("字幕属于 card-A", a)
        self.assertIn("字幕属于 card-B", b)
        self.assertNotIn("card-B", a, "两个卡片的字幕串了")
        self.assertNotIn("card-A", b, "两个卡片的字幕串了")
        self.assertEqual(self.tw.download_calls, [], "字幕也不该按推文 URL 下载")

    def test_control_url_level_subtitle_download_really_does_cross(self):
        """对照组：字幕改回按推文 URL 下载，两个卡片必然拿到同一份字幕。"""
        self.tw.MEDIA = []
        self.tw.CARDS = [self.SUBBED_A, self.SUBBED_B]
        real = xsub.fetch_native_subs

        def url_level(info, url, out_dir, want_lang, retry, download_info=None,
                      platform=xsub.PLATFORM_X):
            # 故意丢掉条目级隔离
            return real(info, url, out_dir, want_lang, retry, None, platform)

        with TempDir() as root:
            results = self._run_native_subs(root, patch_subs=url_level)
            a, b = self._text_of(results[0]), self._text_of(results[1])
        self.assertEqual(
            "字幕属于 card-B" in a, "字幕属于 card-B" in b,
            "对照组本就该两个目录拿到同一份字幕；不串说明 fixture 没复刻 outtmpl 覆盖",
        )

    # ---- E3-F02：不再靠 id == 推文 ID 猜来源
    def test_native_media_whose_id_equals_the_status_id_converges(self):
        """X 从没承诺原生视频的 media id 一定不等于推文 ID。

        真等上了也不能被当成卡片：bare 与 /video/1 必须收敛到同一个身份、同一个目录。
        """
        self.tw.MEDIA = [("video", FakeTwitter.STATUS, "原生视频")]
        self.tw.CARDS = []
        bare = xsub.resolve_targets(FakeTwitter.BASE, "555", None, xsub.CookieRetry(False))
        indexed = xsub.resolve_targets(f"{FakeTwitter.BASE}/video/1", "555", 1, xsub.CookieRetry(False))
        self.assertEqual(len(bare), 1)
        self.assertEqual(bare[0].media_id, "555", "被 /video/1 定位得到的就是原生视频，不该另造卡片身份")
        self.assertEqual(bare[0].media_id, indexed[0].media_id)
        self.assertEqual(bare[0].identity, indexed[0].identity, "两种写法必须命中同一份缓存")
        self.assertEqual(bare[0].source, "index")

    def test_a_native_video_is_not_double_processed_as_a_card(self):
        """id 撞上推文 ID 时，旧写法会既按原生处理一遍、又合成一个卡片身份。"""
        self.tw.MEDIA = [("video", FakeTwitter.STATUS, "原生视频"), ("video", "m-bbb", "另一个")]
        self.tw.CARDS = []
        targets = xsub.resolve_targets(FakeTwitter.BASE, "555", None, xsub.CookieRetry(False))
        self.assertEqual([t.media_id for t in targets], ["555", "m-bbb"])
        self.assertNotIn("xsubcard-555-1", [t.media_id for t in targets], "同一个视频被处理了两遍")

    # ---- E3-F03：transcript 身份只认 target
    def test_id_less_cards_report_their_own_identity_in_the_transcript(self):
        self.tw.MEDIA = []
        self.tw.CARDS = [self.CARD_A, self.CARD_B]
        with TempDir() as root:
            results = self._run_audio(root)
            for r, mid in zip(results, ["xsubcard-555-1", "xsubcard-555-2"]):
                line = f"- 媒体ID: {mid}    推文ID: 555"
                text = self._text_of(r)
                self.assertIn(line, text, "transcript 的身份必须和目录名/manifest 一致")
                self.assertNotIn("- 媒体ID: 555    ", text, "身份不能退回继承来的推文 ID")

    def test_external_card_transcript_reports_the_media_id_not_the_foreign_id(self):
        """外链卡片的 info["id"] 是外站的视频 ID，直接拿它当身份会和目录名分裂。"""
        self.tw.MEDIA = []
        self.tw.CARDS = [{"_type": "url", "url": TestF08CardVideos.PLAYER_URL}]
        self.tw.EXTERNAL = {
            TestF08CardVideos.PLAYER_URL: {**TestF08CardVideos.PLAYER_RESULT, "xsub_tag": "ext"}
        }
        with TempDir() as root:
            results = self._run_audio(root)
            self.assertEqual(results[0].media_id, "ext-999")
            self.assertIn("- 媒体ID: ext-999    推文ID: 555", self._text_of(results[0]))

    # ---- E3-F04：解析阶段的非 XsubError 异常
    def test_an_external_card_raising_a_plain_download_error_is_reported(self):
        """外链解析抛的是 yt-dlp 的 DownloadError，不是 XsubError。

        只 catch XsubError 的话它会直接穿出去，"哪几个没解析出来"的清单就永远打不出来。
        这里断言拿到的是 XsubError 而不是 DownloadError，本身就是那条保护的对照。
        """
        self.tw.MEDIA = [("video", "m-aaa", "原生视频")]
        self.tw.CARDS = [self.LONE_PLAYER]  # EXTERNAL 里没人认领 → 上游报 Unsupported URL
        with self.assertRaises(xsub.XsubError) as cm:
            xsub.resolve_targets(FakeTwitter.BASE, "555", None, xsub.CookieRetry(False))
        msg = str(cm.exception)
        self.assertIn("无法解析", msg)
        self.assertIn("第 2 个", msg)
        self.assertIn("DownloadError", msg, "要说清楚是什么错，而不是只报个数")
        self.assertNotIsInstance(cm.exception, self.tw.DownloadError)

    def test_a_download_that_yields_more_than_one_audio_file_is_rejected(self):
        """一个 target 只该产出一个音频。出现多个 = 下的不止当前这一个媒体。

        不同视频的容器格式本就可能不同（m4a / webm），%(ext)s 会落成不同文件名而不是
        互相覆盖。此时挑哪个都是猜，猜错就是"字幕属于错误视频"，只能整条拒绝。
        """
        self.tw.MEDIA = []
        self.tw.CARDS = [
            {**self.CARD_A, "xsub_ext": "m4a"},
            {**self.CARD_B, "xsub_ext": "webm"},
        ]
        with TempDir() as root, mock.patch.object(xsub.shutil, "which", return_value="/usr/bin/ffmpeg"):
            with self.assertRaises(xsub.XsubError) as cm:
                xsub.download_audio(
                    FakeTwitter.BASE, root, {"schema": 3, "status_id": "555", "media_id": "xsubcard-555-1"},
                    xsub.CookieRetry(False),
                )
            self.assertIn("多个音频文件", str(cm.exception))
            self.assertIsNone(
                next(iter(sorted(root.glob("audio.json"))), None), "拒绝时不该写下 manifest"
            )

    def test_a_synthesised_card_id_lives_in_its_own_namespace(self):
        """合成身份必须待在自带前缀的命名空间里。

        上一版只断言"合成身份不全是数字"，那只证明了它不会撞上 X 的数字型雪花号；
        可这个工具同时支持**外链播放器**，外站提取器的 ID 没有任何"必须是纯数字"的合同
        （YouTube 的 dq4Oj5quskI 就不是）。所以改成断言前缀命名空间——
        碰撞时还有 resolve_targets() 的硬失败兜底，见下一条用例。
        """
        for pos in (1, 2, 17):
            mid = xsub.entry_media_id({}, "2090176521335959721", pos)
            self.assertEqual(mid, f"xsubcard-2090176521335959721-{pos}")
            self.assertTrue(
                mid.startswith(xsub.SYNTHETIC_ID_PREFIX),
                "合成身份必须待在自己的命名空间里，不能和上游给的真实 ID 混在同一片字符串空间",
            )
            self.assertNotEqual(mid, "2090176521335959721")
        # entry 自带真实 id 时不得被合成身份顶掉
        self.assertEqual(xsub.entry_media_id({"id": "m-real"}, "555", 1), "m-real")

    def test_a_card_raising_keyboardinterrupt_is_not_swallowed(self):
        """兜底 except 只能吞 Exception：Ctrl-C 必须照常中断，不能被当成"这个视频解析失败"。"""
        self.tw.MEDIA = [("video", "m-aaa", "原生视频")]
        self.tw.CARDS = [self.CARD_A]
        with mock.patch.object(xsub, "card_target", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                xsub.resolve_targets(FakeTwitter.BASE, "555", None, xsub.CookieRetry(False))


class TestE3Round2FailClosed(FakeTwitterCase):
    """xsub-E3 第 2 轮：不知道的时候不许假装知道。

    两个毛病是同一件事的两半：
      · 序号探测抖了一下 → 被当成"确定没有" → 原生视频被合成成卡片身份，
        下次探测正常时又用回真实身份，同一个视频裂成两个目录、白转写一遍；
      · 两个条目算出同一个身份 → 后来的被静默丢掉 → 少处理一个视频却退出码 0。
    """

    INLINE_CARD = {"formats": [{"url": "https://vmap.example/card.m4a"}], "duration": 9.0}

    def _flaky_probe(self, *bad_indexes):
        """让指定序号的探测抛一个**临时性**错误：既不是越界，也不是图片。"""
        real = xsub.fetch_info
        bad = {f"{FakeTwitter.BASE}/video/{n}" for n in bad_indexes}

        def flaky(url, retry):
            if url in bad:
                raise TimeoutError("Read timed out. (read timeout=20)")
            return real(url, retry)

        return mock.patch.object(xsub, "fetch_info", side_effect=flaky)

    # ---- F02 缺口 A：探测的"没探明"必须被看见
    def test_a_transient_probe_failure_never_synthesises_a_card_identity(self):
        """单视频推文、id 恰好等于推文 ID：/video/1 探测抖一下就必须整条失败。"""
        self.tw.MEDIA = [("video", "555", "id 恰好等于推文 ID 的原生视频")]
        self.tw.CARDS = []
        with self._flaky_probe(1):
            with self.assertRaises(xsub.XsubError) as cm:
                xsub.resolve_targets(FakeTwitter.BASE, "555", None, xsub.CookieRetry(False))
        msg = str(cm.exception)
        self.assertIn("没能得到确定答案", msg)
        self.assertIn("TimeoutError", msg)
        self.assertNotIn(xsub.SYNTHETIC_ID_PREFIX, msg, "探不明时连合成身份都不该被造出来")

    def test_the_same_tweet_resolves_normally_once_the_probe_answers(self):
        """对照组：探测正常时同一条推文必须收敛到真实身份。

        没有这条，上一条测的可能只是"这个 fixture 永远失败"，而不是"没探明会 fail closed"。
        """
        self.tw.MEDIA = [("video", "555", "id 恰好等于推文 ID 的原生视频")]
        self.tw.CARDS = []
        targets = xsub.resolve_targets(FakeTwitter.BASE, "555", None, xsub.CookieRetry(False))
        self.assertEqual([t.media_id for t in targets], ["555"])
        self.assertEqual(targets[0].source, "index", "能被 /video/1 定位到，它就是原生视频")

    def test_a_transient_probe_failure_in_a_playlist_fails_the_whole_input(self):
        """多视频推文里探不明的那个条目不得被当成卡片，整条输入必须失败并点名是第几个。"""
        self.tw.MEDIA = [("video", "m-aaa", "第一个"), ("video", "m-bbb", "第二个")]
        self.tw.CARDS = []
        with self._flaky_probe(1):
            with self.assertRaises(xsub.XsubError) as cm:
                xsub.resolve_targets(FakeTwitter.BASE, "555", None, xsub.CookieRetry(False))
        msg = str(cm.exception)
        self.assertIn("第 1 个", msg)
        self.assertIn("没得到确定答案", msg)

    def test_a_photo_slot_stays_a_conclusive_answer(self):
        """图片占位是**确定**的回答，不能被误判成"没探明"——否则图文混排的推文全都失败。"""
        self.tw.MEDIA = [("video", "m-aaa", "原生视频"), ("photo", None, None)]
        self.tw.CARDS = [self.INLINE_CARD]
        targets = xsub.resolve_targets(FakeTwitter.BASE, "555", None, xsub.CookieRetry(False))
        self.assertEqual([t.media_id for t in targets], ["m-aaa", "xsubcard-555-2"])

    # ---- F02 缺口 B：身份撞车不许静默丢
    def test_an_external_card_colliding_with_a_synthetic_id_is_reported_not_dropped(self):
        """外站提取器给的 ID 不受"必须是纯数字"约束，理论上能撞上合成身份。

        撞了必须报错点名。静默 continue 会让这条推文少处理一个视频，
        却仍以退出码 0 收场 —— 正是"看着成功、其实漏了"的老毛病。
        """
        player = "https://player.example/watch?v=collide"
        self.tw.MEDIA = []
        self.tw.CARDS = [
            {"title": "没有自己 id 的卡片", "formats": [{"url": "https://vmap.example/a.m4a"}]},
            {"_type": "url", "url": player},
        ]
        self.tw.EXTERNAL = {
            player: {"id": "xsubcard-555-1", "title": "撞车的外站视频", "duration": 5.0,
                     "formats": [{"url": "https://ext.example/x.m4a"}]},
        }
        with self.assertRaises(xsub.XsubError) as cm:
            xsub.resolve_targets(FakeTwitter.BASE, "555", None, xsub.CookieRetry(False))
        msg = str(cm.exception)
        self.assertIn("第 2 个", msg)
        self.assertIn("xsubcard-555-1", msg)
        self.assertIn("撞", msg)

    # ---- F06：同一个视频，链接怎么写产物都必须一样
    def test_the_transcript_is_identical_for_bare_and_indexed_urls(self):
        """字幕正文里的「来源」行改用规范链接，两种写法的产物必须逐字相同。

        否则同一份字幕会算出两个 sha256，摘要与字幕的绑定就会把一份仍然有效的
        摘要判成「非本轮产物」。
        """
        single = FakeTwitter()
        single.MEDIA = [("video", "m-only", "唯一的视频")]
        sys.modules["yt_dlp"] = single.as_module()
        with TempDir() as root:
            with mock.patch.object(xsub, "fetch_native_subs", side_effect=self._subs_per_media):
                bare = xsub.process(FakeTwitter.BASE, Args(root, no_summary=True, native_subs=True))[0]
                first = (bare.out_dir / "transcript.md").read_text(encoding="utf-8")
                indexed = xsub.process(
                    f"{FakeTwitter.BASE}/video/1", Args(root, no_summary=True, native_subs=True)
                )[0]
                second = (indexed.out_dir / "transcript.md").read_text(encoding="utf-8")
        self.assertEqual(bare.out_dir, indexed.out_dir, "同一个媒体必须落在同一个目录")
        self.assertEqual(first, second, "同一个视频换个链接写法，字幕正文不该有任何差别")
        self.assertIn(f"- 来源: {FakeTwitter.BASE}\n", first, "来源行应是不带序号的规范链接")

    def test_a_valid_summary_survives_reopening_the_media_with_an_index(self):
        """bare 出的摘要，换 /video/1 打开后仍应算「本轮产物」。"""
        single = FakeTwitter()
        single.MEDIA = [("video", "m-only", "唯一的视频")]
        sys.modules["yt_dlp"] = single.as_module()
        with TempDir() as root:
            with mock.patch.object(xsub, "fetch_native_subs", side_effect=self._subs_per_media):
                bare = xsub.process(FakeTwitter.BASE, Args(root, no_summary=True, native_subs=True))[0]
                md = (bare.out_dir / "transcript.md").read_text(encoding="utf-8")
                (bare.out_dir / "summary.md").write_text("摘要正文", encoding="utf-8")
                (bare.out_dir / "summary.meta.json").write_text(
                    json.dumps({"transcript_sha256": xsub.sha256_text(md)}), encoding="utf-8"
                )
                self.assertEqual(xsub.summary_state(bare.out_dir), "current")
                xsub.process(
                    f"{FakeTwitter.BASE}/video/1", Args(root, no_summary=True, native_subs=True)
                )
                self.assertEqual(
                    xsub.summary_state(bare.out_dir), "current",
                    "换个链接写法不该把一份仍然有效的摘要判成过期",
                )


class TestSummaryAuthPreflight(unittest.TestCase):
    """xsub-E2 事后取证 F01：剔除环境变量挡不住机器上存着的 Console 登录凭据。

    那种登录照样按 API 用量计费，与"摘要只走订阅额度"的承诺冲突，
    所以摘要前先问一次 `claude auth status`，不是订阅就不出摘要。
    """

    def _status(self, payload, returncode=0):
        return mock.Mock(returncode=returncode, stdout=json.dumps(payload), stderr="")

    SUBSCRIPTION = {
        "loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty",
        "email": "someone@example.com", "subscriptionType": "max",
    }

    def test_subscription_login_passes(self):
        with mock.patch.object(xsub.subprocess, "run", return_value=self._status(self.SUBSCRIPTION)):
            ok, how = xsub.claude_auth_kind("/usr/bin/claude")
        self.assertTrue(ok)
        self.assertIn("authMethod=claude.ai", how)
        self.assertIn("max", how)

    def test_the_description_never_leaks_the_account_email(self):
        with mock.patch.object(xsub.subprocess, "run", return_value=self._status(self.SUBSCRIPTION)):
            _, how = xsub.claude_auth_kind("/usr/bin/claude")
        self.assertNotIn("someone@example.com", how)
        self.assertNotIn("@", how, "登录状态描述里不该出现任何账号标识")

    def test_console_and_bedrock_logins_are_rejected(self):
        cases = [
            {"loggedIn": True, "authMethod": "apiKey", "apiProvider": "firstParty"},
            {"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "bedrock"},
            {"loggedIn": False, "authMethod": "claude.ai", "apiProvider": "firstParty"},
            {},
        ]
        for payload in cases:
            with mock.patch.object(xsub.subprocess, "run", return_value=self._status(payload)):
                ok, _ = xsub.claude_auth_kind("/usr/bin/claude")
            self.assertFalse(ok, f"{payload} 不是订阅登录，必须拒绝")

    def test_a_wrong_route_is_rejected_even_when_a_real_plan_is_present(self):
        """档位对了也不算数：登录方式和计费路由必须**各自**过关。

        上一条用例里的反面样本都没带 subscriptionType，于是档位那道闸顺手也拦住了它们——
        变异测试因此发现「登录方式 / apiProvider」这道闸即使整个撤掉也没有测试变红。
        这里把带着真实档位的错误路由单独钉住：Bedrock/Vertex 走的是云厂商账单，
        apiKey 走的是 Console 按量计费，两者都不消耗订阅额度。
        """
        cases = [
            {"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "bedrock",
             "subscriptionType": "max"},
            {"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "vertex",
             "subscriptionType": "enterprise"},
            {"loggedIn": True, "authMethod": "apiKey", "apiProvider": "firstParty",
             "subscriptionType": "max"},
        ]
        for payload in cases:
            with mock.patch.object(xsub.subprocess, "run", return_value=self._status(payload)):
                ok, why = xsub.claude_auth_kind("/usr/bin/claude")
            self.assertFalse(ok, f"{payload} 不走订阅额度，有档位也必须拒绝")
            self.assertIn("不是 claude.ai 订阅登录", why, "拒绝理由该指向路由，而不是档位")

    def test_unparseable_or_failing_status_is_rejected(self):
        with mock.patch.object(
            xsub.subprocess, "run", return_value=mock.Mock(returncode=0, stdout="not json", stderr="")
        ):
            self.assertFalse(xsub.claude_auth_kind("/usr/bin/claude")[0])
        for exc in (OSError("no such file"), xsub.subprocess.TimeoutExpired(cmd="claude", timeout=60)):
            with mock.patch.object(xsub.subprocess, "run", side_effect=exc):
                self.assertFalse(xsub.claude_auth_kind("/usr/bin/claude")[0], "查不清楚就不出摘要")

    def test_the_preflight_never_carries_api_keys_into_the_probe(self):
        seen = {}

        def fake_run(argv, **kw):
            seen["argv"] = argv
            seen["env"] = kw.get("env")
            return self._status(self.SUBSCRIPTION)

        with mock.patch.dict(xsub.os.environ, {"ANTHROPIC_API_KEY": "sk-should-not-leak"}), \
             mock.patch.object(xsub.subprocess, "run", side_effect=fake_run):
            xsub.claude_auth_kind("/usr/bin/claude")
        self.assertEqual(seen["argv"][1:], ["auth", "status"])
        self.assertNotIn("ANTHROPIC_API_KEY", seen["env"])

    def test_a_firstparty_claude_ai_login_without_a_plan_is_rejected(self):
        """anthropics/claude-code#36769：authMethod=claude.ai + apiProvider=firstParty
        也可能 subscriptionType=null，而那种状态实测走的是 API 计费、命中的是 API 限额。
        光看认证方式和 provider 推不出"不会计费"，必须看到订阅档位本身。
        """
        for plan in (None, "", "   "):
            payload = {**self.SUBSCRIPTION, "subscriptionType": plan}
            with mock.patch.object(xsub.subprocess, "run", return_value=self._status(payload)):
                ok, how = xsub.claude_auth_kind("/usr/bin/claude")
            self.assertFalse(ok, f"subscriptionType={plan!r} 拿不到订阅证据，必须拒绝")
            self.assertIn("订阅", how)

    def test_a_missing_subscription_field_never_invokes_the_summary(self):
        """字段整个缺失时也必须 fail closed，而且**一次 claude -p 都不能真的跑**。"""
        payload = {k: v for k, v in self.SUBSCRIPTION.items() if k != "subscriptionType"}
        with mock.patch.object(xsub.subprocess, "run", return_value=self._status(payload)):
            self.assertFalse(xsub.claude_auth_kind("/usr/bin/claude")[0])

        with TempDir() as d:
            md = d / "transcript.md"
            md.write_text("hello", encoding="utf-8")
            seen = []

            def watch(argv, **kw):
                seen.append(list(argv))
                if list(argv[1:3]) == ["auth", "status"]:
                    return self._status(payload)
                raise AssertionError("拿不到订阅档位就绝不能跑 claude -p（会产生 API 账单）")

            with mock.patch.object(xsub.shutil, "which", return_value="/usr/bin/claude"), \
                 mock.patch.object(xsub.subprocess, "run", side_effect=watch):
                status = xsub.summarize(md, d / "summary.md", d / "summary.meta.json")
        self.assertEqual(status, "skipped")
        self.assertEqual([a[1:3] for a in seen], [["auth", "status"]], "问过身份就该就此打住")

    def test_an_unknown_subscription_plan_is_rejected(self):
        """白名单之外的档位值一律当作"没问清楚"，宁可不出摘要。"""
        with mock.patch.object(
            xsub.subprocess, "run",
            return_value=self._status({**self.SUBSCRIPTION, "subscriptionType": "some-new-tier"}),
        ):
            self.assertFalse(xsub.claude_auth_kind("/usr/bin/claude")[0])

    def test_a_supported_oauth_schema_with_a_real_plan_is_accepted(self):
        """同一种订阅登录在不同 CLI 版本里认证方式字符串写法不同，只认一种会误拒真订阅用户。"""
        for method in ("claude.ai", "oauth", "oauth_token"):
            payload = {**self.SUBSCRIPTION, "authMethod": method, "subscriptionType": "pro"}
            with mock.patch.object(xsub.subprocess, "run", return_value=self._status(payload)):
                ok, _ = xsub.claude_auth_kind("/usr/bin/claude")
            self.assertTrue(ok, f"{method} + 有效订阅档位应当放行")

    def test_a_nonzero_auth_status_exit_is_rejected(self):
        """进程失败时 stdout 里可能还留着看着正常的 JSON，退出码必须先看。"""
        with mock.patch.object(
            xsub.subprocess, "run", return_value=self._status(self.SUBSCRIPTION, returncode=1)
        ):
            self.assertFalse(xsub.claude_auth_kind("/usr/bin/claude")[0])

    def test_summarize_refuses_to_run_on_a_non_subscription_login(self):
        with TempDir() as d:
            md = d / "transcript.md"
            md.write_text("hello", encoding="utf-8")

            def must_not_run(*a, **kw):
                raise AssertionError("不是订阅登录就绝不能真的去跑 claude -p（会产生 API 账单）")

            with mock.patch.object(xsub.shutil, "which", return_value="/usr/bin/claude"), \
                 mock.patch.object(xsub, "claude_auth_kind", return_value=(False, "authMethod=apiKey")), \
                 mock.patch.object(xsub.subprocess, "run", side_effect=must_not_run):
                status = xsub.summarize(md, d / "summary.md", d / "summary.json")
        self.assertEqual(status, "skipped")
        self.assertFalse((d / "summary.md").exists() if d.exists() else False)


class TestF01MediaIdentity(FakeTwitterCase):
    """F01 剩余根因：缓存与目录身份只有推文 ID，同一条推文的多个视频会串用/覆盖。"""

    def test_parse_x_url_keeps_the_video_selector(self):
        url, sid, idx = xsub.parse_x_url("https://twitter.com/multi/status/555/video/2?s=20")
        self.assertEqual(url, "https://x.com/multi/status/555/video/2")
        self.assertEqual((sid, idx), ("555", 2))
        url, sid, idx = xsub.parse_x_url("https://x.com/multi/status/555")
        self.assertEqual((url, sid, idx), ("https://x.com/multi/status/555", "555", None))

    def test_parse_x_url_rejects_photo_and_absurd_index(self):
        with self.assertRaises(xsub.XsubError):
            xsub.parse_x_url("https://x.com/multi/status/555/photo/1")
        with self.assertRaises(xsub.XsubError):
            xsub.parse_x_url("https://x.com/multi/status/555/video/999")

    def test_ydl_opts_always_sets_noplaylist(self):
        for cookies in (False, True):
            self.assertIs(xsub.ydl_opts(cookies).get("noplaylist"), True)

    def test_fixture_reproduces_the_upstream_trap(self):
        """自证 fixture 有效：不开 noplaylist 时，/video/3 会被忽略、退回整条 playlist。"""
        with sys.modules["yt_dlp"].YoutubeDL({}) as ydl:
            got = ydl.extract_info(f"{FakeTwitter.BASE}/video/3", download=False)
        self.assertTrue(xsub.is_playlist_result(got))

    def test_indexed_urls_resolve_to_distinct_media_ids(self):
        retry = xsub.CookieRetry(False)
        one = xsub.resolve_targets(f"{FakeTwitter.BASE}/video/1", "555", 1, retry)
        three = xsub.resolve_targets(f"{FakeTwitter.BASE}/video/3", "555", 3, retry)
        self.assertEqual([t.media_id for t in one], ["m-aaa"])
        self.assertEqual([t.media_id for t in three], ["m-bbb"])
        self.assertNotEqual(one[0].identity, three[0].identity)

    def test_bare_url_expands_to_every_video_and_skips_photos(self):
        """不得悄悄拿第一条顶包：多视频推文要逐个解析、逐个输出。"""
        targets = xsub.resolve_targets(FakeTwitter.BASE, "555", None, xsub.CookieRetry(False))
        self.assertEqual([t.media_id for t in targets], ["m-aaa", "m-bbb"])
        self.assertEqual([t.index for t in targets], [1, 3], "图片占用的序号 2 必须跳过")

    def test_playlist_result_is_never_processed_as_a_single_video(self):
        """playlist 没有 formats、标题时长都是推文级的；当单视频用就会产出混合产物。"""
        playlist = {"_type": "playlist", "id": "555", "title": "整条推文", "entries": [{"id": "m-aaa"}]}
        self.assertTrue(xsub.is_playlist_result(playlist))
        self.assertTrue(xsub.is_playlist_result({"entries": []}))
        self.assertFalse(xsub.is_playlist_result({"id": "m-aaa", "formats": [{}]}))

        # 指定了序号却拿回 playlist（例如上游行为变化）→ 必须报错，不能硬着头皮用
        with mock.patch.object(xsub, "fetch_info", return_value=playlist):
            with self.assertRaises(xsub.XsubError):
                xsub.resolve_targets(f"{FakeTwitter.BASE}/video/1", "555", 1, xsub.CookieRetry(False))

        # 兜底断言：即使 target 被伪造成 playlist，process_media 也要拒绝
        target = xsub.MediaTarget.__new__(xsub.MediaTarget)
        target.url, target.index, target.info = FakeTwitter.BASE, None, playlist
        target.status_id, target.media_id = "555", "555"
        with TempDir() as root:
            with self.assertRaises(xsub.XsubError):
                xsub.process_media(target, Args(root, no_summary=True), xsub.CookieRetry(False))

    def test_cookie_retry_does_not_change_the_selected_entry(self):
        self.tw.fail_anonymous = True
        retry = xsub.CookieRetry(False)
        targets = xsub.resolve_targets(f"{FakeTwitter.BASE}/video/3", "555", 3, retry)
        self.assertTrue(retry.cookies, "403 之后应已切到 Chrome 登录态")
        self.assertEqual(targets[0].media_id, "m-bbb", "换认证方式不得换掉选中的媒体")

    def _run(self, root, raw_url):
        args = Args(root, no_summary=True, native_subs=True)
        with mock.patch.object(xsub, "fetch_native_subs", side_effect=self._subs_per_media):
            return xsub.process(raw_url, args)

    def test_two_videos_of_one_tweet_never_share_a_directory_or_cache(self):
        for order in (["/video/1", "/video/3"], ["/video/3", "/video/1"]):
            with TempDir() as root:
                results = [self._run(root, FakeTwitter.BASE + suffix)[0] for suffix in order]
                dirs = {r.out_dir for r in results}
                self.assertEqual(len(dirs), 2, f"{order}: 两个视频必须落在两个目录")
                for r in results:
                    text = (r.out_dir / "transcript.md").read_text(encoding="utf-8")
                    self.assertIn(f"字幕属于 {r.media_id}", text, f"{order}: 字幕串到了别的视频")
                    cache = json.loads((r.out_dir / "segments.json").read_text(encoding="utf-8"))
                    self.assertEqual(cache["media_id"], r.media_id)
                    self.assertEqual(cache["status_id"], "555")

    def test_bare_url_processes_both_videos_into_separate_outputs(self):
        with TempDir() as root:
            results = self._run(root, FakeTwitter.BASE)
            self.assertEqual(len(results), 2)
            self.assertEqual({r.media_id for r in results}, {"m-aaa", "m-bbb"})
            self.assertEqual(len({r.out_dir for r in results}), 2)

    def test_transcript_and_manifests_agree_on_media_id(self):
        with TempDir() as root:
            r = self._run(root, f"{FakeTwitter.BASE}/video/3")[0]
            cache = json.loads((r.out_dir / "segments.json").read_text(encoding="utf-8"))
            md = (r.out_dir / "transcript.md").read_text(encoding="utf-8")
            self.assertEqual(cache["media_id"], "m-bbb")
            self.assertEqual(cache["media_index"], 3)
            self.assertEqual(cache["media_url"], f"{FakeTwitter.BASE}/video/3")
            self.assertIn("媒体ID: m-bbb", md)
            self.assertIn("推文ID: 555", md)
            self.assertIn("m-bbb", r.out_dir.name, "目录名必须带媒体身份")

    def test_same_media_via_bare_and_indexed_url_shares_the_cache(self):
        """单视频推文里 /status/<id> 与 /status/<id>/video/1 是同一个视频，不能白跑两遍。"""
        single = FakeTwitter()
        single.MEDIA = [("video", "m-only", "唯一的视频")]
        sys.modules["yt_dlp"] = single.as_module()
        with TempDir() as root:
            first = self._run(root, FakeTwitter.BASE)[0]
            calls_after_first = len(single.extract_calls)
            second = self._run(root, f"{FakeTwitter.BASE}/video/1")[0]
        self.assertEqual(first.out_dir, second.out_dir, "同一个媒体必须落在同一个目录")
        self.assertEqual(first.media_id, "m-only")
        self.assertGreater(calls_after_first, 0)

    def test_segments_cache_of_another_media_is_a_miss(self):
        with TempDir() as d:
            identity = {
                "schema": xsub.CACHE_SCHEMA,
                "status_id": "555",
                "media_id": "m-bbb",
                "model": xsub.DEFAULT_MODEL,
                "lang_request": None,
            }
            (d / "segments.json").write_text(
                json.dumps({**identity, "media_id": "m-aaa", "segments": [{"start": 0, "end": 1, "text": "别的视频"}]}),
                encoding="utf-8",
            )
            self.assertIsNone(xsub.load_segments_cache(d / "segments.json", identity))

    def test_audio_manifest_of_another_media_is_a_miss(self):
        with TempDir() as d:
            identity = {"schema": xsub.CACHE_SCHEMA, "status_id": "555", "media_id": "m-bbb"}
            (d / "audio.m4a").write_bytes(b"1234")
            (d / "audio.json").write_text(
                json.dumps({**identity, "media_id": "m-aaa", "file": "audio.m4a", "size": 4}), encoding="utf-8"
            )
            with mock.patch.object(xsub.shutil, "which", return_value=None):
                with self.assertRaises(xsub.XsubError):  # 判 miss → 要下载 → 缺 ffmpeg 报错
                    xsub.download_audio(f"{FakeTwitter.BASE}/video/3", d, identity, xsub.CookieRetry(False))

    def test_one_failing_media_does_not_discard_the_successful_one(self):
        with TempDir() as root:
            args = Args(root, no_summary=True, native_subs=True)
            calls = {"n": 0}

            def flaky(info, url, out_dir, want_lang, retry, download_info=None,
                      platform=xsub.PLATFORM_X):
                calls["n"] += 1
                if info["id"] == "m-bbb":
                    raise xsub.XsubError("模拟第二个视频失败")
                return self._subs_per_media(
                    info, url, out_dir, want_lang, retry, download_info, platform
                )

            with mock.patch.object(xsub, "fetch_native_subs", side_effect=flaky):
                with self.assertRaises(xsub.MediaGroupError) as ctx:
                    xsub.process(FakeTwitter.BASE, args)
            self.assertEqual(len(ctx.exception.results), 1, "已成功的媒体不能被丢掉")
            self.assertEqual(ctx.exception.results[0].media_id, "m-aaa")


class TestE4PathIdentityAndBilling(FakeTwitterCase):
    """xsub-E4：身份落到磁盘路径时必须仍然一一对应；计费承诺不得超出能证明的范围。

    E3 第 3 轮关闭性复审留下的两条同根因缺口：
      - F02：resolve_out_dir 把媒体身份截到前 25 个字符，而真实 19 位推文 ID 生成的
        合成身份有 30 个字符，区分两个卡片的尾号正好被砍掉——内存里身份唯一，
        磁盘上却重新碰撞，后一个把前一个的音频/字幕/摘要整个覆盖，两次都报成功。
        E3 的双卡片 fixture 用的是三位 status「555」，短到根本触发不了截断。
      - F05：认证预检只能机械证明「走的是订阅登录路由」，证明不了「这次调用不会
        按量计费」。区分不了就不能替用户决定。
    """

    REAL_STATUS = "2090176521335959721"  # 真实长度的推文 ID，就是 F02 的触发条件
    LONG_TITLE = "Greg Isenberg on the twelve startup ideas that actually printed money"

    ENTERPRISE = {
        "loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty",
        "subscriptionType": "enterprise",
    }

    @staticmethod
    def _status(payload, returncode=0):
        return mock.Mock(returncode=returncode, stdout=json.dumps(payload), stderr="")

    @contextlib.contextmanager
    def _without_metered_optin(self):
        """确保跑测试的机器上恰好设了授权环境变量时，负向用例不会被悄悄放行。"""
        with mock.patch.dict(xsub.os.environ, {}, clear=False):
            xsub.os.environ.pop(xsub.ALLOW_METERED_ENV, None)
            yield

    def _real_length_tweet(self) -> str:
        """真实长度的推文 ID + 两个没有独立 ID 的卡片 + 共用长标题。"""
        self.tw.STATUS = self.REAL_STATUS
        self.tw.BASE = f"https://x.com/multi/status/{self.REAL_STATUS}"
        self.tw.MEDIA = []
        self.tw.CARDS = [
            {"xsub_tag": "card-1", "duration": 9.0, "title": f"{self.LONG_TITLE} #1",
             "formats": [{"url": "https://vmap.example/1.m4a"}]},
            {"xsub_tag": "card-2", "duration": 8.0, "title": f"{self.LONG_TITLE} #2",
             "formats": [{"url": "https://vmap.example/2.m4a"}]},
        ]
        return self.tw.BASE

    # ---- E3-F02 余留：身份 → 路径的编码必须一一对应
    def test_a_media_id_too_long_for_the_path_keeps_its_own_key(self):
        a = xsub.synthetic_media_id(self.REAL_STATUS, 1)
        b = xsub.synthetic_media_id(self.REAL_STATUS, 2)
        self.assertGreater(len(a), xsub.MEDIA_KEY_LIMIT, "这条用例的前提就是身份放不进路径")
        ka, kb = xsub.media_path_key(a), xsub.media_path_key(b)
        self.assertNotEqual(ka, kb, "两个不同的媒体身份不能编码成同一个目录名")
        self.assertLessEqual(len(ka), xsub.MEDIA_KEY_LIMIT)
        self.assertLessEqual(len(kb), xsub.MEDIA_KEY_LIMIT)

    def test_a_normal_length_media_id_is_left_alone(self):
        """真实 X 媒体 ID 只有 19 位，目录名必须保持原样，否则老用户的缓存全部失效。"""
        self.assertEqual(xsub.media_path_key("1234567890123456789"), "1234567890123456789")

    def test_two_id_less_cards_never_share_an_output_directory(self):
        """长标题被截到 50 个字符后上游追加的 #1/#2 就没了，只剩媒体身份能区分它们。"""
        with TempDir() as root:
            dirs = {
                xsub.resolve_out_dir(
                    root,
                    {"upload_date": "20260101", "uploader_id": "multi",
                     "title": f"{self.LONG_TITLE} #{k}"},
                    self.REAL_STATUS,
                    xsub.synthetic_media_id(self.REAL_STATUS, k),
                )
                for k in (1, 2)
            }
        self.assertEqual(len(dirs), 2, f"两个卡片算出了同一个目录：{[d.name for d in dirs]}")

    def test_a_full_run_of_two_id_less_cards_leaves_two_intact_products(self):
        """完整跑一遍：目录数、结果数都必须是 2，且每份产物都是它自己的。"""
        base = self._real_length_tweet()
        with TempDir() as root:
            args = Args(root, no_summary=True)
            with mock.patch.object(xsub, "fetch_native_subs", return_value=None), \
                 mock.patch.object(xsub.shutil, "which", return_value="/usr/bin/ffmpeg"), \
                 mock.patch.object(
                     xsub, "transcribe",
                     side_effect=lambda a, m, l, v: (
                         [{"start": 0.0, "end": 1.0,
                           "text": a.read_text(encoding="utf-8", errors="replace")}], "en",
                     ),
                 ):
                results = xsub.process(base, args)

            self.assertEqual(len(results), 2, "两个卡片都该有产物")
            self.assertEqual(
                len({r.out_dir for r in results}), 2,
                f"两个卡片落进了同一个目录：{[r.out_dir.name for r in results]}",
            )
            on_disk = [d for d in root.iterdir() if d.is_dir()]
            self.assertEqual(len(on_disk), 2, f"磁盘上只剩 {len(on_disk)} 个目录，有产物被覆盖了")

            tags = []
            for r in results:
                text = (r.out_dir / "transcript.md").read_text(encoding="utf-8")
                self.assertIn(f"媒体ID: {r.media_id}", text, "字幕自报的身份必须是它自己的")
                hit = re.search(r"audio:(card-\d)", text)
                self.assertIsNotNone(hit, f"{r.out_dir.name} 里的字幕不是从它自己的音频转出来的")
                tags.append(hit.group(1))
            self.assertEqual(sorted(tags), ["card-1", "card-2"], "两个目录必须各自装着自己的音频")

    # ---- E3-F05 余留：能证明的只有路由，不是「一定不计费」
    def test_a_metered_risk_plan_is_not_treated_as_included_quota(self):
        """存在「席位费只买访问权、用量全部按 API 费率另计」的 Enterprise 形态，
        而 `claude auth status` 没有字段能把它和包含额度的形态区分开。"""
        with self._without_metered_optin(), \
             mock.patch.object(xsub.subprocess, "run", return_value=self._status(self.ENTERPRISE)):
            ok, why = xsub.claude_auth_kind("/usr/bin/claude")
        self.assertFalse(ok, "区分不了就不能替用户决定，默认必须跳过摘要")
        self.assertIn("--allow-metered-summary", why, "拒绝时要告诉用户怎么显式授权")

    def test_a_metered_risk_plan_passes_once_the_user_takes_the_risk(self):
        with mock.patch.object(xsub.subprocess, "run", return_value=self._status(self.ENTERPRISE)):
            ok_flag, _ = xsub.claude_auth_kind("/usr/bin/claude", allow_metered=True)
            with mock.patch.dict(xsub.os.environ, {xsub.ALLOW_METERED_ENV: "1"}):
                ok_env, _ = xsub.claude_auth_kind("/usr/bin/claude")
        self.assertTrue(ok_flag, "显式加了 --allow-metered-summary 就该放行")
        self.assertTrue(ok_env, f"设了 {xsub.ALLOW_METERED_ENV}=1 也该放行")

    def test_included_quota_plans_still_pass_without_any_extra_flag(self):
        """新闸门只针对分不清的档位，不能顺手误伤本来就有包含额度的自助订阅。"""
        for plan in sorted(xsub.INCLUDED_QUOTA_PLANS):
            payload = {**self.ENTERPRISE, "subscriptionType": plan}
            with self._without_metered_optin(), \
                 mock.patch.object(xsub.subprocess, "run", return_value=self._status(payload)):
                ok, _ = xsub.claude_auth_kind("/usr/bin/claude")
            self.assertTrue(ok, f"{plan} 有包含额度，不该被这道新闸门拦下")

    def test_a_metered_risk_plan_never_reaches_the_real_summariser(self):
        """账单是在 `claude -p` 那一步产生的，所以拒绝时它的真实调用次数必须是 0。"""
        with TempDir() as d:
            md = d / "transcript.md"
            md.write_text("hello", encoding="utf-8")
            calls = []

            def spy(argv, *a, **kw):
                calls.append(list(argv))
                return self._status(self.ENTERPRISE)

            with self._without_metered_optin(), \
                 mock.patch.object(xsub.shutil, "which", return_value="/usr/bin/claude"), \
                 mock.patch.object(xsub.subprocess, "run", side_effect=spy):
                status = xsub.summarize(md, d / "summary.md", d / "summary.json")
            self.assertEqual(status, "skipped")
            self.assertFalse((d / "summary.md").exists())
            self.assertEqual([c for c in calls if "-p" in c], [], "预检没过就绝不能真的跑 claude -p")

    # ---- E3-F08：删序号前必须先确认这是 X 推文链接
    def test_an_external_card_url_keeps_its_index_suffix(self):
        """正则本身不看 host。外站地址恰好以 /video/1 结尾也被削掉的话，
        字幕的来源行就会指向一个不存在的页面。"""
        info = {"title": "外链卡片"}
        for url in ("https://player.example/video/1", "https://player.example/photo/1"):
            t = xsub.MediaTarget(
                url=url, index=None, info=info, status_id=self.REAL_STATUS,
                media_id="ext-1", source="entry-url",
            )
            self.assertEqual(t.canonical_url, url, f"外站地址必须原样保留：{url}")

    def test_an_x_url_still_drops_the_selector(self):
        """对照组：X 链接该删的还得删，否则 bare 与 /video/N 会写出两份不同的来源行。"""
        info = {"title": "推文视频"}
        base = f"https://x.com/multi/status/{self.REAL_STATUS}"
        canon = {
            xsub.MediaTarget(
                url=u, index=None, info=info, status_id=self.REAL_STATUS, media_id="m-1",
            ).canonical_url
            for u in (base, f"{base}/video/1", f"{base}/video/2")
        }
        self.assertEqual(canon, {base}, "同一个媒体不该因为链接写法不同而写出不同的来源行")


class TestE4R1PathAliasing(FakeTwitterCase):
    """xsub-E4 第 1 轮 F01：路径键的第二层多对一——**清洗**本身。

    上一轮只堵住了「太长被截断」这一层，短 ID 照样撞：clean_component() 把
    `a..b` 和 `a_b` 都变成 `a_b`、把 `abc-` 和 `abc` 都变成 `abc`。两个身份在
    内存里的 taken 集合里是不同的，落到磁盘上却是同一个目录，后一个把前一个的
    音频/字幕/摘要整个覆盖掉，两次还都报成功、退出码 0。
    """

    STATUS = "2090176521335959721"
    LONG_TITLE = "Greg Isenberg on the twelve startup ideas that actually printed money"

    def _tweet_with_ids(self, id_a: str, id_b: str) -> str:
        """同一条推文里两个卡片，媒体 ID 不同但清洗后会撞在一起；共用长标题。"""
        self.tw.STATUS = self.STATUS
        self.tw.BASE = f"https://x.com/multi/status/{self.STATUS}"
        self.tw.MEDIA = []
        self.tw.CARDS = [
            {"id": id_a, "xsub_tag": "card-1", "duration": 9.0,
             "title": f"{self.LONG_TITLE} #1", "formats": [{"url": "https://vmap.example/1.m4a"}]},
            {"id": id_b, "xsub_tag": "card-2", "duration": 8.0,
             "title": f"{self.LONG_TITLE} #2", "formats": [{"url": "https://vmap.example/2.m4a"}]},
        ]
        return self.tw.BASE

    @staticmethod
    @contextlib.contextmanager
    def _transcribing():
        """把字幕内容做成"音频里写了什么就转出什么"，串了目录一眼看得出来。"""
        with mock.patch.object(xsub, "fetch_native_subs", return_value=None), \
             mock.patch.object(xsub.shutil, "which", return_value="/usr/bin/ffmpeg"), \
             mock.patch.object(
                 xsub, "transcribe",
                 side_effect=lambda a, m, l, v: (
                     [{"start": 0.0, "end": 1.0,
                       "text": a.read_text(encoding="utf-8", errors="replace")}], "en",
                 ),
             ):
            yield

    # ---- 纯函数层：清洗造成的别名
    def test_two_short_ids_that_clean_to_the_same_string_keep_their_own_keys(self):
        """外站播放器的 ID 不受"必须是纯数字"约束，这几对都是真实可达的写法。"""
        for a, b in (("a..b", "a_b"), ("abc-", "abc"), ("x/y", "x_y"), ("v.1", "v_1")):
            ka, kb = xsub.media_path_key(a), xsub.media_path_key(b)
            self.assertNotEqual(ka, kb, f"{a!r} 和 {b!r} 编码成了同一个目录名 {ka!r}")
            self.assertLessEqual(len(ka), xsub.MEDIA_KEY_LIMIT)
            self.assertLessEqual(len(kb), xsub.MEDIA_KEY_LIMIT)

    def test_ids_differing_only_by_case_never_alias_on_the_filesystem(self):
        """macOS 默认文件系统大小写不敏感：键本身不同还不够，小写化之后也必须不同。"""
        ka, kb = xsub.media_path_key("ABC"), xsub.media_path_key("abc")
        self.assertNotEqual(ka.lower(), kb.lower(), "大小写不敏感的文件系统上这两个是同一个目录")

    def test_a_non_ascii_id_never_passes_through_untouched(self):
        """Unicode 归一化（NFC/NFD）同样会在文件系统层面制造别名，一律走哈希。"""
        for raw in ("café", "视频一"):
            key = xsub.media_path_key(raw)
            self.assertNotEqual(key, raw, f"{raw!r} 不该原样进目录名")
            self.assertIn("-", key, "非 ASCII 身份必须带上完整原始 ID 的哈希")

    def test_the_reserved_empty_key_cannot_be_claimed_by_a_real_id(self):
        """一个恰好叫 nomedia 的外站 ID 不能和"根本没有 ID"落进同一个目录。"""
        self.assertNotEqual(xsub.media_path_key("nomedia"), xsub.media_path_key(""))

    def test_a_real_19_digit_media_id_is_still_left_alone(self):
        """向后兼容的底线：真实 X 媒体 ID 原样通过，老目录名和老缓存不失效。"""
        self.assertEqual(xsub.media_path_key("2090175928143867905"), "2090175928143867905")

    def test_the_two_branches_can_never_produce_the_same_shape(self):
        """原样分支只含小写字母数字，哈希分支必定含 `-`：两条路算不出同一个键。"""
        self.assertNotIn("-", xsub.media_path_key("abc123"))
        self.assertIn("-", xsub.media_path_key("a..b"))

    # ---- 端到端：审查方点名要求的那条回归
    def test_a_full_run_of_two_alias_ids_leaves_two_intact_products(self):
        base = self._tweet_with_ids("a..b", "a_b")
        with TempDir() as root, self._transcribing():
            results = xsub.process(base, Args(root, no_summary=True))

            self.assertEqual(len(results), 2, "两个卡片都该有产物")
            self.assertEqual(
                len({r.out_dir for r in results}), 2,
                f"两个卡片落进了同一个目录：{[r.out_dir.name for r in results]}",
            )
            on_disk = [d for d in root.iterdir() if d.is_dir()]
            self.assertEqual(len(on_disk), 2, f"磁盘上只剩 {len(on_disk)} 个目录，有产物被覆盖了")
            self.assertEqual(
                sorted(r.media_id for r in results), ["a..b", "a_b"],
                "两份产物必须分别属于两个原始身份",
            )

            tags = []
            for r in results:
                text = (r.out_dir / "transcript.md").read_text(encoding="utf-8")
                self.assertIn(f"媒体ID: {r.media_id}", text, "字幕自报的身份必须是它自己的")
                hit = re.search(r"audio:(card-\d)", text)
                self.assertIsNotNone(hit, f"{r.out_dir.name} 里的字幕不是从它自己的音频转出来的")
                tags.append(hit.group(1))
            self.assertEqual(sorted(tags), ["card-1", "card-2"], "两个目录必须各自装着自己的音频")

    def test_the_old_key_alone_really_did_collide(self):
        """对照组：把三道保护全撤掉，旧症状必须**真的**复现。

        没有这条，"修好了"就只是主张——它证明这组用例真的踩在缺陷上，而不是空转。
        撤的是：①路径键换回"清洗完直接用"；②本轮路径唯一性检查；③目录身份守卫。
        三道都撤掉之后，两个视频挤进一个目录、后一个覆盖前一个，而且两次都报成功。
        """
        base = self._tweet_with_ids("a..b", "a_b")
        with TempDir() as root, self._transcribing(), \
             mock.patch.object(
                 xsub, "media_path_key",
                 side_effect=lambda m, limit=25: xsub.clean_component(m)[:limit]), \
             mock.patch.object(xsub, "guard_distinct_out_dirs", return_value=None), \
             mock.patch.object(xsub, "guard_dir_identity", return_value=None):
            results = xsub.process(base, Args(root, no_summary=True))
            on_disk = [d for d in root.iterdir() if d.is_dir()]
        self.assertEqual(len(results), 2, "旧写法的症状正是：两次都报成功")
        self.assertEqual(len(on_disk), 1, "旧写法必须真的把两个视频挤进同一个目录，否则这组用例是空转")

    def test_the_run_level_guard_alone_would_have_caught_it(self):
        """分层证明：就算路径键退回旧写法，第二道守卫也必须在写盘之前拦住。"""
        base = self._tweet_with_ids("a..b", "a_b")
        with TempDir() as root, self._transcribing(), \
             mock.patch.object(
                 xsub, "media_path_key",
                 side_effect=lambda m, limit=25: xsub.clean_component(m)[:limit]):
            with self.assertRaises(xsub.XsubError) as ctx:
                xsub.process(base, Args(root, no_summary=True))
            self.assertIn("同一个输出目录", str(ctx.exception))
            self.assertEqual(list(root.iterdir()), [], "拦下来时不该留下任何东西")

    # ---- 写盘之前的全局路径唯一性
    def test_a_run_refuses_to_start_when_two_targets_share_a_directory(self):
        """哈希是截短的，理论上仍可能撞。撞了必须在下载之前整条失败，一个字节都不许写。"""
        base = self._tweet_with_ids("a..b", "a_b")
        with TempDir() as root, self._transcribing(), \
             mock.patch.object(xsub, "media_path_key", return_value="collide"):
            with self.assertRaises(xsub.XsubError) as ctx:
                xsub.process(base, Args(root, no_summary=True))
            self.assertIn("同一个输出目录", str(ctx.exception))
            self.assertEqual(list(root.iterdir()), [], "路径撞车时不该在磁盘上留下任何东西")

    # ---- 跨运行：目录里已有别人的产物
    def _seed_foreign_dir(self, root: Path, media_id: str, name: str) -> Path:
        """按当前身份算出目录，再往里塞一份属于**另一个视频**的产物。"""
        info = {"upload_date": "20260101", "uploader_id": "multi", "title": self.LONG_TITLE}
        out_dir = xsub.resolve_out_dir(root, info, self.STATUS, media_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / name).write_text(
            json.dumps({"schema": xsub.CACHE_SCHEMA, "status_id": self.STATUS,
                        "media_id": "someone-else", "file": "audio.m4a", "size": 1}),
            encoding="utf-8",
        )
        (out_dir / "audio.m4a").write_bytes(b"someone-elses-audio")
        return out_dir

    def test_a_directory_holding_another_videos_products_is_never_overwritten(self):
        for name in ("audio.json", "segments.json"):
            with self.subTest(manifest=name), TempDir() as root, self._transcribing():
                base = self._tweet_with_ids("a..b", "a_b")
                foreign = self._seed_foreign_dir(root, "a..b", name)
                with self.assertRaises(xsub.MediaGroupError) as ctx:
                    xsub.process(base, Args(root, no_summary=True))
                self.assertIn("已经有另一个视频的产物", str(ctx.exception))
                self.assertEqual(
                    (foreign / "audio.m4a").read_bytes(), b"someone-elses-audio",
                    "别人的产物被覆盖或删掉了",
                )

    def test_force_does_not_wipe_a_directory_that_belongs_to_another_video(self):
        """--force 是"重下我自己的"，不是"清空这个目录"。"""
        with TempDir() as root, self._transcribing():
            base = self._tweet_with_ids("a..b", "a_b")
            foreign = self._seed_foreign_dir(root, "a..b", "audio.json")
            with self.assertRaises(xsub.MediaGroupError):
                xsub.process(base, Args(root, no_summary=True, force=True))
            self.assertTrue((foreign / "audio.m4a").exists(), "--force 把别人的音频删了")

    def test_a_directory_of_the_same_video_is_still_reusable(self):
        """反向保证：身份一致时缓存照常复用，这道守卫不能把正常复用也拦掉。"""
        base = self._tweet_with_ids("a..b", "a_b")
        with TempDir() as root:
            with self._transcribing():
                first = xsub.process(base, Args(root, no_summary=True))
                again = xsub.process(base, Args(root, no_summary=True))
        self.assertEqual(
            sorted(r.out_dir.name for r in first), sorted(r.out_dir.name for r in again),
            "同一个视频重跑必须落回同一个目录",
        )


# ------------------------------------------- YouTube 支持（任务 2）
class FakeYouTube:
    """按 yt-dlp YoutubeIE 的真实返回形态模拟一个 YouTube 视频。

    照着实测（2026-08-22，yt-dlp 2026.08.19）的字段形态写：
      - info["id"] 是 11 位 base64url 视频 ID，同时也是帖子级身份；
      - uploader_id 是 "@频道handle"（带 @）；
      - automatic_captions 里有 100+ 条轨，其中只有 `<语言>-orig` 是原语言 ASR，
        其余全是机器翻译。
    """

    VIDEO_ID = "dQw4w9WgXcQ"
    OTHER_ID = "jNQXAC9IVRw"

    def __init__(self, **overrides):
        self.info = {
            "id": self.VIDEO_ID,
            "title": "深入浅出 - 第二讲",  # 标题里的 " - " 是内容，不是 "作者 - " 前缀
            "description": "课程简介",
            "uploader": "某频道",
            "uploader_id": "@somechannel",
            "upload_date": "20260101",
            "duration": 600.0,
            "language": "en",
            "formats": [{"url": "https://yt.example/a.m4a", "ext": "m4a"}],
        }
        self.info.update(overrides)
        self.extract_urls: list[str] = []

    def as_module(self):
        import types

        mod = types.ModuleType("yt_dlp")
        outer = self

        class DownloadError(Exception):
            pass

        class YoutubeDL:
            def __init__(self, opts):
                self.opts = opts

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def extract_info(self, url, download=False):
                outer.extract_urls.append(url)
                return dict(outer.info)

            def download(self, urls):
                for _ in urls:
                    pass

            @staticmethod
            def sanitize_info(info):
                return dict(info)

        mod.YoutubeDL = YoutubeDL
        mod.DownloadError = DownloadError
        mod.utils = types.SimpleNamespace(DownloadError=DownloadError)
        return mod


class TestYouTubeUrlParsing(unittest.TestCase):
    VID = "dQw4w9WgXcQ"

    def test_all_single_video_forms_canonicalise_to_one_url(self):
        for raw in (
            f"https://www.youtube.com/watch?v={self.VID}",
            f"http://youtube.com/watch?v={self.VID}",
            f"  https://m.youtube.com/watch?v={self.VID}  ",
            f"youtu.be/{self.VID}",
            f"https://youtu.be/{self.VID}?si=trackingjunk",
            f"https://www.youtube.com/shorts/{self.VID}",
            f"https://www.youtube.com/live/{self.VID}",
            f"https://www.youtube.com/embed/{self.VID}",
            f"https://music.youtube.com/watch?v={self.VID}",
        ):
            parsed = xsub.parse_url(raw)
            self.assertEqual(parsed.platform, xsub.PLATFORM_YOUTUBE, raw)
            self.assertEqual(parsed.url, f"https://www.youtube.com/watch?v={self.VID}", raw)
            self.assertEqual(parsed.post_id, self.VID, raw)
            self.assertIsNone(parsed.media_index, raw)

    def test_playback_position_and_playlist_context_never_enter_identity(self):
        """t= 说的是"你从哪开始看"、list= 说的是"你从哪个列表点进来"，都不是"这是哪个视频"。

        留着它们，同一个视频会因为分享写法不同裂成多个目录、白下载白转写一遍。
        """
        canonical = f"https://www.youtube.com/watch?v={self.VID}"
        for raw in (
            f"https://www.youtube.com/watch?v={self.VID}&t=42s",
            f"https://www.youtube.com/watch?v={self.VID}&list=PLabc123&index=4",
            f"https://youtu.be/{self.VID}?t=90",
            f"https://www.youtube.com/watch?app=desktop&v={self.VID}&pp=xyz",
        ):
            self.assertEqual(xsub.parse_url(raw).url, canonical, raw)

    def test_playlist_only_links_are_rejected_with_a_useful_message(self):
        for raw in (
            "https://www.youtube.com/playlist?list=PLabc123",
            "https://www.youtube.com/playlist?list=PLabc123&index=2",
        ):
            with self.assertRaises(xsub.XsubError, msg=raw) as ctx:
                xsub.parse_url(raw)
            self.assertIn("播放列表", str(ctx.exception), raw)

    def test_non_video_youtube_pages_are_refused(self):
        """频道页/搜索页放行 = 让 generic extractor 去抓一个我们没打算抓的页面，
        而它抓回来的 metadata 会直接进文件路径。"""
        for raw in (
            "https://www.youtube.com/@somechannel",
            "https://www.youtube.com/c/somechannel/videos",
            "https://www.youtube.com/results?search_query=abc",
            "https://www.youtube.com/feed/subscriptions",
            "https://www.youtube.com/watch?v=tooshort",
            "https://www.youtube.com/watch?v=waaaaaaytoolong123",
            "https://www.youtube.com/shorts/nope",
            "https://www.youtube.com/watch",
        ):
            with self.assertRaises(xsub.XsubError, msg=raw):
                xsub.parse_url(raw)

    def test_lookalike_hosts_are_not_youtube(self):
        for raw in (
            "https://youtube.com.evil.example/watch?v=dQw4w9WgXcQ",
            "https://evil.example/youtube.com/watch?v=dQw4w9WgXcQ",
            "https://notyoutube.com/watch?v=dQw4w9WgXcQ",
        ):
            with self.assertRaises(xsub.XsubError, msg=raw):
                xsub.parse_url(raw)

    def test_x_links_still_route_to_the_x_parser_unchanged(self):
        parsed = xsub.parse_url("x.com/greg/status/2090176521335959721/video/2")
        self.assertEqual(parsed.platform, xsub.PLATFORM_X)
        self.assertEqual(parsed.url, "https://x.com/greg/status/2090176521335959721/video/2")
        self.assertEqual(parsed.post_id, "2090176521335959721")
        self.assertEqual(parsed.media_index, 2)

    def test_unknown_sites_are_still_refused(self):
        for raw in ("https://vimeo.com/12345", "file:///etc/passwd", "", "   "):
            with self.assertRaises(xsub.XsubError, msg=raw):
                xsub.parse_url(raw)


class FakeYouTubeCase(unittest.TestCase):
    def setUp(self):
        self.yt = FakeYouTube()
        self._saved = sys.modules.get("yt_dlp")
        sys.modules["yt_dlp"] = self.yt.as_module()

    def tearDown(self):
        if self._saved is None:
            sys.modules.pop("yt_dlp", None)
        else:
            sys.modules["yt_dlp"] = self._saved

    def _run(self, root, raw_url=None, **kw):
        args = Args(root, no_summary=True, **kw)
        with mock.patch.object(xsub.shutil, "which", return_value="/usr/bin/ffmpeg"), \
             mock.patch.object(xsub, "download_audio", return_value=root / "audio.m4a"), \
             mock.patch.object(
                 xsub, "transcribe",
                 side_effect=lambda a, m, l, v: ([{"start": 0.0, "end": 1.0, "text": "本地转写"}], "en"),
             ):
            return xsub.process(raw_url or f"https://youtu.be/{FakeYouTube.VIDEO_ID}", args)


class TestYouTubeProcessing(FakeYouTubeCase):
    def test_a_video_produces_one_output_with_youtube_identity(self):
        with TempDir() as root:
            results = self._run(root)
            self.assertEqual(len(results), 1)
            r = results[0]
            cache = json.loads((r.out_dir / "segments.json").read_text(encoding="utf-8"))
            self.assertEqual(cache["platform"], xsub.PLATFORM_YOUTUBE)
            self.assertEqual(cache["media_id"], FakeYouTube.VIDEO_ID)
            self.assertEqual(cache["status_id"], FakeYouTube.VIDEO_ID)

    def test_identity_carries_the_platform_so_ids_cannot_collide_across_sites(self):
        """身份必须自己说明自己是谁，而不是依赖"X 的 19 位数字和 YouTube 的 11 位串不会撞"。"""
        yt = xsub.MediaTarget(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ", None,
            {"id": "dQw4w9WgXcQ"}, "dQw4w9WgXcQ",
            media_id="dQw4w9WgXcQ", platform=xsub.PLATFORM_YOUTUBE,
        )
        x = xsub.MediaTarget(
            "https://x.com/a/status/dQw4w9WgXcQ", None,
            {"id": "dQw4w9WgXcQ"}, "dQw4w9WgXcQ", media_id="dQw4w9WgXcQ",
        )
        self.assertNotEqual(yt.identity, x.identity, "同 ID 跨平台必须是两个身份")
        self.assertEqual(yt.identity["platform"], xsub.PLATFORM_YOUTUBE)
        self.assertEqual(x.identity["platform"], xsub.PLATFORM_X)

    def test_every_url_form_lands_in_the_same_directory_and_cache(self):
        with TempDir() as root:
            forms = [
                f"https://youtu.be/{FakeYouTube.VIDEO_ID}",
                f"https://www.youtube.com/watch?v={FakeYouTube.VIDEO_ID}&t=90s",
                f"https://m.youtube.com/watch?v={FakeYouTube.VIDEO_ID}&list=PLxyz",
                f"https://www.youtube.com/shorts/{FakeYouTube.VIDEO_ID}",
            ]
            dirs = {self._run(root, f)[0].out_dir for f in forms}
            self.assertEqual(len(dirs), 1, f"同一个视频的不同写法必须收敛到一个目录：{dirs}")
            self.assertEqual(len(list(root.iterdir())), 1)

    def test_transcript_uses_youtube_wording_and_a_single_id_line(self):
        with TempDir() as root:
            md = (self._run(root)[0].out_dir / "transcript.md").read_text(encoding="utf-8")
            self.assertIn(f"- 来源: https://www.youtube.com/watch?v={FakeYouTube.VIDEO_ID}\n", md)
            self.assertIn(f"- 视频ID: {FakeYouTube.VIDEO_ID}\n", md)
            self.assertNotIn("推文ID", md)
            self.assertIn("## 视频简介", md)
            self.assertNotIn("## 推文正文", md)
            self.assertIn("(@somechannel)", md)
            self.assertNotIn("(@@", md, "uploader_id 已带 @，不能再套一个")

    def test_folder_keeps_the_full_title_and_encodes_the_case_mixed_id(self):
        with TempDir() as root:
            name = self._run(root)[0].out_dir.name
            self.assertIn("somechannel", name)
            self.assertIn("20260101", name)
            self.assertIn("深入浅出", name, "YouTube 标题里的 ' - ' 是内容，不该被当作作者前缀剥掉")
            self.assertEqual(name.count(xsub.media_path_key(FakeYouTube.VIDEO_ID)), 1,
                             "帖子 ID 与媒体 ID 相同，目录名里只该出现一次")
            self.assertNotIn(f"_{FakeYouTube.VIDEO_ID}_", name,
                             "大小写混合的 ID 不能原样进目录名（macOS 文件系统大小写不敏感）")

    def test_a_different_video_than_requested_is_refused(self):
        """上游因会员/地区限制返回另一个视频时，绝不能把它的字幕挂在我们以为的身份上。"""
        self.yt.info["id"] = FakeYouTube.OTHER_ID
        with TempDir() as root:
            with self.assertRaises(xsub.XsubError) as ctx:
                self._run(root)
            self.assertIn(FakeYouTube.OTHER_ID, str(ctx.exception))

    def test_a_playlist_result_is_refused(self):
        self.yt.info = {"_type": "playlist", "entries": [{"id": "a"}, {"id": "b"}]}
        with TempDir() as root:
            with self.assertRaises(xsub.XsubError) as ctx:
                self._run(root)
            self.assertIn("播放列表", str(ctx.exception))

    def test_a_video_without_any_media_stream_is_refused(self):
        self.yt.info.pop("formats")
        with TempDir() as root:
            with self.assertRaises(xsub.XsubError):
                self._run(root)

    def test_platform_subtitles_are_not_used_unless_asked(self):
        """默认一律本地转写：不下平台字幕，也不因为"有字幕"就改走别的路径。"""
        self.yt.info["subtitles"] = {"en": [{"ext": "vtt"}]}
        with TempDir() as root:
            with mock.patch.object(xsub, "fetch_native_subs") as fake:
                r = self._run(root)[0]
            fake.assert_not_called()
            cache = json.loads((r.out_dir / "segments.json").read_text(encoding="utf-8"))
            self.assertTrue(cache["source"].startswith("whisper:"), cache["source"])

    def test_native_subs_flag_switches_to_platform_subtitles(self):
        self.yt.info["subtitles"] = {"en": [{"ext": "vtt"}]}
        with TempDir() as root:
            with mock.patch.object(
                xsub, "fetch_native_subs",
                return_value=([{"start": 0.0, "end": 1.0, "text": "平台字幕"}], "en", "平台字幕:en"),
            ) as fake:
                r = self._run(root, native_subs=True)[0]
            fake.assert_called_once()
            cache = json.loads((r.out_dir / "segments.json").read_text(encoding="utf-8"))
            self.assertEqual(cache["source"], "平台字幕:en")

    def test_a_youtube_directory_is_never_reused_by_an_x_media(self):
        """同一个目录里已有别的平台的产物 → 拒绝写入，而不是覆盖掉它。"""
        with TempDir() as root:
            out_dir = self._run(root)[0].out_dir
            with self.assertRaises(xsub.XsubError) as ctx:
                xsub.guard_dir_identity(
                    out_dir,
                    {"schema": xsub.CACHE_SCHEMA, "platform": xsub.PLATFORM_X,
                     "status_id": FakeYouTube.VIDEO_ID, "media_id": FakeYouTube.VIDEO_ID},
                )
            self.assertIn("platform", str(ctx.exception))


class TestRollingAutoCaptions(unittest.TestCase):
    """YouTube 自动字幕是滚动式的：每条 cue 重抄上一条的尾巴，中间夹 10ms 过渡 cue。"""

    ROLLING = (
        "WEBVTT\nKind: captions\nLanguage: en\n\n"
        "00:00:00.320 --> 00:00:18.790 align:start position:0%\n \n[Music]\n\n"
        "00:00:18.790 --> 00:00:18.800 align:start position:0%\n \n \n\n"
        "00:00:18.800 --> 00:00:21.790 align:start position:0%\n \n"
        "First<00:00:19.039><c> half</c><00:00:19.359><c> here</c>\n\n"
        "00:00:21.790 --> 00:00:21.800 align:start position:0%\nFirst half here\n \n\n"
        "00:00:21.800 --> 00:00:25.950 align:start position:0%\nFirst half here\n"
        "second<00:00:22.800><c> half</c><00:00:23.039><c> there.</c>\n"
    )

    def test_rolling_duplicates_and_filler_cues_are_removed(self):
        segs = xsub.parse_vtt(self.ROLLING)
        texts = [s["text"] for s in segs]
        self.assertEqual(texts, ["[Music]", "First half here", "second half there."])
        self.assertAlmostEqual(segs[1]["start"], 18.8)
        self.assertAlmostEqual(segs[2]["start"], 21.8)
        joined = " ".join(texts)
        self.assertEqual(joined.count("First half here"), 1, "同一句话不得出现两遍")

    def test_plain_vtt_is_parsed_exactly_as_before(self):
        """人工字幕一个词级标签都没有，必须完全走老路径、逐字不变。"""
        plain = (
            "WEBVTT\n\n00:00:01.000 --> 00:00:04.500\nHello world\n\n"
            "00:00:04.500 --> 00:00:06.000\nHello world\n"
        )
        segs = xsub.parse_vtt(plain)
        self.assertEqual([s["text"] for s in segs], ["Hello world", "Hello world"],
                         "普通字幕里重复的句子是真实内容，不该被去重")

    def test_a_rolling_track_that_repeats_is_still_collapsed(self):
        """重抄行的真实形状：它自己**不带**词级标签，新内容才带。"""
        rolling_dup = (
            "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nsame<00:00:01.500><c> line</c>\n\n"
            "00:00:02.000 --> 00:00:03.000\nsame line\n"
        )
        segs = xsub.parse_vtt(rolling_dup)
        self.assertEqual(len(segs), 1)
        self.assertAlmostEqual(segs[0]["end"], 3.0, msg="并段时结束时间要延伸到后一段")

    def test_a_repeated_phrase_that_carries_its_own_word_timings_is_kept(self):
        """R2-F04：连着说两遍的同一句话，各带各的词级时间，是真实语音不是重绘。

        词级标签是"这些字此刻正在被说出来"的记号。重抄行从来不带它；带标签的那一行
        永远是本条 cue 的新内容。只按文本相等去重，"No!" "No!" 就会被删掉一句。
        """
        vtt = (
            "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nNo!<00:00:01.500><c> </c>\n\n"
            "00:00:02.000 --> 00:00:03.000\nNo!<00:00:02.500><c> </c>\n"
        )
        segs = xsub.parse_vtt(vtt)
        self.assertEqual([s["text"] for s in segs], ["No!", "No!"],
                         "两次都带自己的词级时间 → 两次都是真实语音")
        self.assertAlmostEqual(segs[0]["end"], 2.0)
        self.assertAlmostEqual(segs[1]["start"], 2.0)


# =========================================================== 第 1 轮审查修复回归
# 每个 class 对应一条 MUST_FIX。命名里带 finding id，便于日后回溯"这测试在防什么"。


class TestR2F01CacheKnowsSubtitleSource(FakeYouTubeCase):
    """F01：字幕来源策略必须进缓存身份，否则换来源跑会命中上一次的缓存。"""

    def _native(self, *a, **kw):
        return [{"start": 0.0, "end": 1.0, "text": "平台字幕内容"}], "en", "平台字幕:en"

    def _source(self, out_dir):
        return json.loads((out_dir / "segments.json").read_text(encoding="utf-8"))["source"]

    def test_switching_to_native_subs_is_not_served_from_the_local_cache(self):
        with TempDir() as root:
            first = self._run(root)[0]
            self.assertTrue(self._source(first.out_dir).startswith("whisper:"))
            with mock.patch.object(xsub, "fetch_native_subs", side_effect=self._native):
                second = self._run(root, native_subs=True)[0]
            self.assertEqual(first.out_dir, second.out_dir, "同一个视频应落回同一个目录")
            self.assertEqual(
                self._source(second.out_dir), "平台字幕:en",
                "加了 --native-subs 却拿到上一次的本地转写缓存——策略没进身份",
            )

    def test_switching_back_to_local_is_not_served_from_the_native_cache(self):
        with TempDir() as root:
            with mock.patch.object(xsub, "fetch_native_subs", side_effect=self._native):
                first = self._run(root, native_subs=True)[0]
            self.assertEqual(self._source(first.out_dir), "平台字幕:en")
            second = self._run(root)[0]
            self.assertTrue(
                self._source(second.out_dir).startswith("whisper:"),
                "去掉 --native-subs 却拿到上一次的平台字幕缓存",
            )

    def test_identity_is_a_miss_when_only_the_policy_differs(self):
        base = {"platform": "youtube", "media_id": "x", "model": "m", "lang_request": None}
        with TempDir() as d:
            cache = d / "segments.json"
            xsub.write_atomic(cache, json.dumps(
                {**base, "subtitle_policy": xsub.SUBTITLE_POLICY_LOCAL_ONLY,
                 "language": "en", "source": "whisper:x",
                 "segments": [{"start": 0, "end": 1, "text": "hi"}]},
                ensure_ascii=False,
            ))
            self.assertIsNotNone(xsub.load_segments_cache(
                cache, {**base, "subtitle_policy": xsub.SUBTITLE_POLICY_LOCAL_ONLY}))
            self.assertIsNone(
                xsub.load_segments_cache(
                    cache, {**base, "subtitle_policy": xsub.SUBTITLE_POLICY_NATIVE_FIRST}),
                "只有字幕来源策略不同也必须判 miss",
            )


class TestR2F02DefaultsAreePerPlatform(unittest.TestCase):
    """F02：X 的默认必须保持"平台字幕优先"（冻结基线的行为），只有 YouTube 默认本地转写。"""

    def test_policy_matrix(self):
        cases = {
            (xsub.PLATFORM_X, False): xsub.SUBTITLE_POLICY_NATIVE_FIRST,
            (xsub.PLATFORM_X, True): xsub.SUBTITLE_POLICY_NATIVE_FIRST,
            (xsub.PLATFORM_YOUTUBE, False): xsub.SUBTITLE_POLICY_LOCAL_ONLY,
            (xsub.PLATFORM_YOUTUBE, True): xsub.SUBTITLE_POLICY_NATIVE_FIRST,
        }
        for (platform, flag), want in cases.items():
            self.assertEqual(xsub.subtitle_policy(platform, flag), want, f"{platform} native={flag}")

    def test_native_subs_flag_does_not_invalidate_the_x_cache(self):
        """X 上带不带 --native-subs 是同一件事，不该因此白白重跑一遍转写。"""
        self.assertEqual(
            xsub.subtitle_policy(xsub.PLATFORM_X, False),
            xsub.subtitle_policy(xsub.PLATFORM_X, True),
        )


class TestR2F02XStillPrefersPlatformSubs(FakeTwitterCase):
    def test_x_default_run_consults_platform_subtitles(self):
        with TempDir() as root:
            spy = mock.Mock(side_effect=self._subs_per_media)
            with mock.patch.object(xsub, "fetch_native_subs", spy):
                results = xsub.process(FakeTwitter.BASE, Args(root, no_summary=True))
            self.assertTrue(spy.called, "X 默认必须先看平台字幕（冻结基线行为）")
            for r in results:
                cache = json.loads((r.out_dir / "segments.json").read_text(encoding="utf-8"))
                self.assertEqual(cache["source"], "平台字幕:en")
                self.assertEqual(cache["subtitle_policy"], xsub.SUBTITLE_POLICY_NATIVE_FIRST)


class TestR2F02YouTubeDefaultsToLocal(FakeYouTubeCase):
    def test_youtube_default_run_never_touches_platform_subtitles(self):
        self.yt.info["automatic_captions"] = {"en-orig": [{}], "ja": [{}]}
        with TempDir() as root:
            spy = mock.Mock(return_value=None)
            with mock.patch.object(xsub, "fetch_native_subs", spy):
                r = self._run(root)[0]
            self.assertFalse(spy.called, "YouTube 默认应直接本地转写")
            cache = json.loads((r.out_dir / "segments.json").read_text(encoding="utf-8"))
            self.assertEqual(cache["subtitle_policy"], xsub.SUBTITLE_POLICY_LOCAL_ONLY)


class TestR2F03AutoCaptionsMustBeTheOriginalTrack(unittest.TestCase):
    """F03：YouTube 的 automatic_captions 是"一条原件 + 一百多条机翻"的扇出，
    只有 `<语言>-orig` 是原件，同名的 `en` 轨可能是译件，不能当原文用。

    第 2 轮加严：这条限制按**平台**生效，不再看表里当时有没有 -orig。"""

    YT = xsub.PLATFORM_YOUTUBE
    JA_VIDEO = {"language": "ja", "automatic_captions": {"en": [{}], "fr": [{}], "ja-orig": [{}]}}

    def test_requesting_a_translated_language_falls_back_to_whisper(self):
        self.assertIsNone(
            xsub.pick_native_lang(self.JA_VIDEO, "en", self.YT),
            "en 轨是从日语机翻过去的，拿它当英文字幕就是二手转述",
        )

    def test_requesting_the_original_language_still_works(self):
        self.assertEqual(
            xsub.pick_native_lang(self.JA_VIDEO, "ja", self.YT), ("ja-orig", "平台自动字幕")
        )

    def test_spoken_language_is_used_when_no_lang_is_given(self):
        self.assertEqual(
            xsub.pick_native_lang(self.JA_VIDEO, None, self.YT), ("ja-orig", "平台自动字幕")
        )

    def test_a_youtube_auto_table_without_orig_is_refused_not_downgraded(self):
        """R2-F03：门按平台开关，不按"表里有没有 -orig"这个运行期签名。

        yt-dlp 版本、地区、视频年代都会改变那张表的形状。按签名判定，一旦某次
        返回的表里只剩普通的 `en`，门就自己开了——而那条 `en` 完全可能是机翻。
        """
        for table in (
            {"en": [{}]},
            {"en": [{}], "en-US": [{}]},
            {"en-US": [{}], "fr": [{}], "zh-Hans": [{}]},
        ):
            info = {"language": "en", "automatic_captions": table}
            self.assertIsNone(
                xsub.pick_native_lang(info, None, self.YT),
                f"{sorted(table)} 里没有 en-orig，YouTube 自动字幕就该退回本地转写",
            )
            self.assertIsNone(xsub.pick_native_lang(info, "en", self.YT), sorted(table))

    def test_the_same_table_on_x_keeps_the_old_lenient_matching(self):
        """同一张表在 X 上必须逐字保持旧行为——严格规则只对 YouTube。"""
        x_like = {"language": "en", "automatic_captions": {"en-US": [{}]}}
        self.assertEqual(xsub.pick_native_lang(x_like, "en"), ("en-US", "平台自动字幕"))
        self.assertEqual(
            xsub.pick_native_lang(x_like, "en", xsub.PLATFORM_X), ("en-US", "平台自动字幕")
        )

    def test_manual_subtitles_are_unaffected_by_the_orig_rule(self):
        info = {"language": "ja", "subtitles": {"en": [{}]},
                "automatic_captions": {"en": [{}], "ja-orig": [{}]}}
        self.assertEqual(xsub.pick_native_lang(info, "en", self.YT), ("en", "平台字幕"),
                         "人工字幕是创作者上传的，不受机翻扇出规则影响")

    def test_orig_still_outranks_a_same_name_track_when_not_strict(self):
        """兜底优先级：即使调用方没开 orig_only，原语言轨也必须排在同名轨前面。"""
        self.assertEqual(xsub.match_lang_track(["en", "en-US", "en-orig"], "en"), "en-orig")

    def test_match_lang_track_orig_only_refuses_to_downgrade(self):
        langs = ["en", "en-US", "ja-orig"]
        self.assertIsNone(xsub.match_lang_track(langs, "en", orig_only=True))
        self.assertEqual(xsub.match_lang_track(langs, "en", orig_only=False), "en")


class TestR2F04CueBoundaries(unittest.TestCase):
    """F04：cue 分界必须按时间戳判，不能按空行判；滚动去重必须按内容 + 时间连续性判。"""

    def test_a_space_only_separator_line_does_not_swallow_the_next_cue(self):
        vtt = ("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nAlpha\n \n"
               "00:00:03.000 --> 00:00:04.000\nBravo\n")
        self.assertEqual([s["text"] for s in xsub.parse_vtt(vtt)], ["Alpha", "Bravo"])

    def test_a_missing_blank_line_does_not_swallow_the_next_cue(self):
        vtt = ("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nAlpha\n"
               "00:00:03.000 --> 00:00:04.000\nBravo\n")
        segs = xsub.parse_vtt(vtt)
        self.assertEqual([s["text"] for s in segs], ["Alpha", "Bravo"])
        self.assertAlmostEqual(segs[1]["start"], 3.0)

    def test_untagged_payload_in_a_tagged_cue_is_kept(self):
        """同一条 cue 里没打词级标签的行也可能是真正文，不能因为"没标签"就丢掉。"""
        vtt = ("WEBVTT\n\n00:00:01.000 --> 00:00:05.000\n"
               "genuine untagged sentence\ntagged<00:00:02.000><c> words</c>\n")
        self.assertEqual(
            [s["text"] for s in xsub.parse_vtt(vtt)],
            ["genuine untagged sentence tagged words"],
        )

    def test_a_repeat_across_a_time_gap_is_real_speech_not_a_repaint(self):
        vtt = ("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nAlpha<00:00:01.500><c> x</c>\n\n"
               "00:00:05.000 --> 00:00:07.000\nAlpha x\n")
        segs = xsub.parse_vtt(vtt)
        self.assertEqual([s["text"] for s in segs], ["Alpha x", "Alpha x"],
                         "隔着 3 秒空档又说了一遍，不是滚动重抄")
        self.assertAlmostEqual(segs[0]["end"], 2.0, msg="不得把结束时间拉过空档")

    def test_cue_identifiers_are_not_treated_as_payload(self):
        vtt = ("WEBVTT\n\ncue-1\n00:00:01.000 --> 00:00:02.000\nAlpha\n\n"
               "cue-2\n00:00:03.000 --> 00:00:04.000\nBravo\n")
        self.assertEqual([s["text"] for s in xsub.parse_vtt(vtt)], ["Alpha", "Bravo"])

    def test_a_real_rolling_stream_emits_each_phrase_once(self):
        """照抄真实 YouTube 滚动字幕的形状：重抄行 + 10ms 过渡 cue + 无标签的新内容。"""
        vtt = (
            "WEBVTT\nKind: captions\nLanguage: en\n\n"
            "00:00:03.879 --> 00:00:05.070\n \ndo\n\n"
            "00:00:05.070 --> 00:00:05.080\ndo\n \n\n"
            "00:00:05.080 --> 00:00:07.709\ndo\nit<00:00:06.080><c> just</c>\n\n"
            "00:00:07.709 --> 00:00:07.719\nit just\n \n\n"
            "00:00:07.719 --> 00:00:10.830\nit just\ndo it\n"
        )
        segs = xsub.parse_vtt(vtt)
        self.assertEqual([s["text"] for s in segs], ["do", "it just", "do it"],
                         "重抄行要去掉，没打标签的新内容（末条）要留下")


class TestR2F05PlaylistEmbedIsRefused(unittest.TestCase):
    """F05：/embed/videoseries?list=... 里 "videoseries" 恰好是 11 位合法字符，
    会原样通过 ID 形态校验，被当成真视频 ID 送去 yt-dlp。"""

    def test_videoseries_embed_is_refused_as_a_playlist(self):
        for raw in (
            "https://www.youtube.com/embed/videoseries?list=PLabcdefghijklmnop",
            "https://www.youtube-nocookie.com/embed/videoseries?list=PLabcdefghijklmnop",
            "https://youtube.com/embed/videoseries",
        ):
            with self.assertRaises(xsub.XsubError, msg=raw) as ctx:
                xsub.parse_url(raw)
            self.assertIn("播放列表", str(ctx.exception), raw)

    def test_videoseries_never_reaches_yt_dlp(self):
        called = []
        with mock.patch.object(xsub, "fetch_info", side_effect=lambda *a, **k: called.append(a)):
            with self.assertRaises(xsub.XsubError):
                xsub.parse_url("https://www.youtube.com/embed/videoseries?list=PLxx")
        self.assertEqual(called, [], "拒绝必须发生在任何网络调用之前")


class TestR2F06XDirectoryNamesAreUnchanged(unittest.TestCase):
    """F06："帖子 ID 与媒体 ID 相同就只写一次"这条只能对 YouTube 开。
    X 上两者也可能相等，省掉一次会让所有既有目录改名、缓存全失效。"""

    INFO = {"title": "作者 - 正文摘要", "uploader_id": "someone", "upload_date": "20260101"}

    def test_x_keeps_the_duplicated_post_id(self):
        """冻结基线对 X 无条件拼帖子 ID，两者相等时目录名就是 ..._555_555。"""
        with TempDir() as root:
            got = xsub.resolve_out_dir(root, self.INFO, "555", "555", xsub.PLATFORM_X).name
        self.assertTrue(got.endswith("_555_555"), f"X 目录名必须与冻结基线逐字一致: {got}")

    def test_youtube_writes_the_id_once(self):
        info = {"title": "深入浅出 - 第二讲", "uploader_id": "@ch", "upload_date": "20260101"}
        vid = "dQw4w9WgXcQ"
        with TempDir() as root:
            got = xsub.resolve_out_dir(root, info, vid, vid, xsub.PLATFORM_YOUTUBE).name
        self.assertEqual(got.count(vid), 1, f"YouTube 目录名不该重复写 ID: {got}")

    def test_x_with_distinct_ids_is_unchanged_too(self):
        with TempDir() as root:
            got = xsub.resolve_out_dir(root, self.INFO, "555", "m-aaa", xsub.PLATFORM_X).name
        self.assertTrue(got.endswith("_555_" + xsub.media_path_key("m-aaa")), got)


class TestR2F07AdmissionGate(FakeYouTubeCase):
    """F07：直播和超长视频必须在**下载之前**拒掉。"""

    def _expect_refusal(self, root, needle, **info):
        self.yt.info.update(info)
        with self.assertRaises(xsub.XsubError) as ctx:
            self._run(root)
        self.assertIn(needle, str(ctx.exception))
        self.assertEqual(list(root.iterdir()), [], "拒绝之前不该建任何输出目录")

    def test_a_live_stream_is_refused(self):
        with TempDir() as root:
            self._expect_refusal(root, "直播", is_live=True)

    def test_an_upcoming_premiere_is_refused(self):
        with TempDir() as root:
            self._expect_refusal(root, "直播", live_status="is_upcoming")

    def test_a_video_over_the_cap_is_refused_with_the_escape_hatch_named(self):
        with TempDir() as root:
            self._expect_refusal(root, "--allow-long-video", duration=xsub.MAX_DURATION_SEC + 1)

    def test_exactly_at_the_cap_is_allowed(self):
        self.yt.info["duration"] = xsub.MAX_DURATION_SEC
        with TempDir() as root:
            self.assertEqual(len(self._run(root)), 1)

    def test_the_escape_hatch_lets_a_long_video_through(self):
        self.yt.info["duration"] = xsub.MAX_DURATION_SEC + 1
        with TempDir() as root:
            self.assertEqual(len(self._run(root, allow_long_video=True)), 1)

    def test_an_unknown_duration_is_refused_before_any_download(self):
        """R2-F07：拿不到时长就放行，等于把上限当不存在。

        duration 缺失/非数值/NaN/<=0 全部按"判断不了"处理，一律 fail-closed，
        并且要在**下载之前**就拒——断言 download_audio 一次都没被调用。
        """
        for dur in (None, "3600", float("nan"), float("inf"), 0, -1, True):
            with self.subTest(duration=dur):
                self.yt.info.pop("duration", None)
                if dur is not None:
                    self.yt.info["duration"] = dur
                with mock.patch.object(xsub, "download_audio") as dl, TempDir() as root:
                    with self.assertRaises(xsub.XsubError) as ctx:
                        self._run(root)
                    self.assertIn("拿不到这个视频的时长", str(ctx.exception))
                    self.assertIn("--allow-long-video", str(ctx.exception))
                    dl.assert_not_called()
                    self.assertEqual(list(root.iterdir()), [], "拒绝之前不该建任何输出目录")

    def test_the_escape_hatch_does_not_override_an_unknown_duration(self):
        """R3-F07：给未知时长开放行口，就是给一条没有上限的路。

        硬上限比的是 duration，而 duration 正是拿不到的那个数——放行之后没有任何
        可执行的兜底（下载和转写两层都没有独立于 metadata 的字节数或墙钟限制）。
        所以 --allow-long-video 对未知时长必须无效，且必须在下载之前就拒。
        """
        for dur in (None, "3600", float("nan"), float("inf"), 0, -1, True):
            with self.subTest(duration=dur):
                self.yt.info.pop("duration", None)
                if dur is not None:
                    self.yt.info["duration"] = dur
                with mock.patch.object(xsub, "download_audio") as dl, TempDir() as root:
                    with self.assertRaises(xsub.XsubError) as ctx:
                        self._run(root, allow_long_video=True)
                    self.assertIn("拿不到这个视频的时长", str(ctx.exception))
                    dl.assert_not_called()
                    self.assertEqual(list(root.iterdir()), [], "拒绝之前不该建任何输出目录")

    def test_the_gate_says_plainly_that_the_escape_hatch_will_not_help(self):
        """报错文案不能把用户支到一个其实无效的开关上。"""
        self.yt.info.pop("duration", None)
        with TempDir() as root:
            with self.assertRaises(xsub.XsubError) as ctx:
                self._run(root)
        msg = str(ctx.exception)
        self.assertIn("--allow-long-video", msg)
        self.assertIn("不放行", msg)

    def test_the_hard_cap_holds_even_with_the_escape_hatch(self):
        """R2-F07：--allow-long-video 是"我知道它长"，不是"多长都行"。"""
        self.yt.info["duration"] = xsub.HARD_MAX_DURATION_SEC + 1
        with mock.patch.object(xsub, "download_audio") as dl, TempDir() as root:
            with self.assertRaises(xsub.XsubError) as ctx:
                self._run(root, allow_long_video=True)
            self.assertIn("硬上限", str(ctx.exception))
            dl.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])

    def test_exactly_at_the_hard_cap_is_allowed_with_the_escape_hatch(self):
        self.yt.info["duration"] = xsub.HARD_MAX_DURATION_SEC
        with TempDir() as root:
            self.assertEqual(len(self._run(root, allow_long_video=True)), 1)

    def test_the_hard_cap_is_strictly_above_the_soft_cap(self):
        self.assertGreater(xsub.HARD_MAX_DURATION_SEC, xsub.MAX_DURATION_SEC)


class TestR2F08VideoIdShape(unittest.TestCase):
    """F08：^...$ 的 $ 会在结尾换行前匹配，带尾随控制字符的 ID 能蒙混过关。"""

    def test_a_trailing_newline_is_refused(self):
        for raw in (
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ%0A",
            "https://youtu.be/dQw4w9WgXcQ%0A",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ%0D",
        ):
            with self.assertRaises(xsub.XsubError, msg=raw):
                xsub.parse_url(raw)

    def test_the_regex_itself_is_anchored_at_both_ends(self):
        self.assertIsNone(xsub.YT_ID_RE.fullmatch("dQw4w9WgXcQ\n"))
        self.assertIsNotNone(xsub.YT_ID_RE.fullmatch("dQw4w9WgXcQ"))


# ================================================= 第 2 轮审查修复回归（R2 复审）


class TestR3F02XTrackSelectionIsFrozen(unittest.TestCase):
    """R2-F02：X 的取轨规则必须与冻结基线**逐字相同**。

    第 1 轮为治 YouTube 的机翻扇出而收紧了 pick_native_lang，但那次收紧是**平台无关**的，
    连带改掉了 X 的行为：没有语言依据时冻结基线取 sorted(langs)[0]，收紧后返回 None
    → 一条本来直接用平台字幕出结果的推文，改成跑本地 Whisper。

    下面这张表逐条抄自冻结基线 593c15d 的算法，是行为契约本身，不是"当前实现的快照"。
    """

    # (info, want, 冻结基线的返回值)
    ORACLE = [
        # 无 --lang、无 language：人工字幕表取字母序第一条
        ({"subtitles": {"zh": [{}], "en": [{}], "ar": [{}]}}, None, ("ar", "平台字幕")),
        # 无 --lang、有 language 且表里有这条：取它
        ({"subtitles": {"zh": [{}], "en": [{}]}, "language": "zh"}, None, ("zh", "平台字幕")),
        # 无 --lang、有 language 但表里没有：仍退回字母序第一条（基线不做前缀匹配）
        ({"subtitles": {"zh": [{}], "en": [{}]}, "language": "fr"}, None, ("en", "平台字幕")),
        # 人工字幕表为空 → 落到自动字幕表，同样的规则
        ({"subtitles": {}, "automatic_captions": {"ja": [{}], "en": [{}]}}, None,
         ("en", "平台自动字幕")),
        # 指定 --lang：精确优先
        ({"subtitles": {"en-US": [{}], "en": [{}]}}, "en", ("en", "平台字幕")),
        # 指定 --lang：没有精确匹配时用同语族前缀
        ({"subtitles": {"en-US": [{}], "fr": [{}]}}, "en", ("en-US", "平台字幕")),
        # 指定 --lang：人工字幕表里没有就**跳到**自动字幕表继续找
        ({"subtitles": {"fr": [{}]}, "automatic_captions": {"en": [{}]}}, "en",
         ("en", "平台自动字幕")),
        # 指定 --lang：两张表都没有 → None（交给本地转写）
        ({"subtitles": {"fr": [{}]}, "automatic_captions": {"de": [{}]}}, "en", None),
        # -orig 在 X 上没有特殊含义：基线不认它，字母序里 en 排在 en-orig 前面
        ({"subtitles": {"en-orig": [{}], "en": [{}]}}, None, ("en", "平台字幕")),
        ({"subtitles": {"en-orig": [{}], "en": [{}]}}, "en", ("en", "平台字幕")),
        # 两张表都空/缺 → None
        ({}, None, None),
        ({"subtitles": {}, "automatic_captions": {}}, "en", None),
    ]

    def test_every_frozen_case_still_holds_on_x(self):
        for info, want, expected in self.ORACLE:
            with self.subTest(info=info, want=want):
                self.assertEqual(xsub.pick_native_lang(info, want), expected)
                self.assertEqual(
                    xsub.pick_native_lang(info, want, xsub.PLATFORM_X), expected
                )

    def test_the_default_platform_is_x(self):
        """省略 platform 就是 X：老调用点不写平台也必须落在冻结行为上。"""
        import inspect

        self.assertEqual(
            inspect.signature(xsub.pick_native_lang).parameters["platform"].default,
            xsub.PLATFORM_X,
        )

    def test_youtube_is_the_only_platform_that_gets_the_strict_rule(self):
        """严格规则只挂在 YouTube 上，别的平台值一律走冻结路径。"""
        info = {"subtitles": {"zh": [{}], "ar": [{}]}}
        self.assertIsNone(xsub.pick_native_lang(info, None, xsub.PLATFORM_YOUTUBE))
        for other in (xsub.PLATFORM_X, "vimeo", ""):
            self.assertEqual(
                xsub.pick_native_lang(info, None, other), ("ar", "平台字幕"), other
            )


class TestR3F02XEndToEndReallyUsesPlatformSubs(FakeTwitterCase):
    """R2-F02：不打桩 fetch_native_subs，让真实取轨 + 真实字幕下载整条跑通。

    此前 X 的"仍优先平台字幕"只由一个 spy 断言"被调用过"——那证明不了取轨的结果对不对，
    正是因此第 1 轮把 X 的取轨改坏了也没有测试报警。这里让 transcribe 直接抛异常：
    只要退回了本地转写就是失败，字幕内容和来源都拿真实产物核对。
    """

    def _run(self, root, **kw):
        args = Args(root, no_summary=True, **kw)
        with mock.patch.object(xsub.shutil, "which", return_value="/usr/bin/ffmpeg"), \
             mock.patch.object(
                 xsub, "transcribe",
                 side_effect=lambda *a: (_ for _ in ()).throw(
                     AssertionError("X 有平台字幕就不该退回本地转写")
                 ),
             ):
            return xsub.process(FakeTwitter.BASE + "/video/1", args)

    def test_x_default_run_downloads_and_uses_the_platform_track(self):
        self.tw.ENTRY_EXTRA = {"subtitles": {"en": [{}]}}
        with TempDir() as root:
            r = self._run(root)[0]
            cache = json.loads((r.out_dir / "segments.json").read_text(encoding="utf-8"))
            self.assertEqual(cache["source"], "平台字幕:en")
            self.assertIn("字幕属于 m-aaa", (r.out_dir / "transcript.md").read_text("utf-8"))

    def test_with_no_language_evidence_x_still_takes_the_first_track(self):
        """冻结基线行为：三条人工轨、没有 language，取字母序第一条（ar），不退回转写。"""
        self.tw.ENTRY_EXTRA = {"subtitles": {"ar": [{}], "en": [{}], "ja": [{}]}}
        with TempDir() as root:
            r = self._run(root)[0]
            cache = json.loads((r.out_dir / "segments.json").read_text(encoding="utf-8"))
            self.assertEqual(cache["source"], "平台字幕:ar")
            self.assertEqual(cache["language"], "ar")

    def test_native_subs_flag_is_a_no_op_on_x(self):
        """X 上 --native-subs 与默认完全等价，产物应当一模一样。"""
        self.tw.ENTRY_EXTRA = {"subtitles": {"en": [{}]}}
        with TempDir() as a, TempDir() as b:
            plain = json.loads(
                (self._run(a)[0].out_dir / "segments.json").read_text("utf-8")
            )
            flagged = json.loads(
                (self._run(b, native_subs=True)[0].out_dir / "segments.json").read_text("utf-8")
            )
            self.assertEqual(plain, flagged)


class TestR3F04OrdinaryVttIsParsedExactlyAsBefore(unittest.TestCase):
    """R2-F04：普通（非滚动）字幕的解析必须与冻结基线**逐字相同**。

    ORACLE 里的期望值抄自冻结基线 593c15d 的 parse_vtt，包括它保留行内多余空白这一点。
    """

    # (vtt, 冻结基线的 [(start, end, text)])
    ORACLE = [
        ("WEBVTT\n\n00:00:01.000 --> 00:00:04.500\nHello world\n",
         [(1.0, 4.5, "Hello world")]),
        # 重复的句子在普通字幕里是真实内容，不去重
        ("WEBVTT\n\n00:00:01.000 --> 00:00:04.500\nHello\n\n"
         "00:00:04.500 --> 00:00:06.000\nHello\n",
         [(1.0, 4.5, "Hello"), (4.5, 6.0, "Hello")]),
        # 多行正文用单个空格接起来
        ("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nfirst\nsecond\n",
         [(1.0, 2.0, "first second")]),
        # 行内/行首行尾的空白照原样留着——那是字幕作者排的版
        ("WEBVTT\n\n00:00:01.000 --> 00:00:05.000\n   hello   world   \n"
         "   second   line   \n",
         [(1.0, 5.0, "hello   world       second   line")]),
        # 标签去掉，正文留下
        ("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n<v Bob>hi there</v>\n",
         [(1.0, 2.0, "hi there")]),
        # 短时间戳写法（mm:ss.mmm）与逗号小数点
        ("WEBVTT\n\n00:01.000 --> 00:02.000\nshort form\n",
         [(1.0, 2.0, "short form")]),
        ("WEBVTT\n\n00:00:01,000 --> 00:00:02,000\ncomma decimal\n",
         [(1.0, 2.0, "comma decimal")]),
        # cue settings 跟在时间戳后面
        ("WEBVTT\n\n00:00:01.000 --> 00:00:02.000 align:start position:0%\npositioned\n",
         [(1.0, 2.0, "positioned")]),
        # cue 标识符不是正文
        ("WEBVTT\n\ncue-1\n00:00:01.000 --> 00:00:02.000\nAlpha\n\n"
         "cue-2\n00:00:03.000 --> 00:00:04.000\nBravo\n",
         [(1.0, 2.0, "Alpha"), (3.0, 4.0, "Bravo")]),
        # 空正文的 cue 不产出段
        ("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n\n"
         "00:00:03.000 --> 00:00:04.000\nonly this\n",
         [(3.0, 4.0, "only this")]),
    ]

    def test_the_differential_corpus_matches_the_frozen_behaviour(self):
        for vtt, expected in self.ORACLE:
            with self.subTest(vtt=vtt):
                got = [(s["start"], s["end"], s["text"]) for s in xsub.parse_vtt(vtt)]
                self.assertEqual(got, expected)

    def test_timestamp_shaped_text_in_the_body_is_not_a_cue_boundary(self):
        """R2-F04：正文里出现时间戳形状的文本是合法的，不能用 search() 去认边界。

        用 search() 会把这一行当成新 cue 的开头：cue 从中间劈开，
        前半段的正文连同它自己的时间一起消失。
        """
        vtt = ("WEBVTT\n\n00:00:01.000 --> 00:00:05.000\n"
               "he said 00:00:02.000 --> 00:00:03.000 was the timecode\n"
               "and then left\n")
        segs = xsub.parse_vtt(vtt)
        self.assertEqual(len(segs), 1, "整条 cue 必须完整，不能被正文里的时间码劈开")
        self.assertEqual(segs[0]["start"], 1.0)
        self.assertEqual(segs[0]["end"], 5.0)
        self.assertIn("he said", segs[0]["text"])
        self.assertIn("and then left", segs[0]["text"])

    def test_a_timing_pair_must_start_the_line(self):
        """时间行的判据：这一行**以**时间戳对开头，后面最多再跟 cue settings。

        "开头"是关键那一半。正文里的时间码总是跟在别的字后面（"他说 00:00:02.000 -->…"），
        锚在行首就认不上；而一行本身就以时间戳对开头时，按 WebVTT 规范它**就是**时间行——
        settings 的解析是宽容的（不认识的设置忽略掉），所以后面跟什么都不改变这个判定。
        """
        for ok in (
            "00:00:01.000 --> 00:00:02.000",
            "00:00:01.000 --> 00:00:02.000 align:start position:0%",
            "00:00:01.000 --> 00:00:02.000  ",
            "  00:00:01.000 --> 00:00:02.000",
            "00:01.000 --> 00:02.000",
            "00:00:01,000 --> 00:00:02,000",
        ):
            self.assertIsNotNone(xsub.VTT_TS_LINE.fullmatch(ok), ok)
        for bad in (
            "he said 00:00:01.000 --> 00:00:02.000 was the timecode",
            "x 00:00:01.000 --> 00:00:02.000",
            "00:00:01.000",
            "WEBVTT",
            "",
        ):
            self.assertIsNone(xsub.VTT_TS_LINE.fullmatch(bad), bad)


class TestR3F09VideoseriesOnlyBlocksTheEmbedForm(unittest.TestCase):
    """R2-F09：第 1 轮把 "videoseries" 无条件拒了，但它只在 /embed/ 下有播放列表含义。

    "videoseries" 同时也是一个形态完全合法的 11 位视频 ID，无条件拒会把
    /watch?v=videoseries 挡在门外，还配一句"这是 /embed/ 链接"的错误说明。
    """

    def test_the_embed_form_is_still_refused(self):
        for raw in (
            "https://www.youtube.com/embed/videoseries?list=PLabcdefghijklmnop",
            "https://www.youtube-nocookie.com/embed/videoseries?list=PLabcdefghijklmnop",
            "https://youtube.com/embed/videoseries",
            "https://www.youtube.com/embed/VideoSeries?list=PLabcdefghijklmnop",
        ):
            with self.assertRaises(xsub.XsubError, msg=raw) as ctx:
                xsub.parse_url(raw)
            self.assertIn("播放列表", str(ctx.exception))

    def test_every_non_embed_form_accepts_it_as_an_ordinary_id(self):
        """/embed/ 之外的每一种路径写法都必须放行。

        /shorts/、/live/、/v/ 这三种是关键：它们和 /embed/ 一样有两段路径，
        所以只有它们能检验出"限定"这件事真的落在 segments[0] == "embed" 上，
        而不是被前面那句 len(segments) >= 2 顺手挡掉的假象。
        """
        for raw in (
            "https://www.youtube.com/watch?v=videoseries",
            "https://youtu.be/videoseries",
            "https://www.youtube.com/shorts/videoseries",
            "https://www.youtube.com/live/videoseries",
            "https://www.youtube.com/v/videoseries",
        ):
            parsed = xsub.parse_url(raw)
            self.assertEqual(parsed.post_id, "videoseries", raw)
            self.assertEqual(parsed.platform, xsub.PLATFORM_YOUTUBE, raw)

    def test_an_ordinary_id_is_untouched(self):
        """正对照：普通 11 位 ID 一路畅通，拒绝逻辑没有误伤面。"""
        for raw, vid in (
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/shorts/abc-DEF_123", "abc-DEF_123"),
        ):
            self.assertEqual(xsub.parse_url(raw).post_id, vid, raw)


# ================================================= 第 3 轮审查修复回归（R3 复审）


class TestR3F02TheUserFacingContractCannotDrift(unittest.TestCase):
    """R3-F02：代码改对了，但用户实际读到的那份契约漂了。

    第 3 轮审查指出：模块 docstring 和 README 都已按平台分开写，唯独 argparse 的
    `--native-subs` 一行仍写着"默认关闭：一律本地 Whisper 转写"——那是**上一个已被
    推翻的方案**留下的文字，与 X 默认走平台字幕直接相反。用户读 `--help` 形成的预期
    和真实产物来源不一致，这本身就是缺陷，跟代码对不对无关。

    所以这里锁的不是实现，是三处面向用户的文字**互相之间不许漂**：
    argparse help、模块 docstring、README。
    """

    STALE = "一律本地 Whisper"

    @staticmethod
    def _squash(text: str) -> str:
        """把空白全部抹掉再比。

        argparse 按 COLUMNS 折行，同一句话在窄终端里会被换行加缩进劈成两截；
        逐字比对会因为跑测试的终端宽度而时绿时红。这里比的是"这句话在不在"，
        不是"它排版成什么样"。
        """
        return re.sub(r"\s+", "", text)

    def setUp(self):
        self.help = xsub.build_parser().format_help()
        self.doc = xsub.__doc__ or ""
        self.readme = (Path(xsub.__file__).resolve().parent / "README.md").read_text("utf-8")

    @staticmethod
    def option_help(help_text: str, flag: str) -> str:
        """把某个选项那一段从 format_help() 里切出来。

        两个坑：选项名在开头的 usage 行里也会出现一次，直接 index() 会切到用法块去；
        而选项说明又会按终端宽度折行，所以也不能只取一行。这里锚定"行首两空格 + 选项名"
        找到选项列表里的那一处，再切到下一个同样形状的行首为止。
        """
        anchor = f"\n  {flag}"
        i = help_text.index(anchor)
        rest = help_text[i + 1 :]
        j = rest.find("\n  --", 1)
        return rest[: j if j > 0 else len(rest)]

    def _native_subs_help(self) -> str:
        return self.option_help(self.help, "--native-subs")

    def test_the_option_help_no_longer_carries_the_withdrawn_default(self):
        self.assertNotIn(self._squash(self.STALE), self._squash(self._native_subs_help()))

    def test_the_withdrawn_default_is_gone_from_every_user_facing_text(self):
        for name, text in (("--help", self.help), ("docstring", self.doc), ("README", self.readme)):
            with self.subTest(where=name):
                self.assertNotIn(self._squash(self.STALE), self._squash(text))

    def test_the_option_help_names_both_platforms(self):
        """光删掉错话不够——必须说清楚这个开关对哪个平台有意义。"""
        seg = self._native_subs_help()
        self.assertIn("YouTube", seg)
        self.assertIn("X", seg)

    def test_the_option_help_says_it_is_a_no_op_on_x(self):
        self.assertIn(self._squash("对 X 不改变任何行为"), self._squash(self._native_subs_help()))

    def test_the_help_and_the_docstring_agree_on_the_x_default(self):
        self.assertIn(self._squash("X 默认本就优先使用平台字幕"), self._squash(self.help))
        self.assertIn(self._squash("默认优先用平台自带字幕"), self._squash(self.doc))

    def test_the_help_and_the_docstring_agree_on_the_youtube_default(self):
        self.assertIn(self._squash("YouTube 默认关闭，走本地 Whisper 转写"), self._squash(self.help))
        self.assertIn(self._squash("默认本地 Whisper 转写"), self._squash(self.doc))

    def test_the_readme_describes_the_rules_as_per_platform(self):
        self.assertIn("按平台分开", self.readme)


class TestR3F07UnknownDurationHasNoEscapeHatch(unittest.TestCase):
    """R3-F07：硬上限只能比"已知的 duration"，所以未知时长不能有放行口。

    第 3 轮审查指出：`if not known: ... return` 让 --allow-long-video 直接跳过后面
    整段，6 小时硬上限在这条路径上根本执行不到，下载和转写两层又都没有独立于
    metadata 的字节数或墙钟限制——放行等于无上限。既然给不出可执行的兜底，
    就不给这个放行口。

    这里直接测闸门函数本身，不走整条处理链，把"开关无效"钉在最小单元上。
    """

    UNKNOWN = (None, "3600", float("nan"), float("inf"), 0, -1, True)

    def test_every_unknown_shape_is_refused_regardless_of_the_flag(self):
        for dur in self.UNKNOWN:
            for allow in (False, True):
                with self.subTest(duration=dur, allow_long=allow):
                    info = {} if dur is None else {"duration": dur}
                    with self.assertRaises(xsub.XsubError):
                        xsub.guard_youtube_admission(info, "dQw4w9WgXcQ", allow)

    def test_a_known_long_duration_still_has_a_working_escape_hatch(self):
        """负对照：真正该放行的那一档没有被这次收紧误伤。"""
        info = {"duration": xsub.MAX_DURATION_SEC + 1}
        with self.assertRaises(xsub.XsubError):
            xsub.guard_youtube_admission(info, "dQw4w9WgXcQ", False)
        self.assertIsNone(xsub.guard_youtube_admission(info, "dQw4w9WgXcQ", True))

    def test_the_flag_help_does_not_promise_a_cap_it_cannot_enforce(self):
        """--help 不能再声称"6 小时硬上限仍生效"——对未知时长那是空头承诺。"""
        seg = TestR3F02TheUserFacingContractCannotDrift.option_help(
            xsub.build_parser().format_help(), "--allow-long-video"
        )
        self.assertIn("时长未知也不放行", re.sub(r"\s+", "", seg))


if __name__ == "__main__":
    unittest.main(verbosity=2)
