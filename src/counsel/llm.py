"""역할별 LLM 호출 계층.

config.yaml 의 llm.roles.<role>.provider 로 백엔드를 고르면 같은 인터페이스로 호출된다.
- openrouter : HTTP API (환경변수 OPENROUTER_API_KEY)
- claude_cli : Claude 구독 (`claude -p`)
- codex_cli  : ChatGPT 구독 (`codex exec`)
CLI 백엔드는 도구를 끄고 빈 임시 폴더에서 실행해 코딩 에이전트로 동작하지 않게 한다.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import httpx

from .config import load_config, path

Message = dict  # {"role": "user" | "assistant", "content": str}

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class LLMResult:
    text: str
    role: str
    provider: str
    model: str
    latency_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None  # 구독 CLI는 실제 청구가 아닌 환산 비용


class LLMError(RuntimeError):
    pass


class UsageLimitError(LLMError):
    """구독 사용량 한도나 429. 기다리면 풀리는 오류."""


_LIMIT_RE = re.compile(r"usage limit|session limit|weekly limit|hit your .{0,30}limit|rate.?limit|limit reached|"
                       r"too many requests|quota|\b429\b", re.I)
_log_lock = threading.Lock()


def _render_prompt(messages: list[Message]) -> str:
    """CLI 백엔드는 프롬프트 하나만 받으므로 대화 이력을 텍스트로 펼친다."""
    if len(messages) == 1 and messages[0]["role"] == "user":
        return messages[0]["content"]
    parts = ["<conversation>"]
    parts += [f"[{m['role']}]\n{m['content']}\n" for m in messages]
    parts.append("</conversation>\n위 대화에 이어질 [assistant]의 다음 응답만 출력하세요.")
    return "\n".join(parts)


def _raise_for_text(msg: str) -> None:
    if _LIMIT_RE.search(msg):
        raise UsageLimitError(msg[:500])
    raise LLMError(msg[:500])


def _child_env() -> dict:
    # 상위 Claude Code 세션의 표식이 하위 CLI 동작을 바꾸지 않도록 제거
    return {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}


# --- backends: (cfg, system, messages, timeout) -> (text, in_tokens, out_tokens, cost) ---

def _call_openrouter(cfg, system, messages, timeout):
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise LLMError("OPENROUTER_API_KEY 환경변수가 없습니다")
    payload = {
        "model": cfg["model"],
        "messages": ([{"role": "system", "content": system}] if system else []) + messages,
    }
    for k in ("temperature", "max_tokens"):
        if k in cfg:
            payload[k] = cfg[k]
    payload.update(cfg.get("extra") or {})
    r = httpx.post(OPENROUTER_URL, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=timeout)
    if r.status_code == 429:
        raise UsageLimitError(r.text[:500])
    if r.status_code >= 400:
        raise LLMError(f"HTTP {r.status_code}: {r.text[:500]}")
    data = r.json()
    if "error" in data:
        _raise_for_text(json.dumps(data["error"], ensure_ascii=False))
    text = data["choices"][0]["message"].get("content") or ""
    u = data.get("usage") or {}
    return text, u.get("prompt_tokens"), u.get("completion_tokens"), u.get("cost")


def _call_claude_cli(cfg, system, messages, timeout):
    exe = cfg.get("cmd") or shutil.which("claude")
    if not exe:
        raise LLMError("claude CLI를 찾을 수 없습니다")
    with tempfile.TemporaryDirectory(prefix="counsel_claude_") as tmp:
        args = [exe, "-p", "--model", cfg["model"], "--output-format", "json",
                "--tools", "", "--no-session-persistence", "--strict-mcp-config",
                "--disable-slash-commands"]
        if system:
            sp = Path(tmp) / "system.txt"
            sp.write_text(system, encoding="utf-8")
            args += ["--system-prompt-file", str(sp)]
        proc = subprocess.run(args, input=_render_prompt(messages).encode("utf-8"), capture_output=True,
                              cwd=tmp, env=_child_env(), timeout=timeout)
    out = proc.stdout.decode("utf-8", "replace").strip()
    err = proc.stderr.decode("utf-8", "replace").strip()
    try:
        data = json.loads(out.splitlines()[-1]) if out else {}
    except json.JSONDecodeError:
        data = {}
    if proc.returncode != 0 or data.get("is_error") or not data:
        _raise_for_text(f"claude exit={proc.returncode} {data.get('result') or out or err}")
    u = data.get("usage") or {}
    tin = sum(u.get(k) or 0 for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
    return data.get("result") or "", tin, u.get("output_tokens"), data.get("total_cost_usd")


def _codex_cmd(cfg) -> list[str]:
    if cfg.get("cmd"):
        return list(cfg["cmd"])
    shim = shutil.which("codex")
    if not shim:
        raise LLMError("codex CLI를 찾을 수 없습니다")
    if shim.lower().endswith(".cmd"):
        # npm .CMD 래퍼는 인자 따옴표가 깨지기 쉬워 node 로 codex.js 를 직접 실행
        return [shutil.which("node"), str(Path(shim).parent / "node_modules/@openai/codex/bin/codex.js")]
    return [shim]


def _call_codex_cli(cfg, system, messages, timeout):
    prompt = _render_prompt(messages) + "\n\n(파일·명령·도구를 사용하지 말고 최종 답변 텍스트만 바로 출력하세요.)"
    with tempfile.TemporaryDirectory(prefix="counsel_codex_") as tmp:
        last = Path(tmp) / "last.txt"
        args = _codex_cmd(cfg) + [
            "exec", "-m", cfg["model"], "-c", f'model_reasoning_effort="{cfg.get("reasoning_effort", "medium")}"',
            "-s", "read-only", "--skip-git-repo-check", "--ephemeral", "--ignore-user-config", "--ignore-rules",
            "-C", tmp, "--json", "-o", str(last)]
        if system:
            # Codex 기본 지시문(코딩 에이전트)을 역할의 시스템 프롬프트로 교체
            sp = Path(tmp) / "system.md"
            sp.write_text(system, encoding="utf-8")
            args += ["-c", f'model_instructions_file="{sp.as_posix()}"']
        args.append("-")
        proc = subprocess.run(args, input=prompt.encode("utf-8"), capture_output=True,
                              cwd=tmp, env=_child_env(), timeout=timeout)
        text = last.read_text(encoding="utf-8").strip() if last.exists() else ""
    tin = tout = None
    errors = []
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = ev.get("type", "")
        if t == "turn.completed":
            u = ev.get("usage") or {}
            tin, tout = u.get("input_tokens"), u.get("output_tokens")
        elif t == "item.completed" and (ev.get("item") or {}).get("type") == "agent_message" and not text:
            text = ev["item"].get("text", "")
        elif t in ("error", "turn.failed"):
            errors.append(json.dumps(ev, ensure_ascii=False))
    if proc.returncode != 0 or errors or not text:
        _raise_for_text(f"codex exit={proc.returncode} {' '.join(errors)} {proc.stderr.decode('utf-8', 'replace')[-500:]}")
    return text, tin, tout, None


_BACKENDS = {"openrouter": _call_openrouter, "claude_cli": _call_claude_cli, "codex_cli": _call_codex_cli}


def _log(record: dict) -> None:
    log_path = path(load_config()["llm"].get("call_log", "logs/llm_calls.jsonl"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with _log_lock, open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def set_interactive(max_retries: int = 1, rate_limit_max_waits: int = 0, timeout_s: int = 180) -> None:
    """대화형 앱용: 사람이 기다리는 호출이라 재시도·한도 대기·타임아웃을 줄이고 오류를 바로 올린다.
    config 의 값을 이 프로세스에서만 덮어쓴다(시뮬레이션 루프는 영향 없음)."""
    conf = load_config()["llm"]
    conf.update(max_retries=max_retries, rate_limit_max_waits=rate_limit_max_waits, timeout_s=timeout_s)


def chat(role: str, messages: list[Message], system: str | None = None,
         max_retries: int | None = None, **overrides) -> LLMResult:
    """역할 설정대로 LLM을 호출한다. 일시 오류는 재시도, 사용량 한도는 대기 후 재시도."""
    conf = load_config()["llm"]
    max_retries = conf["max_retries"] if max_retries is None else max_retries
    cfg = {**conf["roles"][role], **overrides}
    backend = _BACKENDS[cfg["provider"]]
    retries = waits = 0
    while True:
        t0 = time.perf_counter()
        base = {"ts": datetime.now().isoformat(timespec="seconds"), "role": role,
                "provider": cfg["provider"], "model": cfg["model"]}
        try:
            text, tin, tout, cost = backend(cfg, system, messages, conf["timeout_s"])
            res = LLMResult(text.strip(), role, cfg["provider"], cfg["model"],
                            round(time.perf_counter() - t0, 2), tin, tout, cost)
            _log({**base, **{k: v for k, v in asdict(res).items() if k != "text"}, "ok": True})
            return res
        except UsageLimitError as e:
            _log({**base, "ok": False, "error": f"usage_limit: {e}"[:500]})
            waits += 1
            if waits > conf["rate_limit_max_waits"]:
                raise
            time.sleep(conf["rate_limit_wait_s"])
        except (LLMError, httpx.HTTPError, subprocess.TimeoutExpired) as e:
            _log({**base, "ok": False, "error": str(e)[:500]})
            retries += 1
            if retries > max_retries:
                raise LLMError(f"{role} 호출 실패 ({retries}회): {e}") from e
            time.sleep(min(60, 2 ** retries))
