"""One model decision via ``codex app-server`` (JSON-RPC over stdio) with the saved ChatGPT subscription login.

Unlike ``codex exec``, which always puts attached images before the prompt text, ``turn/start`` takes an
ordered list of text and image parts. That is what the RoboDawn demonstration format needs: every
demonstration image directly followed by its own text, then the current views, then the turn text.

One app-server process serves a whole episode; every call starts a fresh ephemeral thread (no memory
between turns: the episode memory is in the prompt), so each request is self-contained like before.

Every call writes, under its own folder, the same review files as ``CodexCaller``: ``input.json`` (the
ordered parts, images as file paths), ``prompt.txt`` (readable rendering), ``command.json`` (process
argv and the exact RPC parameters, images as paths), ``events.jsonl`` (every server message of the call,
image data URLs elided), ``stderr.log``, ``response.json``, and ``call.json`` with usage, reasoning
summaries, latency and every retry. Reasoning effort is fixed to ``low`` by the experiment protocol.
"""

from __future__ import annotations

import base64
import json
import os
import queue
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

from .cost import add_usage
from .codex import CAPACITY_MARKERS, MAX_ATTEMPTS, MODEL, REASONING_EFFORT, CallResult, ModelCallError, write_input

CALL_TIMEOUT_S = 300
RPC_TIMEOUT_S = 60
SECRET_ENV = ("OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN")
TOOL_ITEMS = ("commandExecution", "fileChange", "mcpToolCall", "dynamicToolCall", "webSearch", "imageGeneration")


def _elide(value):
    """Copy of a JSON value with image data URLs replaced by their size (keeps events.jsonl readable)."""
    if isinstance(value, str) and value.startswith("data:image"):
        return f"<image data url, {len(value)} chars>"
    if isinstance(value, dict):
        return {k: _elide(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_elide(v) for v in value]
    return value


def wire_input(parts: list[dict]) -> list[dict]:
    """Input parts in the app-server ``UserInput`` form; images are sent inline as PNG data URLs."""
    out = []
    for part in parts:
        if part["type"] == "text":
            out.append({"type": "text", "text": part["text"]})
        else:
            data = base64.b64encode(Path(part["path"]).read_bytes()).decode()
            out.append({"type": "image", "url": "data:image/png;base64," + data})
    return out


def usage_from(breakdown: dict) -> dict:
    """App-server ``TokenUsageBreakdown`` in the field names used by ``codex exec`` and ``cost.estimate``."""
    return {"input_tokens": breakdown.get("inputTokens", 0), "cached_input_tokens": breakdown.get("cachedInputTokens", 0),
            "output_tokens": breakdown.get("outputTokens", 0),
            "reasoning_output_tokens": breakdown.get("reasoningOutputTokens", 0)}


class AppServerCaller:
    """Calls ``codex app-server`` for one turn; see the module docstring for what is recorded."""

    def __init__(self, executable: str, codex_home: str, system_prompt_path: Path, schema_path: Path,
                 workdir: Path, model: str = MODEL):
        self.executable, self.codex_home = executable, codex_home
        self.system_prompt = Path(system_prompt_path).read_text()
        self.schema = json.loads(Path(schema_path).read_text())
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.proc = None
        self.serial = 0
        self.log = None  # events of the current call are appended here

    def argv(self) -> list[str]:
        return [self.executable, "app-server", "--strict-config", "-c", 'model_provider="openai"',
                "-c", 'forced_login_method="chatgpt"', "--listen", "stdio://"]

    # -- process and JSON-RPC -------------------------------------------------------------------
    def _start(self, stderr_path: Path) -> None:
        env = {k: v for k, v in os.environ.items() if k not in SECRET_ENV}
        env["CODEX_HOME"] = self.codex_home
        self.stderr = stderr_path.open("a")
        self.proc = subprocess.Popen(self.argv(), cwd=self.workdir, env=env, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.stderr, text=True, encoding="utf-8", bufsize=1)
        self.incoming: queue.Queue = queue.Queue()
        self.pending: deque = deque()  # notifications that arrived while waiting for an RPC result

        def reader(stream, sink):
            for line in stream:
                try:
                    sink.put(json.loads(line))
                except json.JSONDecodeError:
                    sink.put({"unparsed": line})
            sink.put(None)

        threading.Thread(target=reader, args=(self.proc.stdout, self.incoming), daemon=True).start()
        self.rpc("initialize", {"clientInfo": {"name": "astra_robodawn", "title": "astra_robodawn", "version": "1"}})
        self._send({"method": "initialized", "params": {}})
        account = self.rpc("account/read", {"refreshToken": False}).get("account") or {}
        if account.get("type") != "chatgpt":
            raise ModelCallError(f"expected the ChatGPT subscription login, got account type {account.get('type')!r}")

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.proc:
            self.stderr.close()
        self.proc = None

    def _send(self, message: dict) -> None:
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def _receive(self, deadline: float) -> dict:
        try:
            message = self.incoming.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty:
            raise TimeoutError("no answer from codex app-server before the deadline") from None
        if message is None:
            raise ConnectionError("codex app-server exited")
        if self.log:
            self.log.write(json.dumps(_elide(message)) + "\n")
        if "method" in message and "id" in message:  # server-initiated request (approval, tool input): deny
            self._send({"id": message["id"], "error": {"code": -32601, "message": "denied: the robot controller has no tools"}})
        return message

    def rpc(self, method: str, params: dict, timeout: float = RPC_TIMEOUT_S) -> dict:
        self.serial += 1
        request_id = self.serial
        self._send({"id": request_id, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while True:
            message = self._receive(deadline)
            if message.get("id") == request_id and "method" not in message:
                if "error" in message:
                    raise RuntimeError(f"{method} failed: {json.dumps(message['error'])}")
                return message["result"]
            self.pending.append(message)

    # -- one decision ---------------------------------------------------------------------------
    def _turn(self, parts: list[dict], folder: Path, suffix: str) -> dict:
        thread_params = {"model": self.model, "modelProvider": "openai", "cwd": str(self.workdir),
                         "approvalPolicy": "never", "sandbox": "read-only", "ephemeral": True,
                         "baseInstructions": self.system_prompt}
        turn_params = {"input": parts, "model": self.model, "effort": REASONING_EFFORT, "summary": "detailed",
                       "outputSchema": self.schema}
        (folder / f"command{suffix}.json").write_text(json.dumps(
            {"argv": self.argv(), "thread/start": {**thread_params, "baseInstructions": "<system_prompt.md>"},
             "turn/start": {**turn_params, "input": "<input.json; images sent as PNG data URLs>"}}, indent=1))
        if self.proc is None or self.proc.poll() is not None:
            self._start(folder / f"stderr{suffix}.log")
        self.pending.clear()
        thread = self.rpc("thread/start", thread_params)
        thread_id = thread["thread"]["id"]
        out = {"thread_id": thread_id, "resolved_model": thread.get("model"), "usage": {}, "reasoning": [],
               "messages": [], "errors": [], "warnings": [], "status": None}
        reply = self.rpc("turn/start", {**turn_params, "threadId": thread_id, "input": wire_input(parts)})
        turn_id = reply["turn"]["id"]
        deadline = time.monotonic() + CALL_TIMEOUT_S
        while out["status"] is None:
            message = self.pending.popleft() if self.pending else self._receive(deadline)
            method, params = message.get("method"), message.get("params") or {}
            if params.get("threadId") not in (None, thread_id):
                continue
            if method == "thread/tokenUsage/updated" and params.get("turnId") == turn_id:
                out["usage"] = usage_from(params["tokenUsage"]["last"])
            elif method == "item/completed":
                item = params.get("item") or {}
                if item.get("type") == "reasoning":
                    out["reasoning"] += item.get("summary") or []
                elif item.get("type") == "agentMessage":
                    out["messages"].append(item)
                elif item.get("type") in TOOL_ITEMS:
                    out["errors"].append(f"unexpected tool item {item['type']}")
            elif method == "error":
                # willRetry: a transient connection problem the server recovers from itself ("Reconnecting... 2/5")
                key = "warnings" if params.get("willRetry") else "errors"
                out[key].append(params.get("error", {}).get("message", ""))
            elif "id" in message and "method" in message:
                out["errors"].append(f"server request denied: {method}")
            elif method == "turn/completed" and params.get("turn", {}).get("id") == turn_id:
                turn = params["turn"]
                out["status"] = turn.get("status")
                if turn.get("error"):
                    out["errors"].append(turn["error"].get("message", json.dumps(turn["error"])))
        finals = [m for m in out["messages"] if m.get("phase") == "final_answer"] or \
                 [m for m in out["messages"] if m.get("phase") is None]
        out["text"] = finals[-1]["text"] if finals else None
        return out

    def call(self, parts: list[dict], folder: Path) -> CallResult:
        folder.mkdir(parents=True, exist_ok=True)
        write_input(folder, parts)
        attempts = []
        start = time.monotonic()
        for attempt in range(MAX_ATTEMPTS):
            suffix = "" if attempt == 0 else f".retry{attempt}"
            t0 = time.monotonic()
            record = {"attempt": attempt + 1}
            with (folder / f"events{suffix}.jsonl").open("w") as log:
                self.log = log
                try:
                    out = self._turn(parts, folder, suffix)
                except (TimeoutError, ConnectionError, RuntimeError) as exc:
                    out = {"status": "client_error", "errors": [f"{type(exc).__name__}: {exc}"], "usage": {}, "text": None}
                    self.close()  # a fresh process for the retry
                finally:
                    self.log = None
            record.update(seconds=round(time.monotonic() - t0, 2), status=out["status"], usage=out["usage"],
                          errors=out["errors"], warnings=out.get("warnings", []), thread_id=out.get("thread_id"),
                          resolved_model=out.get("resolved_model"))
            attempts.append(record)
            if out["status"] == "completed" and out["text"] and not out["errors"]:
                (folder / "response.json").write_text(out["text"])
                result = CallResult(out["text"], out["usage"], out["reasoning"], round(time.monotonic() - start, 2), attempts)
                self._write_record(folder, result)
                return result
            error_text = " ".join(out["errors"])
            if out["status"] == "client_error" or any(m in error_text for m in CAPACITY_MARKERS):
                wait = min(120, 15 * (attempt + 1))
                record["action"] = f"retry after {wait} s (simulator is paused, no action applied)"
                (folder / "attempts.json").write_text(json.dumps(attempts, indent=1))
                time.sleep(wait)
                continue
            break
        (folder / "attempts.json").write_text(json.dumps(attempts, indent=1))
        error = ModelCallError(f"no final answer after {len(attempts)} attempt(s); see {folder}")
        error.usage = {}  # discarded answers are still billed: the loop adds this to the episode total
        for record in attempts:
            add_usage(error.usage, record["usage"])
        raise error

    def _write_record(self, folder: Path, result: CallResult) -> None:
        (folder / "call.json").write_text(json.dumps({
            "caller": "codex app-server", "model": self.model, "reasoning_effort": REASONING_EFFORT,
            "reasoning_summary_requested": True, "seconds": result.seconds, "usage": result.usage,
            "reasoning": result.reasoning, "final_text": result.text, "attempts": result.attempts,
        }, indent=1))
