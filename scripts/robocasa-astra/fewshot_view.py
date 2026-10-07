"""Write docs/astra-robodawn-fewshot.html: exactly how the few-shot demonstrations enter each Astra request.

For every task it renders the request of turn 1 in the order the model receives it (system prompt, then
the demonstration block with each image followed by its own text, the END line, the current views and
the turn text), built by the same functions the runner uses (``prompts.demo_parts``, ``system_prompt``,
``turn_text``). Images are embedded, so the file is self-contained. No model or simulator is used:

    PYTHONPATH=plugins/inspect-robots-robocasa-astra/src python3 scripts/robocasa-astra/fewshot_view.py
"""

import base64
import html
import json
from pathlib import Path

from robocasa_astra.astra_robodawn import prompts
from robocasa_astra.astra_robodawn.codex import MODEL, REASONING_EFFORT

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "astra-robodawn-fewshot.html"
TASKS = ("OpenCabinet", "PickPlaceSinkToCounter", "PrepareCoffee", "PanTransfer", "StirVegetables")
SAMPLE_STATE = {"fingertip_cm": [25.0, 0.0, 129.0], "approach": [0.0, 0.0, -1.0], "finger_axis": [0.0, 1.0, 0.0],
                "gripper_opening": 1.0, "gripper_command": "open", "surface_z_cm": 92.0, "steps_used": 0,
                "step_budget": 1800}
CAPTIONS = ["left overview camera (annotated grid)", "right overview camera (annotated grid)", "wrist camera"]

CSS = """
:root { --bg:#fbfaf7; --fg:#1d1d1b; --muted:#6b6a65; --line:#e3e0d8; --card:#ffffff; --sys:#eef3fb;
        --demo:#f6f1e6; --cur:#eaf5ee; --end:#fbe9e7; --code:#f3f1ec; --accent:#2f5d9e; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#17181a; --fg:#e8e6e1; --muted:#9a988f; --line:#33353a; --card:#1f2124; --sys:#1d2633;
  --demo:#2a261d; --cur:#1b2a21; --end:#33201d; --code:#26282c; --accent:#8fb3ea; } }
:root[data-theme="dark"] { --bg:#17181a; --fg:#e8e6e1; --muted:#9a988f; --line:#33353a; --card:#1f2124;
  --sys:#1d2633; --demo:#2a261d; --cur:#1b2a21; --end:#33201d; --code:#26282c; --accent:#8fb3ea; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 system-ui,-apple-system,"Noto Sans KR",sans-serif; }
main { max-width:980px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:24px; margin:0 0 4px; } h2 { font-size:19px; margin:36px 0 10px; } h3 { font-size:16px; margin:18px 0 8px; }
p, li { color:var(--fg); } .muted { color:var(--muted); font-size:13px; }
table { border-collapse:collapse; width:100%; font-size:14px; margin:8px 0; }
th, td { border:1px solid var(--line); padding:6px 8px; text-align:left; vertical-align:top; }
th { background:var(--code); }
pre { white-space:pre-wrap; word-break:break-word; background:var(--code); padding:10px 12px; border-radius:6px;
      font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; margin:0; }
.flow { display:grid; gap:6px; margin:10px 0; }
.box { border:1px solid var(--line); border-radius:8px; padding:8px 12px; }
.sys { background:var(--sys); } .demo { background:var(--demo); } .cur { background:var(--cur); } .end { background:var(--end); }
.part { display:grid; grid-template-columns:44px 1fr; gap:8px; align-items:start; margin:6px 0; }
.idx { font:12px ui-monospace,monospace; color:var(--muted); padding-top:4px; }
.img img { width:256px; max-width:100%; border-radius:6px; border:1px solid var(--line); display:block; }
.img .cap { font-size:12px; color:var(--muted); }
details { border:1px solid var(--line); border-radius:8px; background:var(--card); padding:6px 12px; margin:10px 0; }
summary { cursor:pointer; font-weight:600; }
nav a { color:var(--accent); margin-right:12px; } code { background:var(--code); padding:1px 4px; border-radius:4px; }
"""


def _img(path: str) -> str:
    data = base64.b64encode(Path(path).read_bytes()).decode()
    rel = Path(path).resolve().relative_to(REPO)
    return f'<div class="img"><img alt="{html.escape(str(rel))}" src="data:image/png;base64,{data}"><div class="cap">{html.escape(str(rel))}</div></div>'


def _parts_html(parts: list[dict], start: int, cls: str) -> str:
    rows = []
    for k, part in enumerate(parts, start=start):
        body = _img(part["path"]) if part["type"] == "image" else f"<pre>{html.escape(part['text'])}</pre>"
        rows.append(f'<div class="part {cls}"><div class="idx">#{k}<br>{"img" if part["type"] == "image" else "text"}</div>{body}</div>')
    return "\n".join(rows)


def _task_section(task: str) -> str:
    demos = prompts.demos_for(task, 1)
    block = prompts.demo_parts(demos)
    system = prompts.system_prompt(prompts.load_profile(), task, shown_demos=True)
    turn = prompts.turn_text(1, 45, demos[-1].data["instruction"] or task, SAMPLE_STATE, [], "MEMORY: (empty)",
                             CAPTIONS, task)
    n_img = sum(p["type"] == "image" for p in block)
    task_demo = demos[-1].data
    verified = task_demo.get("source", {}).get("verified", {})
    source = task_demo.get("source", {})
    current = [f"[현재 카메라 C{i + 1}: {c}]" for i, c in enumerate(CAPTIONS)]
    return f"""
<h2 id="{task}">{task}</h2>
<table>
<tr><th>시연 출처</th><td>전문가 에피소드 {source.get('episode')} (layout {source.get('layout_id')} / style {source.get('style_id')}), 평가 장면과 다른 주방</td></tr>
<tr><th>만든 방식</th><td>{html.escape(json.dumps(verified, ensure_ascii=False))}</td></tr>
<tr><th>시연 블록</th><td>입력 조각 {len(block)}개, 이미지 {n_img}장 (명령 입문 포함), 텍스트 {sum(len(p.get('text', '')) for p in block):,}자</td></tr>
<tr><th>한 요청 이미지</th><td>시연 {n_img}장 + 현재 {len(CAPTIONS)}장 = {n_img + len(CAPTIONS)}장</td></tr>
</table>
<details><summary>① 시스템 프롬프트 ({len(system):,}자, 매 요청 동일)</summary><pre>{html.escape(system)}</pre></details>
<details open><summary>② 시연 블록: 매 요청 맨 앞에 같은 순서로 들어감 (이미지 바로 뒤에 그 턴의 글)</summary>
{_parts_html(block[:-1], 1, "demo")}
{_parts_html(block[-1:], len(block), "end")}
</details>
<details><summary>③ 현재 턴 (매 턴 바뀌는 부분): 현재 이미지 {len(CAPTIONS)}장, 그다음 턴 텍스트 (1턴 예시)</summary>
<pre>{html.escape(chr(10).join(current))}</pre>
<div style="height:6px"></div>
<pre>{html.escape(turn)}</pre>
</details>
"""


def main() -> None:
    sections = "\n".join(_task_section(t) for t in TASKS)
    nav = " ".join(f'<a href="#{t}">{t}</a>' for t in TASKS)
    page = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Astra few-shot 입력 구조</title><style>{CSS}</style></head>
<body><main>
<h1>Astra few-shot 입력 구조 (RoboDawn 방식)</h1>
<p class="muted">scripts/robocasa-astra/fewshot_view.py가 실행기와 같은 함수(prompts.demo_parts, system_prompt, turn_text)로 생성.
모델 {MODEL}, reasoning effort {REASONING_EFFORT}, 호출 방식 codex app-server.</p>

<h2>한 요청의 구성</h2>
<p>매 턴은 서로 독립된 요청입니다. 모델은 이전 턴을 기억하지 못하므로, 아래 전체를 매 턴 다시 보냅니다.
①과 ②는 에피소드 내내 똑같아서 서버의 prompt cache가 적용되고(캐시 입력 단가 1/10), ③만 매 턴 바뀝니다.</p>
<div class="flow">
<div class="box sys"><b>① 시스템 프롬프트</b>: 역할, 로봇 설명, 명령 문법, 성공 조건, 응답 형식 (baseInstructions)</div>
<div class="box demo"><b>② 시연 블록</b>: 명령 입문(PRIMER) 8단계 → 과제 시연(DEMO) 각 턴. 이미지가 있는 턴은 [이미지 → 그 턴의 글(state, scene, plan, commands, net effect, FAILED)] 순서</div>
<div class="box end"><b>END 문장</b>: "시연 끝. 이제 당신의 에피소드, 위치는 다시 측정하라"</div>
<div class="box cur"><b>③ 현재 턴</b>: 현재 카메라 3장 → 턴 텍스트(과제, 성공 조건, 지난 명령 결과, 현재 상태, 메모리)</div>
</div>

<h2>이전 방식(P4~P9)과의 차이</h2>
<table>
<tr><th></th><th>이전 (codex exec)</th><th>지금 (codex app-server, RoboDawn과 같은 구조)</th></tr>
<tr><td>시연 글 위치</td><td>시스템 프롬프트 안</td><td>요청 입력 안, 각 이미지 바로 뒤</td></tr>
<tr><td>이미지 순서</td><td>모든 이미지(D1.., C1..)가 텍스트보다 먼저. 글에서 "[image D3]" 번호로만 연결</td><td>이미지와 그 턴의 글이 붙어 있음</td></tr>
<tr><td>과제 시연 이미지</td><td>최대 6장</td><td>핵심 턴(첫 턴, gripper/point 턴, 지정 턴) + 마지막, 최대 16장</td></tr>
<tr><td>효과(effect)</td><td>손끝 이동량(cm)</td><td>실행기로 만든 시연(OpenCabinet, PickPlace): 문 열림 %, 손잡이/물체까지 거리 등 물체 기준 net effect. 전문가 녹화 시연(나머지 3개)은 녹화 손끝 이동량</td></tr>
<tr><td>실패 명령</td><td>commands 안에 섞여 있음</td><td>commands에서 빼고 "(FAILED: 명령: 이유)"로 따로 표시</td></tr>
<tr><td>끝 표시</td><td>없음</td><td>END 문장</td></tr>
</table>
<p class="muted">RoboDawn과 남은 차이: RoboDawn은 시연을 별도 user 메시지에 넣고 assistant의 "Understood" 응답을 붙입니다.
app-server의 한 턴에는 user 입력 하나만 넣을 수 있어서, 시연 블록과 현재 턴을 한 입력 안에 END 문장으로 구분합니다.</p>

<nav>{nav}</nav>
{sections}
</main></body></html>"""
    OUT.write_text(page)
    print(f"wrote {OUT} ({len(page) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
