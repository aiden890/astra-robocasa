# VLA 실험 코드: depth, 선택형 성공 예시, 중단 복구

`feat/context_astra_robodawn`의 VLA 버전을 기준으로 수정했습니다. 모델이 반환하는 행동은 기존과 같은 RoboCasa365 순서의 12차원 벡터 16개입니다. 한 chunk는 20Hz에서 0.8초이며, native 성공 조건이나 스텝 한도에 도달하면 chunk 도중에도 끝납니다.

| 벡터 위치 (1부터) | 의미 |
| --- | --- |
| 1–3 | 베이스 전후·좌우·회전 |
| 4 | 몸통 높이 |
| 5 | 팔/베이스 제어 모드 |
| 6–8 | 손끝 이동 X·Y·Z |
| 9–11 | 손끝 회전 X·Y·Z |
| 12 | 그리퍼 닫기/열기 |

이 표는 모델 출력의 dataset 순서입니다. `action_format.py`가 실제 native controller 순서로 변환합니다.

## 입력 조건과 성공 예시

| 옵션 | 모델에 추가되는 입력 | 모델 출력 |
| --- | --- | --- |
| `--condition rgb` | 기존 VLA RGB·로봇 상태·피드백·텍스트 기억 | `scene`, `progress`, `memory`, `plan`, `actions` |
| `--condition color` | RGB와 정합된 grayscale depth 이미지·각 이미지의 미터 단위 스케일 | 같은 행동 출력 |
| `--condition pixel` | 현재 RGB의 관측 ID와 카메라 이름; 모델이 요청한 픽셀의 Z 답변 | 행동 또는 `queries` |
| `--condition hybrid` | RGB + 정렬된 depth 이미지 + 선택형 pixel Z 조회. 연구 조건 C | 행동 또는 `queries` |
| `--condition grid` | 같은 관측 ID; 16×16 셀의 중심 Z와 구역 통계 | 행동 또는 `queries` |
| `--shots 0` | 성공 데모 없음. 기본값 | 출력 형식 동일 |
| `--shots 1` | 다른 주방의 같은 태스크 성공 데모 1개 | 출력 형식 동일 |
| `--primer` | 행동을 설명하는 primer. 성공 데모와 독립 옵션 | 출력 형식 동일 |

Zero-shot에는 primer도 자동으로 들어가지 않습니다. 두 overview의 10cm grid·손끝 annotation과 작업면 높이 등 기존 VLA 정보는 모든 조건에 공통으로 유지합니다. 기존 VLA의 현재 관측 3장, 로봇 상태, 최근 12턴 피드백, 모델이 갱신하는 텍스트 기억은 유지합니다. 과거 영상 입력이나 새로운 컨텍스트 전략은 이번 변경에 추가하지 않았습니다.

`pixel`/`grid`/`hybrid`에서는 먼저 거리 질문만 반환할 수 있습니다. 물리 상태를 움직이지 않고 답변을 다음 모델 호출에 넣습니다. 한 관측에서 최대 4번 질문할 수 있고, 이전 질문과 답변을 모두 유지합니다. 모델 호출 재시도는 이 4라운드와 별개입니다. RGB 조건에는 depth 이미지·거리 답변·원시 depth가 들어가지 않습니다.

픽셀 좌표는 RGB 해상도 그대로, 좌상단 원점의 `(u,v)`입니다. 반환값은 카메라 광축 방향 Z이며 유클리드 거리가 아닙니다. 관측 ID, 카메라, 좌표 범위와 조건을 검사합니다. Overview는 384×384, wrist는 256×256이며 depth도 같은 크기와 상하 방향으로 렌더링합니다.

## 기능별 모듈

모든 VLA 모듈은 `src/robocasa_astra/astra_vla/`에 있습니다.

| 파일 | 담당 기능 |
| --- | --- |
| `run.py` | 실행·종료 흐름. 정상 native 평가만 시뮬레이터를 닫고, 중단 시 detach |
| `experiment.py` | CLI 옵션, 새 실행 설정, 소스·예시 해시, 기존 설정 채택 |
| `prompts.py` | VLA 지시문·태스크 성공 설명·선택형 성공 예시 구성 |
| `depth_input.py` | 조건별 depth 입력, query 스키마, RGB/depth 정합, 조회 결과 |
| `loop.py` | 관측 → 조회/행동 응답 → chunk → 다음 관측; 중간 진행 저장 |
| `memory.py` | chunk 피드백과 grasp 기억. 재개 시 durable trace에서 복원 |
| `caller.py` | 모델 요청 생성·기존 요청 채택·응답 재사용·토큰 합산 |
| `call_runner.py` | 호스트와 분리된 모델 요청 프로세스·무제한 대기·실패 receipt |
| `sim_client.py` | private socket 접속·살아 있는 worker 채택·종료된 worker 복구 |
| `sim_server.py` | native 시뮬레이터·depth 관측·거리 질문·idempotent chunk 처리 |
| `chunk_runner.py` | 벡터를 native 행동으로 바꾸고 성공/스텝 한도를 매 스텝 검사 |
| `action_format.py` | 기존 12차원 벡터 순서와 16행 검증. 의미와 순서 유지 |
| `recovery.py` | 행동 intent/ack 체크포인트·남은 chunk 재개·재생 검증 |
| `persistence.py` | fsync 원자 저장·프로세스의 정확한 PID/argv·잠금·상태 digest |

공통 `depth.py`, `depth_query.py`, `checkpoint.py`는 기존 실험 코드에서 가져왔습니다. 공유 `astra_robodawn/appserver.py`에는 timeout과 effort를 주입하는 옵션 및 프로세스 기록 hook만 추가했습니다. 다른 skill 실행의 기본 300초 제한과 low effort는 유지합니다.

## 응답 대기와 중단 복구

- VLA 모델 응답 timeout과 전체 wall timeout은 없습니다. 실제 프로세스가 살아 있으면 기다립니다.
- Native 통신도 고정 600초 뒤에 종료하지 않습니다. 살아 있는 정확한 worker를 계속 기다립니다.
- 모델의 실제 capacity/통신 실패는 같은 관측에서 15초 후 재시도합니다. 느리다는 이유만으로 재호출하지 않습니다.
- 실행당 실제 모델 attempt 예산은 3,000입니다. query·재시도도 이 예산에 포함합니다.
- 호스트 중단 후에도 model runner와 native worker는 살아 있습니다. `--resume`은 기존 PID/argv와 요청을 채택합니다.
- Worker가 종료되면 같은 frozen scene에서 승인된 행동을 재생합니다. 매 스텝의 physics·이미지/depth 관측·native 성공 플래그·controller 목표 digest를 검사합니다. 불일치 시 추가 행동을 막습니다.
- 행동을 실행하기 전에 intent를, 실행 후에는 ack와 chunk cursor를 fsync 원자 저장합니다. 승인되지 않은 pending 행동은 검증된 재생 후 한 번만 적용합니다.
- 완료 응답과 chunk 결과를 저장하므로, 응답을 받기 전에 호스트가 끊겨도 모델 호출이나 행동을 중복 실행하지 않습니다.
- 복구는 누적 행동 재생이므로 진행한 스텝 수에 비례해 시간이 걸립니다. 직접 상수시간 상태 로드는 아닙니다.
- 정상 `task_success=false`는 유효한 평가입니다. 실행 오류는 `task_success=null`로 남고 재개 가능한 원본을 보존합니다.

이 모듈은 단일 episode의 실행·재개를 담당합니다. 기존 배치 감독/모니터링을 자동으로 다시 켜거나 병렬도를 변경하지 않습니다. `--resume`은 같은 소스·설정의 실행만 허용합니다.

## 결과 파일 구조

```text
<run>/
  config.json                   # 조건, 예시, 모델, effort, native horizon, source manifest
  system_prompt.md
  response_schema.json
  demo_block.json / demo_block.txt
  episode_start.json
  turns/                        # 실제 RGB/depth 모델 입력
  trace.jsonl                   # 모델 출력, query/답변, 결과, 토큰, 지연
  memory.json
  summary.json                  # native 결과와 전체 사용량
  calls/turnNNN-qR/
    request.json                # 관측에 묶인 입력과 immutable identity
    input.json / prompt.txt
    events.attemptN.jsonl        # 원본 app-server events
    command.attemptN.json
    stderr.attemptN.log
    receipts.jsonl              # 실제 시도별 시작/완료/시간/usage
    inflight.json
    response.json / result.json / call.json
    lease.json / appserver-lease.json
  policy/
    progress.json               # 현재 관측, query round, 받은 응답, 미실행 chunk
    trace-records/              # 원자 저장한 재개용 턴 기록
    attempts.jsonl              # 실행 전체 모델 attempt 예산
  worker/
    checkpoint.json            # 승인 행동, pending intent, chunk cursor, 결과
    lease.json
  replay/                       # 기존 native replay 결과
  video.mp4                     # 현재 worker 영상
  segments/                     # worker 복구 전의 원본 영상/replay 보존
```

입력/캐시 입력/출력/reasoning 토큰을 시도별로 기록합니다. 보고되지 않은 usage는 `null`로 보존하며, `unknown_usage_attempts`로 집계합니다. 전체 합계는 실제 receipt를 한 번씩 읽습니다. 재사용한 응답을 새 호출로 세지 않습니다. 알 수 없는 usage가 있으면 비용 합계는 하한임을 표시합니다. 모델의 공개 reasoning 요약과 JSON 설명 필드를 기록합니다.

`receipt.seconds`는 개별 시도의 시간입니다. `call.json.seconds`는 실패 재시도 대기를 포함한 요청 전체 시간이며, `attempt_seconds`는 실제 시도 시간의 합입니다.

## 실행 예

Native 실행은 Spark2 환경에서 진행하고, 영상/입력 프레임이 저장되는 `--output`은 Lab 저장소를 사용해야 합니다. 아래 경로는 배포 환경에서 지정하는 변수입니다. 이번 변경 작업에서는 새로운 유료 평가를 시작하지 않았습니다.

```bash
python -m robocasa_astra.astra_vla.run \
  --task PrepareCoffee --scene 0 --scene-root "$SCENE_ROOT" \
  --condition pixel --shots 0 --effort medium \
  --output "$LAB_RUN_DIR" --allow-astra

# 성공 예시를 사용하는 실행에는 --shots 1을 추가합니다.
# 행동 primer도 필요할 때에만 --primer를 추가합니다.

python -m robocasa_astra.astra_vla.run \
  --resume "$LAB_RUN_DIR" --allow-astra
```

VLA 원본 기본 effort는 `low`입니다. 기존 medium 실험과 비교할 때는 새 실행에 `--effort medium`을 명시합니다. 재개할 때는 저장된 설정을 사용합니다. Frozen scene 준비·Spark2 접속·Lab 저장 경로 연결은 실행 전에 구성해야 합니다.

## 검증

로컬 테스트는 depth 조건 분리, query 좌표·관측 ID, 4개 query round, 무제한 대기, 부분 chunk 재개, 응답 재사용, 원본 retry 토큰/시간, attempt 예산, PID 재사용 방지를 확인합니다. 실제 모델 대신 fake app-server를 사용하여 호스트 종료 후 detached runner의 응답을 재사용하고 모델 요청이 1회뿐인 것도 확인했습니다.

Spark2의 실제 PandaOmron에서는 5개 태스크 각각 4스텝 ack 후 환경 재생, 5번째 pending 행동, 8스텝까지 재개, 같은 chunk 결과 재요청 시 추가 행동 0회를 검증했습니다. 이 검증은 모델 호출 0회·영상 생성 0회이며 평가 성공률에 포함하지 않습니다. [원본 검증 요약](../reports/vla-depth-recovery/native-proof.json)을 보존했습니다.

PrepareCoffee에서는 행동이 physics에 적용됐지만 ack 전에 예외가 난 경우도 별도로 검증했습니다. [Native 오류 복구 증거](../reports/vla-depth-recovery/native-fault-proof.json)에서 승인된 4스텝부터 재생해 pending 5번째 행동을 한 번 적용하고 8스텝까지 이어가는 것을 확인했습니다.

최종 로컬 검사: Astra 플러그인 94개 테스트 통과·1개 선택 의존성 검사 건너뜀, 코어 2,089개 통과·6개 건너뜀 및 coverage 100%, strict mypy 89개 파일 통과입니다. 변경한 Python 파일 19개는 Ruff 검사와 포맷 검사를 통과했습니다. 전체 저장소의 Ruff/포맷 검사에는 기준 브랜치부터 존재하던 다른 파일의 오류가 남아 있으므로 전체 저장소가 lint clean이라고 주장하지 않습니다.

## 작업 및 배포 위치

이 브랜치의 이후 구현·정리·검사·커밋·push는 모두 Lab-desktop에서 수행합니다. 작업 폴더는 `/home/aiden/Desktop/lab/robot/astra-vla-cartesian-rgb-depth`입니다. 기존 `/home/aiden/Desktop/lab/robot/astra-robocasa`의 미커밋 시각화·평가 변경을 보존하기 위해 별도 worktree를 사용합니다. 기존 실행 폴더에는 VLA 모듈과 필요한 공통 파일만 백업 후 반영합니다. 브랜치 전체를 전환하거나 실험을 자동으로 시작하지 않습니다.

## 세 태스크 비교 계획

A 기본(규리)은 RGB·현재 로봇 상태·최근 12턴 텍스트 기억을 사용하고 성공 예시는 제외합니다. B few-shot(규리)은 A에 평가 씬과 분리된 성공 예시를 추가합니다. C depth(민경호)는 A에 정렬된 depth 이미지와 pixel Z 조회를 함께 제공하는 `--condition hybrid --shots 0`입니다. C에는 별도 과거 영상이나 성공 예시를 추가하지 않습니다. 모델은 depth 이미지를 받은 상태에서 필요할 때 같은 관측의 픽셀을 질문하고 답변을 받은 뒤 행동합니다. 실제 요청/답변은 토큰·시간과 함께 기록됩니다.
