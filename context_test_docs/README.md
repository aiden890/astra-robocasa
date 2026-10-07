# Astra RoboCasa 실험 컨텍스트 (amp2)

작성일: 2026-10-06. 이 폴더는 amp2 서버에서 `gpt-6-astra`로 RoboCasa 과제를 제어하는 실험의 고정 조건과 진행 상황을 기록합니다.

- 개선 방향(RoboDawn 논문의 아이디어 적용): [robodawn_ideas.md](robodawn_ideas.md)
- 구현 계획(파일 구성, 명령 문법, 과제 설계, 단계): [implementation_plan.md](implementation_plan.md)
- 진행 기록(단계별 검증): [progress.md](progress.md)
- **실험 결과와 리뷰(Astra 6번): [results.md](results.md)**

## 1. 실험 조건

| 항목 | 값 |
|---|---|
| 서버 | amp2 (RTX 3090 1장, 다른 학습과 함께 사용) |
| 동시 실행 | 1개 |
| 로봇 | PandaOmron |
| 모델 | `gpt-6-astra`, reasoning effort `low` |
| 인증 | ChatGPT 구독 로그인 (`.runtime/auth`, API 키 없음) |
| Codex CLI | 0.159.2 단독 실행 파일 (`.runtime/bin/codex`) |
| 시뮬레이터 | conda `vjrc` 환경 (RoboCasa 1.0.1, robosuite 1.5.2, MuJoCo 3.3.1), `--local --native-scene` |
| 시뮬레이터 우선순위 | `nice -n 10` (학습보다 낮게) |
| 성공 판정 | 실행 종료 시점의 `success_at_end` (RoboCasa 내장 `_check_success`) |

## 2. 고정 과제 5개

스텝 예산은 RoboCasa 공식 `dataset_registry.py`의 horizon 값입니다.

| 과제 (RoboCasa 이름) | 종류 | 스텝 예산 |
|---|---|---|
| `PrepareCoffee` | composite | 1800 |
| `PanTransfer` | composite | 1800 |
| `OpenCabinet` | atomic | 1050 |
| `PickPlaceSinkToCounter` | atomic | 900 |
| `StirVegetables` | composite | 2400 |

5개 과제를 한 번씩 다 돌리면 최대 7,950스텝입니다.

## 3. 실행할 때마다 남겨야 하는 것

### 3.1 토큰 사용량

모델 호출마다 `codex exec --json` 이벤트의 `turn.completed.usage`를 저장하고, 실행 단위로 합칩니다.

- `input_tokens`: 전체 입력 토큰. 캐시된 입력도 여기에 포함됩니다.
- `cached_input_tokens`: 캐시에서 읽은 입력.
- `cache_write_input_tokens`: 캐시에 새로 쓴 입력.
- `output_tokens`: 출력 토큰.
- `reasoning_output_tokens`: 추론 토큰. 출력 토큰에 포함되어 있다고 가정합니다. 이 가정은 확인이 필요합니다.

예시(`2+2` 테스트 호출): input 8,664 (cached 7,552), output 5.

### 3.2 비용 추정치

구독 계정이라 실제로 청구되는 금액은 없습니다. 같은 사용량을 API로 썼다면 얼마였을지를 환산한 값입니다.

| 구분 | 단가 (1M 토큰당, Standard) |
|---|---|
| 입력 (캐시 아님) | $10.00 |
| 캐시된 입력 | $1.00 |
| 캐시 쓰기 | $12.50 |
| 출력 | $50.00 |

출처: [OpenAI API Pricing](https://developers.openai.com/api/docs/pricing), [GPT-6 Astra 모델 페이지](https://developers.openai.com/api/docs/models/gpt-6-astra). 입력이 272K 토큰을 넘으면 장문맥 요금(입력 2배, 출력 1.5배)이 적용되지만, 이 실험의 호출(약 12K)은 해당되지 않습니다.

```
비용 = (input - cached - cache_write) × 10
     + cached × 1
     + cache_write × 12.5
     + output × 50          (단위: 달러 / 1,000,000 토큰)
```

### 3.3 영상

- Inspect Robots가 매 스텝 카메라 프레임을 `runs/<실행>/eval/frames/` 아래에 `.npy`로 저장합니다(`store_frames=True`).
- 실행이 끝나면 카메라 3대(`agentview_left`, `agentview_right`, `eye_in_hand`)를 가로로 붙여서 20fps MP4 하나로 인코딩합니다.
- 인코더는 vjrc 환경의 `imageio-ffmpeg`입니다.
- 기존 [publish.py](../scripts/robocasa-astra/astra_ops/media/publish.py)는 원래 실험의 status-page 구조에 묶여 있어서, 실행 1개 단위로 인코딩하는 스크립트를 따로 만들어야 합니다.

### 3.4 실행별 요약 파일 (예정)

실행 폴더마다 `summary.json`을 남깁니다. 들어갈 항목:

- 과제, seed, 성공 여부, 사용한 스텝 수, 모델 호출 수
- 호출당 평균 및 최대 지연시간
- 토큰 합계(3.1의 항목별), 비용 추정치(3.2 공식)
- 실행 오류 여부: 모델 오류나 시간 초과로 끝난 실행은 실패와 따로 셉니다.
- 영상 경로

## 4. 사전 추정치 (확인 전)

전자레인지 켜기(TurnOnMicrowave) 16스텝 테스트에서 측정한 값입니다.

- **호출:** 호출 1번에 약 12,000토큰, 10.5~12.9초가 걸렸습니다.
- **행동 반복:** 모델이 매번 같은 행동을 4스텝씩 반복했습니다(`repeat=4`).

5개 과제의 추정치는 아래 가정으로 계산했습니다.

- 호출 1번당 약 7.5K 토큰이 캐시되고, 약 4.5K 토큰은 캐시되지 않으며, 출력은 약 100토큰입니다. 그러면 호출 1번에 약 $0.057입니다.
- 실패하면 스텝 예산을 끝까지 다 쓰는 것으로 계산했습니다(최대치).

| 과제 | 예상 호출 수 | 예상 시간 | 예상 비용 |
|---|---|---|---|
| PrepareCoffee | 약 450 | 약 1.5시간 | 약 $26 |
| PanTransfer | 약 450 | 약 1.5시간 | 약 $26 |
| OpenCabinet | 약 263 | 약 0.9시간 | 약 $15 |
| PickPlaceSinkToCounter | 약 225 | 약 0.8시간 | 약 $13 |
| StirVegetables | 약 600 | 약 2시간 | 약 $34 |
| 합계 (seed 1개) | 약 2,000 | 약 6.5시간 | 약 $115 |

디스크는 스텝당 약 0.6MB(프레임 `.npy`와 호출 PNG 포함)가 쌓입니다. 5개 과제를 한 번씩 돌리면 약 5GB입니다. /data의 남은 공간은 약 96GB입니다.

## 5. 실행 방법

```bash
cd /data/astra-robocasa
source .runtime/amp2.env
bash scripts/robocasa-astra/run.sh --local --native-scene \
  --task PrepareCoffee --steps 1800 --seed 771001 \
  --output runs/amp2-PrepareCoffee-s771001
```

## 6. 진행 상황

- [x] Codex 0.159.2 설치, 구독 로그인, `gpt-6-astra` 호출 확인
- [x] `run.py --local` (Spark2 없이 실행), `--effort`, `--strict-config` 추가
- [x] TurnOnMicrowave probe (모델 없음) 및 16스텝 연결 테스트 통과. 시뮬레이터 GPU 메모리는 순간 약 1GB.
- [ ] 호출별 토큰 사용량 저장 (`--json` 이벤트)
- [ ] 실행별 `summary.json` (토큰, 비용 추정, 성공 여부)
- [ ] 실행별 20fps MP4 인코딩
- [ ] 5개 과제 probe (모델 없음)
- [ ] 5개 과제 본 실행

## 7. 주의할 점

- **입력 정보:** 모델은 이미지 외에 물체 위치 같은 시뮬레이터 상태값도 받습니다. 이미지만 보는 실험이 아닙니다.
- **장면 고정:** 주방 배치(layout)와 스타일(style)을 각각 첫 번째 것 하나로 고정합니다.
- **버전 차이:** 원래 실험(Spark2, RoboCasa 0.5.1)과 RoboCasa 버전이 다릅니다. 숫자를 직접 비교하면 안 됩니다.
- **부가 출력:** Codex가 bubblewrap 없음, code-mode host 없음 경고를 냅니다. 두 기능 모두 꺼진 상태로 동작하므로 결과에는 영향이 없습니다.
