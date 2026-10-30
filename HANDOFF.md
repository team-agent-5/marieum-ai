# ML 서버 연동 안내 (백엔드용)

- 기준: `design/ml-server-protocol.md` v1.0 + 아래 추가·변경 사항. 근거와 실측은 [CALIBRATION.md](CALIBRATION.md) §0
- 작성: 2026-10-31

## 1. 접속 정보

| 항목 | 값 |
|---|---|
| WebSocket | `ws://10.1.2.11:8001/asr-stream` (v1.0 문서의 `localhost:8001` 대신) |
| 준비 여부 | `GET http://10.1.2.11:8001/healthz` → 200이면 준비 완료, 503이면 로딩 중 |
| 프로세스 생존 | `GET http://10.1.2.11:8001/livez` → 항상 200 (추가) |
| 접근 허용 | **백엔드 서버 `10.1.1.11`만.** 다른 곳에서는 연결이 안 된다 (GPU 서버 방화벽) |
| 인증 | 없음 |
| Mock | `python scripts/mock_asr_server.py` → `ws://<실행한 곳>:9001/asr-stream`. GPU 없이 같은 메시지를 보낸다 |

- 백엔드를 Docker 컨테이너에서 돌려도 기본 브리지 네트워크면 나가는 주소가 `10.1.1.11`이라 그대로 된다. 백엔드 서버 IP가 바뀌면 ML 쪽에 알려 달라 (방화벽 수정)
- ML 서버는 systemd로 돈다. 죽으면 자동 재시작하고, 재시작 후 20초 안에 `/healthz`가 200이 된다 (재시작 후 18초에 확인)

## 2. 꼭 지킬 것

1. **세션마다 WebSocket 연결을 따로 연다.** v1.0 §5.1.2는 오디오 프레임을 "가장 최근 `start`의 세션"에 넣는다. 연결 하나에서 두 환자 세션이 겹치면 앞 환자의 오디오가 뒤 환자 세션으로 들어간다. 연결 하나를 계속 쓰려면 세션이 절대 겹치지 않게 해야 한다
2. **오디오는 PCM16, 16kHz, 모노, little-endian.** 100ms(3,200샘플 = 6,400바이트)씩 바이너리 프레임으로 보낸다. 컨테이너(WAV 헤더, WebM, Ogg)는 백엔드가 벗긴다. **Opus는 받지 않는다** (`INVALID_FORMAT`). 샘플레이트는 8–48kHz를 받지만 16kHz를 권장한다
3. **분기는 `type`과 `code`로 한다.** 오디오는 실제 속도로 흘려 보내면 된다. 한꺼번에 몰아 보내도 동작은 하지만 interim이 덜 나온다
4. `trace_id`는 영문·숫자·`-`·`_` 1–64자 (UUIDv4 그대로 OK)

## 3. 보내는 메시지

```json
{"type": "start", "trace_id": "uuid", "patient_id": "305-KIM-7X3K",
 "sample_rate": 16000, "encoding": "pcm16", "language": "ko", "mode": "utterance"}
```
- `mode` (추가, 선택): `"utterance"`(기본) 또는 `"conversation"`. 아래 §5
- `language`는 `"ko"`만 받는다

이어서 오디오 바이너리 프레임을 보내고, 녹음이 끝나면:
```json
{"type": "end", "trace_id": "uuid"}
```
keepalive: `{"type": "ping", "trace_id": "keepalive-..."}` → `{"type": "pong", ...}`

## 4. 받는 메시지

### `transcript_interim` (말하는 도중)
```json
{"type": "transcript_interim", "trace_id": "uuid", "text": "미지근한 물",
 "confidence": 0.62, "is_final": false,
 "segment_id": 0, "stable_text": "미지근한", "unstable_text": "물", "audio_end_ms": 1312}
```
| 필드 | 설명 |
|---|---|
| `text`, `confidence`, `is_final` | v1.0 그대로. 뒤 interim이 앞 interim을 **교체** |
| `segment_id` (추가) | 구간 번호. utterance 모드는 항상 0 |
| `stable_text` (추가) | 확정된 앞부분. 이후 interim에서 바뀌지 않는다 → 앱에서 진하게 |
| `unstable_text` (추가) | 아직 바뀔 수 있는 뒷부분 → 앱에서 흐리게 |
| `audio_end_ms` (추가) | 세션 시작 기준, 이 전사가 반영한 음성의 끝 위치 (지연 측정용) |

- 말소리가 0.3초 이상 쌓인 뒤부터 약 0.5초마다 온다. 간격은 최소 300ms. GPU가 바쁘면 간격이 늘어난다
- 새 말소리가 없으면(쉬는 중) 오지 않는다

### `transcript_final`
```json
{"type": "transcript_final", "trace_id": "uuid", "text": "미지근한 물 주세요",
 "confidence": 0.84, "duration_ms": 1420,
 "model_version": "yoona-J/ASR_Whisper_Disease_General", "no_speech_prob": 0.0,
 "segment_id": 0, "is_last": true}
```
| 필드 | 설명 |
|---|---|
| `text` | 빈 문자열 가능 (무음) |
| `confidence` | 텍스트 토큰 확률 평균. 빈 전사면 0.0 |
| `duration_ms` | 그 구간의 음성 길이 |
| `model_version` | v1.0 문서의 `faster-whisper-large-v3-turbo`가 아니라 위 값이다 |
| `no_speech_prob` | 무음 추정. VAD가 말소리를 못 찾아 전사를 건너뛰면 1.0 |
| `segment_id` (추가) | 같은 `segment_id`의 interim을 이 `text`로 교체 |
| `is_last` (추가) | 세션의 마지막 final. 이걸 받으면 세션 끝 |

- `end` 후 마지막 final은 보통 0.1–0.4초 안에 온다 (§7)
- final은 구간 전체를 다시 전사한 결과라 드물게 앞의 `stable_text`와 다를 수 있다. 항상 final로 바꾼다

### `error`
```json
{"type": "error", "trace_id": "uuid", "code": "AUDIO_TOO_LONG", "message": "녹음은 10초를 초과할 수 없습니다."}
```

## 5. 두 가지 모드

| | `utterance` (기본, v4 그대로) | `conversation` (실시간 대화) |
|---|---|---|
| 용도 | 버튼 누르고 한 문장 → 후보 추천 | 말하는 동안 단어가 계속 화면에 나오는 대화 |
| final | `end` 후 1회 (`segment_id: 0`, `is_last: true`) | 0.8초 쉴 때마다 구간 final (`is_last: false`), `end` 후 마지막 final (`is_last: true`, 남은 말이 없으면 `text: ""`) |
| 길이 상한 | 10초. 넘으면 `AUDIO_TOO_LONG` + 자동 final | 구간 최대 6초 (쉼 없이 말하면 6초에서 끊어 다음 구간으로), 세션 최대 600초 |
| interim | 세션 전체를 다시 전사 | 현재 구간만 다시 전사 |

- 한 세션 안에서 final은 `segment_id` 순서대로 온다. 구간 final 뒤에는 그 구간의 interim이 오지 않는다
- 앱 화면 (conversation): 지난 구간의 final들 + 현재 구간의 `stable_text`(진하게) + `unstable_text`(흐리게)

## 6. 에러와 백엔드 조치

| `code` | 언제 | 연결 | 백엔드 조치 |
|---|---|---|---|
| `MODEL_UNAVAILABLE` | 모델 로딩 중에 연결 | error 후 `close(1013)` | 백오프 재연결, `/healthz` 확인 |
| `INVALID_FORMAT` | encoding·sample_rate·mode·language·trace_id 형식 오류, 같은 trace_id 중복 start, JSON 아님 | **유지** (v1.0은 `close(4001)`) | 세션 폐기, 환자에게 다시 녹음 안내 |
| `AUDIO_TOO_LONG` | utterance 10초 / conversation 600초 초과 | 유지. error 다음에 `is_last: true` final이 온다 | 토스트. 그 뒤 보내는 오디오와 `end`는 조용히 버린다 |
| `NO_SESSION` | `start` 없이 오디오나 `end` | 유지. 오디오는 `start` 사이에 한 번만 알림 | 로그, 세션 재발급 |
| `INTERNAL` | 전사 중 서버 오류 | 유지 | 로그, 재시도 |
| `TIMEOUT` | ML은 보내지 않는다. `end` 후 2초 안에 final이 없으면 백엔드가 판단 | — | fallback |

- 연결이 끊기면 그 연결의 진행 중 세션은 모두 버려진다 (final 안 옴)

## 7. 성능 (2026-10-31, GPU 서버 안에서 측정)
지연 = 그 음성의 마지막 100ms 프레임을 보낸 시각 → 메시지 수신. 서버 간 네트워크와 앱 녹음은 빠져 있다.

| 시험 | interim p50 / p95 | 마지막 final (end 후) p95 |
|---|---|---|
| utterance, 동시 4세션 | 123–157 / 283–380ms | 210–269ms |
| conversation, 실제 발화, 동시 4세션 | 117 / 132ms | — |
| conversation, 쉼 없이 말하는 4세션 (최악) | 312–335 / 543–606ms | 380–407ms |

목표(v1.0 SLO): interim p50 ≤ 300, p95 ≤ 500, final p95 ≤ 800ms. 동시 4세션까지 시험했다.

## 8. 주의: 이 모델의 특성

- **무음·잡음에서 없는 말을 지어낸다.** 무음 1초 → "아침 식사 시간은 몇 시까지예요"(confidence 0.73), 잡음 → "땅콩", "떡". 무음은 서버에서 막는다 (VAD + `no_speech_prob`). 잡음이나 아주 짧은 소리는 `unstable_text`에 잠깐 보일 수 있다
- **confidence가 대체로 높다.** 정상 발화 중앙값 0.9996, 지어낸 말도 0.4–0.7. v1.0의 기준(interim < 0.3이면 semantic_similarity 끄기, final < 0.4면 재질의)으로는 거의 걸리지 않는다. 기준값은 실제 환자 음성 분포를 보고 다시 정해야 한다
- **interim의 앞부분이 final의 접두어라는 보장은 없다.** `stable_text`는 거의 유지되지만 final이 이긴다 (v1.0 §7.3과 같음)

## 9. 직접 시험해 보기

```bash
# 백엔드 서버에서 (pip install websockets numpy, 16kHz WAV 하나)
python scripts/stream_client.py --url ws://10.1.2.11:8001/asr-stream --wav a.wav --show
python scripts/stream_client.py --url ws://10.1.2.11:8001/asr-stream --wav a.wav --mode conversation --show
```

## 10. 함께 정할 것

1. 모델: 구음장애 특화 whisper-small 유지 (현재) vs v1.0 문서의 large-v3-turbo
2. 세션마다 연결 (권장) vs 연결 하나 공유
3. 앱이 `conversation` 모드를 쓸지, 쓴다면 화면 표시 방식
4. confidence 기준값 (0.3 / 0.4) 재설정
