# 백엔드 중계 코드 작성용 프롬프트

백엔드 담당자가 AI 코딩 도구에 그대로 붙여 넣는 프롬프트다. `[ ]` 두 군데만 백엔드 상황에 맞게 채운다.

- 같이 넘길 파일: `scripts/mock_asr_server.py` (없으면 테스트 불가), [BACKEND_FAQ.md](BACKEND_FAQ.md), [HANDOFF.md](HANDOFF.md)
- 작성: 2026-10-31

---

````text
# 작업: 앱 ↔ ML 서버 음성 인식 중계(WebSocket) 구현

## 배경
우리 서비스는 구음장애 환자의 음성을 텍스트로 바꿔 주는 앱이다.
앱이 백엔드에 WebSocket으로 오디오를 보내면, 백엔드는 그 오디오를 ML 서버(음성 인식)로 전달하고,
ML 서버가 보낸 전사 결과를 받아 앱에 보내 준다. 백엔드는 이 **중계 역할**을 한다.

  앱 ──WS /utterance/stream──▶ 백엔드 ──WS /asr-stream──▶ ML 서버
     ◀── interim / final / 추천 ──       ◀── interim / final ──

- 백엔드 스택: [예: Python 3.11 + FastAPI]  ← 우리 프로젝트에 맞게
- 넣을 위치: [예: app/routers/utterance.py]  ← 우리 프로젝트에 맞게
- 기존 코드 구조, 인증(JWT) 방식, 설정 관리 방식을 먼저 읽고 그 스타일을 따라라.

## ML 서버 정보
- 주소: ws://10.1.2.11:8001/asr-stream  (환경변수 ML_ASR_URL로 뺄 것)
- 백엔드 서버(10.1.1.11)에서만 접속 가능. 인증 없음
- 상태 확인: GET http://10.1.2.11:8001/healthz → 200 준비됨, 503 로딩 중
- 발화 1번 = ML 연결 1개. 한 연결에서 세션 두 개를 동시에 쓰면 안 된다
  (오디오 프레임에 세션 정보가 없어서 섞인다)

## ML 서버로 보내는 메시지
1) 연결 직후 시작 메시지 (텍스트 프레임)
   {"type":"start","trace_id":"<UUIDv4>","sample_rate":16000,"encoding":"pcm16",
    "mode":"<utterance|conversation>","language":"ko","patient_id":"<환자 ID, 선택>"}
   - mode: 앱의 발화 모드로 정한다 (아래 "두 모드" 참고)
   - trace_id: 영문·숫자·-·_ 1–64자. 세션마다 새로 만든다. 로그에도 남긴다
   - sample_rate는 필수다
2) 오디오 (바이너리 프레임)
   - PCM16 little-endian, 16kHz, 모노, 헤더 없는 raw
   - 앱이 100ms(3,200 bytes = 1,600샘플)마다 보낸다. **변환이나 버퍼링 없이 받는 즉시 그대로 전달**
3) 종료 (텍스트 프레임)
   {"type":"end","trace_id":"<start와 같은 값>"}

## ML 서버에서 받는 메시지
- 중간 전사 (말하는 동안 약 0.5초마다)
  {"type":"transcript_interim","trace_id":"…","text":"물 조세","stable_text":"물",
   "unstable_text":"조세","confidence":0.62,"is_final":false,"segment_id":0,"audio_end_ms":1312}
  - text는 누적값이다(delta 아님). 이어 붙이지 말고 최신 값으로 통째로 교체
- 최종 전사 (end를 보낸 뒤 보통 0.1–0.4초 안에 1번)
  {"type":"transcript_final","trace_id":"…","text":"물 주세요","confidence":0.98,
   "duration_ms":1420,"model_version":"…","no_speech_prob":0.0,"segment_id":0,"is_last":true}
  - is_last:true인 final을 받으면 세션 끝 → ML 연결을 닫는다
  - text가 빈 문자열이면 무음이다 → 후보 계산 없이 앱에 재질의(ask_again)
- 에러
  {"type":"error","trace_id":"…","code":"…","message":"…"}
  - AUDIO_TOO_LONG: utterance 10초 / conversation 600초 초과. 연결을 끊지 말 것. 곧 final(is_last:true)이 자동으로 온다
    (아래 요구사항 2의 "에러 → ask_again 정리"에서 이 코드는 예외다)
  - INVALID_FORMAT: 우리 쪽 메시지가 잘못됨. 로그를 남기고 앱에는 재질의
  - MODEL_UNAVAILABLE: 종료 코드 1013으로 닫힌다. ML 재시작 중(약 20초) → 앱에 재질의
  - NO_SESSION, INTERNAL: 로그를 남기고 앱에 재질의
- ping/pong: {"type":"ping","trace_id":"…"} → {"type":"pong",…} (필요할 때만)

## 두 모드 (기획서 §4.1a, S16)
앱 상단 토글로 고르는 두 모드를 ML mode로 이렇게 연결한다.
(백엔드 설계의 accuracy_mode는 LLM 재추천 on/off라서 이것과 별개다. 이름이 겹치니 헷갈리지 말 것)

| 앱 모드 | ML mode | 흐름 |
|---|---|---|
| ⭐ 정확도 (기본) | "utterance" | 버튼 누르고 말하기(최대 10초) → end → final 1번(is_last:true) → 후보 3개 → 환자 선택 → TTS |
| ⚡ 실시간 | "conversation" | 말하는 동안 계속 전달. 0.8초 쉴 때마다(쉼 없이 말하면 6초마다) 구간 final(is_last:false)이 오고, 올 때마다 후보 없이 바로 TTS(bypass). 환자가 종료하면 end → 마지막 final(is_last:true) → 세션 끝 |

⚡ 실시간 모드 처리 규칙:
- 구간 final(is_last:false)을 받아도 세션을 닫지 않는다. is_last:true일 때만 닫는다
- 구간 final마다 TTS로 보낸다. segment_id 순서로 오므로, TTS 재생 중에 다음 final이 오면 큐에 넣어 차례로 읽는다(겹쳐 읽지 않기)
- text가 빈 구간 final은 읽지 않고 건너뛴다(세션은 계속). confidence가 기준 미만이면 읽지 않고 앱에 재질의 표시 (기준값은 미정, 설정값으로 뺄 것)
- phrase_log에는 source="voice_bypass", selected_rank=null로 구간마다 저장한다
- interim은 앱 화면용으로 계속 전달한다(현재 구간의 stable_text 진하게 + unstable_text 흐리게)
- 길이: 세션 최대 600초. 넘으면 AUDIO_TOO_LONG 뒤 마지막 final(is_last:true)이 자동으로 온다
- 정확도 모드의 "10초 상한"은 실시간 모드에는 적용하지 않는다

## 앱 쪽 (/utterance/stream)
- 기존 설계(api-endpoints.md)를 따른다: JWT 인증, start {codec} → 오디오 → end
- 앱 start를 받으면 ML 연결을 열고 ML start를 보낸다
- interim은 앱에 그대로 전달한다 (나중에 부분 발화 추천 계산을 끼울 자리를 함수로 분리)
- 앱이 고른 모드(⭐/⚡)를 받는 방법(쿼리 파라미터 또는 start 필드)은 기존 설계에 없다 → 질문 목록에 넣어라
- ⭐ 정확도: final은 후보 계산(phrase_suggestions) 함수로 넘긴다. 지금은 빈 구현 + TODO로 둬도 된다
- ⚡ 실시간: 구간 final마다 TTS 요청 함수로 넘긴다(후보 계산 없음). 지금은 빈 구현 + TODO로 둬도 된다

## 요구사항
1. 앱→ML 전달과 ML→앱 전달을 동시에 돌린다 (태스크 2개)
2. 어느 한쪽이 끊기면 다른 쪽도 정리한다
   - 앱이 end 없이 끊김 → ML에 end를 보내고 닫는다
   - ML이 끊김/에러 → 앱에 ask_again을 보내고 정리한다
3. 타임아웃
   - ML 연결: 3초
   - end를 보낸 뒤 final 대기: 5초. 넘으면 ask_again
4. 환경변수 MARIEUM_MOCK_ASR=1이면 ML 대신 mock 서버(ws://localhost:9001/asr-stream)에 붙는다
5. 로그에 trace_id, 세션 시간, interim 개수, final까지 걸린 시간(end → final ms)을 남긴다
6. 오디오 내용과 전사 텍스트는 로그에 남기지 않는다 (환자 개인정보)

## 테스트
- ML 팀이 준 mock 서버로 로컬에서 시험한다:
    pip install fastapi "uvicorn[standard]"
    python scripts/mock_asr_server.py      # ws://localhost:9001/asr-stream
- mock에서 실시간 모드는 mode="conversation"으로 시작하면 오디오 3초마다 구간 final(is_last:false)이 온다
  (실서버는 쉼 기준이라 간격이 다르다. mock은 AUDIO_TOO_LONG을 내지 않는다)
- 테스트 케이스: 정확도 정상 흐름(interim 여러 개 → final), 실시간 흐름(구간 final 여러 개 → TTS 큐 → end → is_last), 빈 final, AUDIO_TOO_LONG(실서버),
  ML 연결 실패, 앱이 중간에 끊김, final 타임아웃
- 실서버 확인은 백엔드 서버(10.1.1.11)에서 한다

## 결과물
- 중계 코드, 테스트 코드, 실행 방법을 README에 짧게
- 내가 정해야 하는 부분(앱 메시지 이름, 에러를 앱에 어떻게 보여줄지 등)은 임의로 정하지 말고 목록으로 질문해라
````
