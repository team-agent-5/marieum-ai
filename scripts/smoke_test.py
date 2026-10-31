"""모델 단독 스모크 테스트 (CALIBRATION.md §4, 단계 5).

확인하는 것:
  1. 노트북 방식(forced_decoder_ids)이 고정한 transformers 버전에서 동작하는지
  2. generation_config의 디코딩 기본값 (#15)
  3. 서버용 옵션(return_dict_in_generate, output_scores, max_new_tokens)을 붙여도
     노트북 호출과 전사가 같은지
  4. scores와 생성 토큰이 순서대로 맞는지, 강제 토큰이 scores에 들어가는지 (#14)
  5. GPU 메모리와 1건당 시간(feat, gen)

사용법:
  python scripts/smoke_test.py --model models/ASR_Whisper_Disease_General a.wav b.wav
"""

import argparse
import json
import time
import wave

import numpy as np
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MAX_NEW_TOKENS = 128


def load_wav(path):
    # 노트북은 torchaudio.load를 썼지만 torchaudio 2.9+는 torchcodec이 필요하다.
    # 서버도 WAV를 직접 파싱할 예정이라 표준 라이브러리로 읽는다.
    with wave.open(path, "rb") as w:
        sr, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        assert width == 2, f"PCM 16-bit만 지원: sampwidth={width}"
        pcm = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
    x = torch.from_numpy(pcm).reshape(-1, ch).T  # (ch, n)
    if ch > 1:
        x = x.mean(dim=0, keepdim=True)
    if sr != 16000:
        x = torchaudio.functional.resample(x, orig_freq=sr, new_freq=16000)
    return x.squeeze(0), sr, n / sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("wavs", nargs="+")
    args = ap.parse_args()

    dev = torch.device("cuda")
    t0 = time.perf_counter()
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(dev).eval()
    torch.cuda.synchronize()
    print(f"[load] {time.perf_counter() - t0:.1f}s, dtype={model.dtype}, "
          f"gpu_alloc={torch.cuda.memory_allocated() / 2**20:.0f}MiB")

    gc = model.generation_config
    print("[generation_config]", json.dumps({
        k: getattr(gc, k, None) for k in
        ["num_beams", "do_sample", "temperature", "max_length", "length_penalty",
         "no_repeat_ngram_size", "repetition_penalty", "return_timestamps"]
    }, ensure_ascii=False))

    forced_decoder_ids = processor.get_decoder_prompt_ids(language="ko", task="transcribe")
    print("[forced_decoder_ids]", forced_decoder_ids)
    tok = processor.tokenizer
    special = set(tok.all_special_ids)

    for path in args.wavs:
        print(f"\n===== {path}")
        audio, sr, dur = load_wav(path)
        print(f"[audio] orig_sr={sr}, dur={dur:.2f}s")

        t = time.perf_counter()
        feats = processor(audio.numpy(), sampling_rate=16000, return_tensors="pt").input_features.to(dev)
        torch.cuda.synchronize()
        feat_ms = (time.perf_counter() - t) * 1000

        # 1) 노트북과 똑같은 호출
        with torch.inference_mode():
            ids_nb = model.generate(feats, forced_decoder_ids=forced_decoder_ids)
        text_nb = processor.batch_decode(ids_nb, skip_special_tokens=True)[0]

        # 2) 서버 호출 (§4)
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                feats,
                forced_decoder_ids=forced_decoder_ids,
                return_dict_in_generate=True,
                output_scores=True,
                output_logits=True,
                max_new_tokens=MAX_NEW_TOKENS,
                max_time=2.5,
            )
        torch.cuda.synchronize()
        gen_ms = (time.perf_counter() - t) * 1000
        seq = out.sequences[0].tolist()
        text = processor.batch_decode(out.sequences, skip_special_tokens=True)[0]

        print(f"[notebook] {text_nb!r}")
        print(f"[server]   {text!r}  same={text == text_nb}")
        print(f"[tokens] sequences({len(seq)})={seq}")
        print(f"[tokens] decoded={tok.convert_ids_to_tokens(seq)}")
        print(f"[scores] len(scores)={len(out.scores)}, len(logits)={len(out.logits)}")

        # scores[i]가 sequences의 어느 위치와 맞는지: 뒤에서부터 맞춘다
        gen = seq[-len(out.scores):]
        argmax = [int(s[0].argmax()) for s in out.scores]
        print(f"[align] gen_tail={gen}")
        print(f"[align] argmax(scores)==gen_tail: {argmax == gen}")
        print(f"[align] 강제 토큰이 scores에 포함?: prompt={seq[:len(seq) - len(out.scores)]}")

        p_proc, p_raw = [], []
        for i, tid in enumerate(gen):
            if tid in special:
                continue
            p_proc.append(torch.softmax(out.scores[i][0].float(), -1)[tid].item())
            p_raw.append(torch.softmax(out.logits[i][0].float(), -1)[tid].item())
        conf_proc = float(np.mean(p_proc)) if p_proc else 0.0
        conf_raw = float(np.mean(p_raw)) if p_raw else 0.0
        print(f"[confidence] text_tokens={len(p_proc)}, processed={conf_proc:.4f}, raw={conf_raw:.4f}")
        print(f"[timing] feat={feat_ms:.0f}ms, gen={gen_ms:.0f}ms, gen_tokens={len(out.scores)}")

    print(f"\n[gpu] max_alloc={torch.cuda.max_memory_allocated() / 2**20:.0f}MiB")
    import transformers, tokenizers, huggingface_hub
    print("[versions]", json.dumps({
        "python": __import__("sys").version.split()[0], "torch": torch.__version__,
        "torchaudio": torchaudio.__version__, "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__, "huggingface_hub": huggingface_hub.__version__,
        "numpy": np.__version__}))


if __name__ == "__main__":
    main()
