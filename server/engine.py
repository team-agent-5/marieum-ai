"""Whisper 추론 (CALIBRATION.md §4, §0.5).

연구 노트북 방식(forced_decoder_ids, greedy)을 그대로 쓰고, 여러 세션의 음성을 한 번에(batch) 돌린다.
인코더는 한 번만 돌려서 no_speech_prob 계산과 generate()가 같이 쓴다.
"""

import time

import numpy as np
import torch
from transformers import (StoppingCriteria, StoppingCriteriaList, WhisperForConditionalGeneration,
                          WhisperProcessor)
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import logging as hf_logging

# generate()마다 "output_scores is not valid" 경고가 찍힌다. 동작에는 영향 없음 (§4 스모크 테스트)
hf_logging.set_verbosity_error()

SR = 16000


class _RowCaps(StoppingCriteria):
    """줄마다 생성 토큰 수가 자기 상한에 닿으면 그 줄만 멈춘다."""

    def __init__(self, caps):
        self.caps = caps
        self.start = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.start is None:  # 첫 호출은 첫 생성 토큰을 붙인 뒤다
            self.start = input_ids.shape[1] - 1
        return (input_ids.shape[1] - self.start) >= self.caps


def row_result(tokens, token_logps, cap, eot, special):
    """한 줄의 앞 cap개 생성 토큰에서 (특수 토큰을 뺀 토큰별 logprob, EOT로 끝났는지)."""
    # Whisper는 pad_token_id가 EOT라 상한으로 멈춘 줄 뒤에 EOT가 채워진다. 줄 전체를 보면 끝난 것으로 잡힌다
    lp = []
    for tid, logp in zip(tokens[:cap], token_logps[:cap]):
        if tid == eot:
            return lp, True
        if tid not in special:
            lp.append(float(logp))
    return lp, False


class AsrEngine:
    def __init__(self, model_dir, dtype="fp32", device="cuda"):
        self.dev = torch.device(device)
        self.dtype = {"fp32": torch.float32, "fp16": torch.float16}[dtype]
        t = time.perf_counter()
        self.processor = WhisperProcessor.from_pretrained(model_dir)
        self.model = (WhisperForConditionalGeneration.from_pretrained(model_dir, torch_dtype=self.dtype)
                      .to(self.dev).eval())
        self.load_s = time.perf_counter() - t
        tok = self.processor.tokenizer
        self.fdi = self.processor.get_decoder_prompt_ids(language="ko", task="transcribe")
        self.special = set(tok.all_special_ids)
        self.eot = tok.eos_token_id
        self.sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
        # whisper-small 다국어 어휘에서 무음 토큰 이름은 <|nocaptions|> (나중 버전은 <|nospeech|>)
        self.no_speech_id = next(
            i for i in (tok.convert_tokens_to_ids(t) for t in ("<|nospeech|>", "<|nocaptions|>"))
            if i is not None and i != tok.unk_token_id)

    @torch.inference_mode()
    def transcribe(self, audios, caps):
        """audios: 16kHz float32 배열 목록, caps: 줄별 출력 토큰 상한. 결과 dict 목록 (같은 순서)."""
        t0 = time.perf_counter()
        feats = self.processor([a for a in audios], sampling_rate=SR, return_tensors="pt").input_features
        feats = feats.to(self.dev, self.dtype)
        t1 = time.perf_counter()

        enc = self.model.model.encoder(feats)
        # no_speech_prob: <|startoftranscript|> 다음 자리에서 무음 토큰의 확률 (openai/whisper와 같은 정의)
        sot = torch.full((len(audios), 1), self.sot, device=self.dev)
        dec = self.model.model.decoder(input_ids=sot, encoder_hidden_states=enc.last_hidden_state)
        first = self.model.proj_out(dec.last_hidden_state[:, -1]).float()
        no_speech = torch.softmax(first, -1)[:, self.no_speech_id].tolist()

        out = self.model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=enc.last_hidden_state),
            forced_decoder_ids=self.fdi, return_dict_in_generate=True, output_scores=True,
            max_new_tokens=max(caps),
            stopping_criteria=StoppingCriteriaList([_RowCaps(torch.tensor(caps, device=self.dev))]))
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        T = len(out.scores)
        logp = torch.stack([torch.log_softmax(s.float(), -1) for s in out.scores], 1)  # [B, T, V]
        gen = out.sequences[:, -T:]
        tok_logp = logp.gather(-1, gen.unsqueeze(-1)).squeeze(-1).cpu().numpy()
        gen = gen.cpu().tolist()
        texts = self.processor.batch_decode(out.sequences, skip_special_tokens=True)

        results = []
        for b in range(len(audios)):
            lp, ended = row_result(gen[b], tok_logp[b], caps[b], self.eot, self.special)
            results.append({
                "text": texts[b].strip(),
                "confidence": round(float(np.mean(np.exp(lp))), 4) if lp else 0.0,
                "avg_logprob": float(np.mean(lp)) if lp else float("-inf"),
                "no_speech_prob": round(float(no_speech[b]), 4),
                "gen_tokens": len(lp) + (1 if ended else 0),
                "ended_eot": ended,
            })
        timing = {"feat_ms": (t1 - t0) * 1000, "gpu_ms": (t2 - t1) * 1000, "batch": len(audios)}
        return results, timing
