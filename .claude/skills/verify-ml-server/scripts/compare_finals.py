"""같은 음성을 기준선(main)과 후보(브랜치)에 보낸 결과를 final 단위로 비교한다.

stream_client.py --out 으로 저장한 JSON 두 개와, 각 RUN의 server.log를 trace_id로 이어 gen_tokens·ended_eot를 붙인다.
전사 텍스트는 AI Hub 재배포 금지라 기본으로 출력하지 않는다 (길이와 유사도만). --show-text는 GPU 서버 터미널에서만.

  python compare_finals.py <BASE_RUN> <CAND_RUN> client_LSY.json
  python compare_finals.py <BASE_RUN> <CAND_RUN> client_LSY.json --show-text

종료 코드: 끝 토큰으로 끝난(ended_eot) 기준선 final이 후보에서 달라졌으면 1 (정상 문장이 바뀜 = 회귀)
"""

import argparse
import difflib
import json
import os
import sys


def server_finals(run):
    out = {}
    path = os.path.join(run, "server.log")
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8", errors="replace"):
        i = line.find("{")
        if i < 0:
            continue
        try:
            r = json.loads(line[i:])
        except ValueError:
            continue
        if isinstance(r, dict) and r.get("event") == "transcript_final":
            out[(r["trace_id"], r.get("segment_id"))] = r
    return out


def finals(run, name):
    sessions = json.load(open(os.path.join(run, name), encoding="utf-8"))
    logs = server_finals(run)
    res = {}
    for s in sessions:
        for m in s["finals"]:
            key = (s["idx"], m["segment_id"])
            res[key] = (m, logs.get((s["trace_id"], m["segment_id"]), {}))
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔(cp949)에서도 한글이 깨지지 않게
    ap = argparse.ArgumentParser()
    ap.add_argument("base_run")
    ap.add_argument("cand_run")
    ap.add_argument("client_json")
    ap.add_argument("--show-text", action="store_true")
    args = ap.parse_args()

    base, cand = finals(args.base_run, args.client_json), finals(args.cand_run, args.client_json)
    same = changed_ok = changed_runaway = 0
    missing = sorted(set(base) ^ set(cand))
    print(f"[비교] 기준선 final {len(base)}  후보 final {len(cand)}  한쪽에만 있는 구간 {len(missing)}")
    for key in sorted(set(base) & set(cand)):
        (bm, bl), (cm, cl) = base[key], cand[key]
        if bm["text"] == cm["text"]:
            same += 1
            continue
        b_eot = bl.get("ended_eot", True)
        if b_eot:
            changed_ok += 1
        else:
            changed_runaway += 1
        ratio = difflib.SequenceMatcher(None, bm["text"], cm["text"]).ratio()
        tag = "회귀(정상 final이 바뀜)" if b_eot else "폭주 구간(바뀌는 게 목적)"
        print(f"  s{key[0]} seg={key[1]} {tag}: 길이 {len(bm['text'])}→{len(cm['text'])} 유사도 {ratio:.2f} "
              f"gen {bl.get('gen_tokens', '-')}→{cl.get('gen_tokens', '-')} "
              f"eot {bl.get('ended_eot', '-')}→{cl.get('ended_eot', '-')} cap {cl.get('token_cap', '-')}")
        if args.show_text:
            print(f"    기준선: {bm['text']}\n    후보:   {cm['text']}")
    for key in missing:
        print(f"  s{key[0]} seg={key[1]} 한쪽에만 있음 (구간 나누기가 달라졌다: 상한과 무관한 변경이 섞였는지 확인)")
    print(f"[결과] 같음 {same}  정상 final 바뀜 {changed_ok}  폭주 final 바뀜 {changed_runaway}")
    return 1 if changed_ok or missing else 0


if __name__ == "__main__":
    sys.exit(main())
