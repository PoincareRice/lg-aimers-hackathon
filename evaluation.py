"""
EXAONE-4.0-1.2B 기반 리더보드 Score 평가 모듈
Score = max(0.5 * PerfNorm + 0.5 * SpeedNorm, 0)

PerfNorm  = base_ppl / model_ppl
SpeedNorm = 1 - (model_tpt / base_tpt)
"""

import gc
import time
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config (기본값) ───────────────────────────────────────────────────────────
BASE_MODEL_ID           = "./base_model"
EVAL_MODEL_ID           = "./model"
NUM_PERF_SAMPLES        = 100
NUM_SPEED_SAMPLES       = 50    # 더미 고정 입력 반복 횟수
MAX_SEQ_LENGTH          = 512
GENERATE_MAX_NEW_TOKENS = 64
DATASET_ID              = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT           = "train"

# 속도 측정용 더미 입력 고정 설정
DUMMY_INPUT_LEN         = 128   # 고정 입력 길이


# ── 모델 로드 ─────────────────────────────────────────────────────────────────

def load_model(model_id: str, label: str):
    print(f"\n[LOAD] {label} ({model_id})")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,       # torch_dtype → dtype (deprecation 경고 제거)
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    model.eval()
    alloc = torch.cuda.memory_allocated(0) / 1e9
    print(f"  GPU 사용: {alloc:.2f} GB")
    return model, tokenizer


# ── 데이터 준비 ───────────────────────────────────────────────────────────────

def prepare_texts(tokenizer, n_perf):
    print(f"\n[DATA] 데이터 로드 중 ({n_perf}개)...")
    ds = load_dataset(
        DATASET_ID,
        split=f"{DATASET_SPLIT}[-{n_perf}:]"
    )

    texts = []
    for ex in ds:
        try:
            text = tokenizer.apply_chat_template(
                ex["conversations"],
                add_generation_prompt=False,
                tokenize=False,
            )
            if text and len(text.strip()) > 20:
                texts.append(text.strip())
        except Exception:
            continue

    print(f"  성능 평가용: {len(texts[:n_perf])}개")
    return texts[:n_perf]


# ── Perplexity 평가 ───────────────────────────────────────────────────────────

def evaluate_perplexity(model, tokenizer, texts, max_len, label):
    print(f"\n[PERF] {label} Perplexity 계산 중...")
    total_nll, total_tokens, skipped = 0.0, 0, 0

    with torch.no_grad():
        for text in tqdm(texts, desc=f"  PPL ({label})"):
            enc = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=max_len)
            input_ids = enc["input_ids"].to(model.device)
            if input_ids.shape[1] < 4:
                skipped += 1
                continue
            try:
                out = model(input_ids, labels=input_ids)
                nll = out.loss.item()
                if np.isfinite(nll):
                    total_nll    += nll * input_ids.shape[1]
                    total_tokens += input_ids.shape[1]
            except Exception:
                skipped += 1

    ppl = np.exp(total_nll / total_tokens)
    print(f"  Perplexity: {ppl:.4f}  (skipped: {skipped})")
    return ppl


# ── 속도 평가 (단일 프롬프트 반복 측정) ──────────────────────────────────────

def evaluate_speed(model, tokenizer, num_runs, input_len, max_new_tokens, label):
    """
    고정 프롬프트로 num_runs회 반복 측정 → 평균 토큰당 시간 반환
    time.time() 사용 (단순하고 안정적)
    """
    print(f"\n[SPEED] {label} 속도 측정 중 ({num_runs}회 반복)...")

    test_prompt = "인공지능의 미래에 대해 설명해줘."
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    # 워밍업 3회
    print("  워밍업 중...")
    for _ in range(3):
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

    # 실제 측정
    total_time, total_tokens = 0.0, 0

    for _ in tqdm(range(num_runs), desc=f"  Speed ({label})"):
        start = time.time()

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        elapsed = time.time() - start
        n_new = len(out[0]) - inputs["input_ids"].shape[1]
        if n_new > 0:
            total_time   += elapsed
            total_tokens += n_new

    tpt = total_time / total_tokens
    print(f"  토큰당 시간: {tpt*1000:.4f} ms/token  ({total_tokens/total_time:.1f} tok/s)")
    return tpt


# ── Score 계산 ────────────────────────────────────────────────────────────────

def compute_score(base_ppl, model_ppl, base_tpt, model_tpt):
    perf_norm  = base_ppl / model_ppl
    speed_norm = 1.0 - (model_tpt / base_tpt)
    score      = max(0.5 * perf_norm + 0.5 * speed_norm, 0.0)
    return perf_norm, speed_norm, score


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    base_model_id  = BASE_MODEL_ID,
    eval_model_id  = EVAL_MODEL_ID,
    perf_samples   = NUM_PERF_SAMPLES,
    speed_runs     = NUM_SPEED_SAMPLES,
    max_len        = MAX_SEQ_LENGTH,
    max_new_tokens = GENERATE_MAX_NEW_TOKENS,
    dummy_input_len = DUMMY_INPUT_LEN,
):
    total_start = time.perf_counter()
    print("=" * 60)
    print("  리더보드 Score 평가 시작")
    print("=" * 60)
    print(f"  기준 모델   : {base_model_id}")
    print(f"  평가 모델   : {eval_model_id}")
    print(f"  성능 샘플   : {perf_samples}")
    print(f"  속도 반복   : {speed_runs}회 (더미 고정 입력 len={dummy_input_len})")
    print(f"  max_len     : {max_len} / max_new_tokens: {max_new_tokens}")

    # ── Base 모델 ──────────────────────────────────────────────────────────
    base_model, base_tok = load_model(base_model_id, "Base Model")
    perf_texts = prepare_texts(base_tok, perf_samples)

    base_ppl = evaluate_perplexity(base_model, base_tok, perf_texts, max_len, "Base")
    base_tpt = evaluate_speed(base_model, base_tok, speed_runs, dummy_input_len, max_new_tokens, "Base")

    del base_model
    torch.cuda.empty_cache()
    gc.collect()

    # ── 평가 모델 ──────────────────────────────────────────────────────────
    eval_model, eval_tok = load_model(eval_model_id, "Eval Model")

    model_ppl = evaluate_perplexity(eval_model, eval_tok, perf_texts, max_len, "Eval")
    model_tpt = evaluate_speed(eval_model, eval_tok, speed_runs, dummy_input_len, max_new_tokens, "Eval")

    del eval_model
    torch.cuda.empty_cache()
    gc.collect()

    # ── Score ──────────────────────────────────────────────────────────────
    perf_norm, speed_norm, score = compute_score(base_ppl, model_ppl, base_tpt, model_tpt)
    total_elapsed = time.perf_counter() - total_start

    print("\n" + "=" * 60)
    print("  📊 평가 결과")
    print("=" * 60)
    print(f"  [Perplexity]")
    print(f"    Base  PPL : {base_ppl:.4f}")
    print(f"    Model PPL : {model_ppl:.4f}")
    print(f"  [Speed (ms/token)]")
    print(f"    Base  TPT : {base_tpt*1000:.4f} ms")
    print(f"    Model TPT : {model_tpt*1000:.4f} ms")
    print(f"    속도 배율 : {base_tpt/model_tpt:.2f}x  {'✅ 빠름' if model_tpt < base_tpt else '❌ 느림'}")
    print(f"  [Score]")
    print(f"    PerfNorm  : {perf_norm:.4f}  (기준: 1.0)")
    print(f"    SpeedNorm : {speed_norm:.4f}  (기준: 0.0)")
    print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"    🏆 Score  : {score:.4f}")
    print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  총 소요 시간: {total_elapsed/60:.1f}분 ({total_elapsed:.1f}초)")
    print("=" * 60)

    print("\n  [해석]")
    if perf_norm >= 1.0:
        print(f"  ✅ PerfNorm {perf_norm:.4f}: base 대비 성능 유지/향상")
    else:
        print(f"  ⚠️  PerfNorm {perf_norm:.4f}: base 대비 성능 {(1-perf_norm)*100:.1f}% 저하")

    if speed_norm >= 0:
        print(f"  ✅ SpeedNorm {speed_norm:.4f}: base 대비 {speed_norm*100:.1f}% 빠름")
    else:
        print(f"  ⚠️  SpeedNorm {speed_norm:.4f}: base 대비 {abs(speed_norm)*100:.1f}% 느림")

    if score >= 1.0:
        print(f"  🎉 Score {score:.4f}: base 모델 초과 달성!")
    elif score >= 0.8:
        print(f"  👍 Score {score:.4f}: 양호한 수준")
    else:
        print(f"  🔧 Score {score:.4f}: 개선 여지 있음")

    return {"perf_norm": perf_norm, "speed_norm": speed_norm, "score": score}


# ── 직접 실행할 때만 main() 호출 ──────────────────────────────────────────────
if __name__ == "__main__":
    main()