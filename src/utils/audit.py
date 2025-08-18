from pathlib import Path
import json, hashlib, time

AUDIT_DIR = Path("outputs/audit"); AUDIT_DIR.mkdir(parents=True, exist_ok=True)
LOG = AUDIT_DIR / "llm_calls.jsonl"

def _sha256(s: str) -> str: return hashlib.sha256(s.encode("utf-8")).hexdigest()

def record_llm_call(product_id: str, prompt: str, model: str, submit: bool):
    rec = {"ts": int(time.time()), "product_id": product_id,
           "prompt_hash": _sha256(prompt), "model": model, "submit": submit}
    with LOG.open("a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def assert_single_call_per_product(submit: bool):
    if not submit or not LOG.exists(): return
    seen = {}
    for line in LOG.read_text(encoding="utf-8").splitlines():
        j = json.loads(line)
        if j.get("submit"): seen[j["product_id"]] = seen.get(j["product_id"], 0) + 1
    viol = [k for k,v in seen.items() if v > 1]
    if viol: raise RuntimeError(f"[COMPLIANCE] 제품당 LLM 호출 1회 초과: {viol}")
