import json, os, re, numpy as np

class LLMClient:
    """제출 전까지 MOCK로 구조 검증. 실제 API로 교체 시 complete_once 구현."""
    def __init__(self, model="gpt-4o-mini"): self.model = model
    def complete_once(self, prompt: str) -> str:
        if os.getenv("LLM_MOCK", "1") == "1": return json.dumps(self._mock(prompt), ensure_ascii=False)
        raise NotImplementedError("실제 LLM API 연동 필요")

    def _mock(self, prompt: str):
        K = int(re.search(r"EXACTLY (\d+)", prompt).group(1))
        pid = re.search(r"- id: ([^\n]+)", prompt).group(1).strip()
        months = ["2024-07","2024-08","2024-09","2024-10","2024-11","2024-12",
                  "2025-01","2025-02","2025-03","2025-04","2025-05","2025-06"]
        def one(i, rng):
            attrs = [{"name": f"attr{j}", "value": float(rng.uniform(0,1)), "weight": float(rng.uniform(0.03,0.15))}
                     for j in range(12)]
            seas = {f"{m:02d}": float(rng.uniform(0.8,1.2)) for m in range(1,13)}
            pp = [float(np.clip(rng.normal(0.15, 0.05), 0.02, 0.6)) for _ in months]
            ff = [float(max(0.2, rng.normal(1.0, 0.2))) for _ in months]
            return {"persona_id": f"{pid}-{i:02d}", "attributes": attrs,
                    "price_sensitivity": float(rng.uniform(0.2, 1.2)),
                    "promo_uplift": float(rng.uniform(0.1, 0.8)),
                    "ad_uplift": {"ModelA": 0.2, "ModelB": 0.1},
                    "channel_pref": {"대형마트":0.5,"편의점":0.2,"온라인":0.3},
                    "seasonality": seas, "purchase_prob_12": pp, "expected_freq_12": ff}
        rng = np.random.default_rng(123)
        return {"product_id": pid, "personas": [one(i, rng) for i in range(1, K+1)], "meta": {"model": self.model}}
