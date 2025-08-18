def build_product_prompt(product_meta: dict, K: int) -> str:
    months = ["2024-07","2024-08","2024-09","2024-10","2024-11","2024-12",
              "2025-01","2025-02","2025-03","2025-04","2025-05","2025-06"]
    return f"""
[System] You are a market-research generator. Output STRICT JSON only.
[User]
Generate EXACTLY {K} consumer personas for ONE product in a SINGLE response.
Product:
- id: {product_meta['id']}
- name: {product_meta['name']}, category: {product_meta['category']}
- price_band_krw: {product_meta['price_band_krw']}
- channels: {product_meta['channels']}
- ad_models: {product_meta['ad_models']}
- months: {months}
Constraints:
- Return JSON: {{"product_id":"{product_meta['id']}","personas":[...],
                 "meta":{{"model":"$MODEL","ts":"$TS"}}}}
- personas: exactly {K} objs; each has ≥10 attributes(name/value/weight[0..1]),
  price_sensitivity(0..3), promo_uplift(0..3), ad_uplift(map), channel_pref(sum=1),
  seasonality("01".."12":0.5..1.5), purchase_prob_12(12×0..1), expected_freq_12(12×≥0)
Formatting:
- STRICT JSON, floats ≤3 decimals, no extra text.
""".strip()
