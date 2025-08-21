import yaml
from pathlib import Path

INPUT_YAML = "./configs/products.yaml"          
OUTPUT_TXT = "outputs/prompts/ko_prompts.txt"  # 결과 저장 파일

PERSONA = """
너는 한국 동원식품의 마케팅 전문가다.
소비자 행동과 시장 데이터를 분석하여
신제품의 잠재 시장성과 마케팅 전략을 제안한다.
""".strip()

FEATURES_GUIDE = """
- 연령대, 성별, 소득 수준 등 인구통계학적 요소
- 제품의 주요 성분, 효능, 차별점
- 경쟁사와 비교했을 때의 강점
""".strip()

INSTRUCTION = """
아래 제품 정보를 바탕으로:
1) 주요 타겟 페르소나(최소 5개 속성, 속성 가중치 포함)
2) 페르소나별 니즈/문제점
3) 제품의 제공 가치와 차별화 포인트
4) 예상 마케팅 메시지
5) 2024-07 ~ 2025-06 월별 구매확률(%)·예상구매횟수
를 JSON 하나로만 출력하라(설명 문장 금지).
""".strip()
# ======================

def load_products(yaml_path: str):
    """products.yaml 로드: {'products': [ {...}, ... ]} 또는 [ {...}, ... ] 모두 지원."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if isinstance(data, dict) and "products" in data and isinstance(data["products"], list):
        return data["products"]
    elif isinstance(data, list):
        return data
    raise ValueError("products.yaml 형식이 올바르지 않습니다. 'products' 리스트가 필요합니다.")

def category_path(cat: dict | None) -> str:
    if not isinstance(cat, dict):
        return "미정"
    parts = [cat.get("level_1"), cat.get("level_2"), cat.get("level_3")]
    parts = [str(x).strip() for x in parts if x and str(x).strip()]
    return " > ".join(parts) if parts else "미정"

def features_list(feats) -> str:
    if not feats:
        return "- 없음"
    return "\n".join(f"- {str(x).strip()}" for x in feats if str(x).strip())

def build_prompts(records: list) -> list[str]:
    prompts: list[str] = []
    for rec in records:
        # products.yaml 스키마에 맞춘 키 접근
        pname = str(rec.get("product_name", "미정")).strip()
        cpath = category_path(rec.get("category"))
        feats_block = features_list(rec.get("features"))

        prompt = f"""[페르소나]
{PERSONA}

[분석 기준]
{FEATURES_GUIDE}

[지시사항]
{INSTRUCTION}

[제품 정보]
- 제품명: {pname}
- 카테고리: {cpath}
- 특징:
{feats_block}

[출력 스키마 힌트]
{{
  "product": "{pname}",
  "personas": [
    {{
      "name": "문자열",
      "attributes": [{{"name":"속성명","value":"값","weight_pct":10}}, ... 최소 10개 ],
      "monthly": {{
        "2024-07": {{"purchase_probability_pct": 0, "expected_purchases": 0}},
        "...": "...",
        "2025-06": {{"purchase_probability_pct": 0, "expected_purchases": 0}}
      }},
      "notes": "추론 근거 요약"
    }}
  ],
  "global_assumptions": {{
    "population_reference": "...",
    "seasonality": "...",
    "promotion": "...",
    "price_elasticity_hint": "..."
  }}
}}"""
        prompts.append(prompt.strip())
    return prompts

def save_prompts(prompts: list[str], output_path: str):
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for i, p in enumerate(prompts, 1):
            f.write(f"### Prompt {i}\n{p}\n\n")

if __name__ == "__main__":
    records = load_products(INPUT_YAML)
    prompts = build_prompts(records)
    save_prompts(prompts, OUTPUT_TXT)
    print(f"✅ {len(prompts)}개 프롬프트 생성 완료 → {OUTPUT_TXT}")
