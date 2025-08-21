import pandas as pd
import yaml
import re

# CSV 파일 경로
csv_path = "./data/product_info.csv"
yaml_output_path = "./configs/products.yaml"

# CSV 로드
df = pd.read_csv(csv_path)

# 필드 정리
def clean_features(features):
    if pd.isna(features):
        return []
    return [f.strip() for f in re.split(r"[,\|;/·•、]+", str(features)) if f.strip()]

products = []
for _, row in df.iterrows():
    product = {
        "product_name": row["product_name"],
        "category": {
            "level_1": row["category_level_1"],
            "level_2": row["category_level_2"],
            "level_3": row["category_level_3"]
        },
        "features": clean_features(row["product_feature"])
    }
    products.append(product)

# YAML 파일로 저장
with open(yaml_output_path, "w", encoding="utf-8") as f:
    yaml.dump({"products": products}, f, allow_unicode=True, sort_keys=False)
