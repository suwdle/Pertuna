from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Union
Number = Union[int, float]

class PersonaAttribute(BaseModel):
    name: str; value: Union[str, Number]; weight: float = Field(ge=0, le=1)

class Persona(BaseModel):
    persona_id: str
    attributes: List[PersonaAttribute]      # ≥10
    price_sensitivity: float                # η
    promo_uplift: float                     # γ
    ad_uplift: Dict[str, float]             # δ_m
    channel_pref: Dict[str, float]          # 합=1
    seasonality: Dict[str, float]           # "01"~"12": 0.5~1.5 권장
    purchase_prob_12: List[float]           # len=12, 0~1
    expected_freq_12: List[float]           # len=12, ≥0

    @field_validator("channel_pref")
    @classmethod
    def _sum_to_one(cls, v):
        s = sum(v.values());  return {k: (val/s if s>0 else 0.0) for k,val in v.items()}

class PersonaBag(BaseModel):
    product_id: str
    personas: List[Persona]                 # 길이 K
    meta: Dict[str, str]
