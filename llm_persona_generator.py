import pandas as pd
import yaml
import os
import re
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

# This script requires OPENAI_API_KEY to be set as an environment variable.

def sanitize_filename(name):
    """Sanitizes a string to be used as a filename."""
    return re.sub(r'[^a-zA-Z0-9가-힣\s-]', '', name).replace(' ', '_')

def create_persona_generation_chain(llm):
    """
    Creates a LangChain chain for generating consumer personas.
    """
    system_message = """
You are an expert market researcher. Your task is to create a detailed and realistic consumer persona based on the provided product information and current market insights. The persona should represent a typical consumer who is likely to purchase the product, considering their sensitivity to various market factors. The output should be in YAML format.
"""

    user_message_template = """
    Please generate a single consumer persona for the following product, considering the provided market insights:

    Product Name: {product_name}
    Product Category: {category_1} > {category_2} > {category_3}
    Product Features: {features}

    --- Current Market Insights ---
    - 소비자 심리: 소비 심리지수(CSI)가 상승하지만, 식품 소비자 물가 지수(CPI)는 급등하여 소비자들이 가격에 민감한 '계획 소비'를 지향하고 있습니다.
    - 마케팅 효과: 광고 캠페인 및 SNS 챌린지가 수요를 직접적으로 견인하는 핵심 변수입니다.
    - 계절성 확장: 소셜 트렌드가 제품 활용 범위를 확장시키고 있어 복합적인 계절성 지수 도입이 필요합니다.
    - 주요 식품 카테고리별 동향:
        - 호상 발효유(요거트): '그릭 요거트' 등 프리미엄화 및 '건강' 트렌드.
        - 참치캔: 마케팅 캠페인(예: 안유진)이 판매를 견인.
        - 조미료(액상): '집밥', '만능 소스' 트렌드 확대로 시장 성장.
        - 고급 축산캔(햄): '명절 선물 세트' 수요 및 '저나트륨 햄' 건강 관심.
        - 컵커피: 전통적인 여름 매출 외에 '컵빙수' 등 소셜 트렌드 영향.
    ---

    The persona must have the following attributes. Assign a value (e.g., a specific age, a description, or a rating 1-10) and a weight (1-10) to each attribute, indicating its importance in the purchase decision and how it is influenced by the market insights provided.

    --- Persona Attributes ---
    {attributes}
    ---

    Please provide the output in the following YAML format. Ensure all attributes from the 'Persona Attributes' section are included with appropriate 'value' and 'weight'.

    ```yaml
    name: (Persona Name)
    attributes:
    {yaml_attributes}
    ```
    """

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", user_message_template)
    ])

    chain = prompt_template | llm | StrOutputParser()
    return chain

def main(dry_run=True):
    """
    Main function to generate personas for all products.
    """
    product_df = pd.read_csv('data/product_info.csv')

    with open('configs/persona_attributes.yml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    attributes_list = config['persona_attributes']
    attributes_str = "\n".join([f"- {attr['name']}: ({attr['description']})" for attr in attributes_list])
    yaml_attributes_str = "\n".join([f"  - name: {attr['name']}\n    value: (value)\n    weight: (weight)" for attr in attributes_list])

    # Create directories if they don't exist
    os.makedirs("outputs/personas", exist_ok=True)
    os.makedirs("outputs/prompts", exist_ok=True)

    if not dry_run:
        # Initialize the LLM only when not in dry_run mode
        llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
        persona_chain = create_persona_generation_chain(llm)

    for index, product_info in product_df.iterrows():
        product_name = product_info['product_name']
        print(f"Generating persona for: {product_name}")

        chain_input = {
            "product_name": product_name,
            "category_1": product_info['category_level_1'],
            "category_2": product_info['category_level_2'],
            "category_3": product_info['category_level_3'],
            "features": product_info['product_feature'],
            "attributes": attributes_str,
            "yaml_attributes": yaml_attributes_str
        }

        if dry_run:
            # In dry run, just create a dummy prompt for inspection
            # The actual prompt construction happens via LangChain's chain.invoke
            # We'll print the input to the chain here for inspection
            print(f"\n--- Dry Run: Chain Input for {product_name} ---")
            for key, value in chain_input.items():
                if key == "attributes" or key == "yaml_attributes":
                    # For long strings, print only a snippet or indicate content
                    print(f"{key}: (content from persona_attributes.yml)")
                else:
                    print(f"{key}: {value}")
            print("--------------------------------------------------\n")

            # Still write a placeholder file to indicate generation
            prompt_filename = f"outputs/prompts/llm_prompt_{sanitize_filename(product_name)}.txt"
            with open(prompt_filename, 'w', encoding='utf-8') as f:
                f.write(f"System: (See llm_persona_generator.py system_message)\nUser: (See llm_persona_generator.py user_message_template with input below)\n\nChain Input for {product_name}:\n{chain_input}")
        else:
            # In a real run, invoke the chain and save the persona
            try:
                llm_output = persona_chain.invoke(chain_input)
                cleaned_yaml = llm_output.strip().replace("```yaml", "").replace("```", "").strip()
                persona_data = yaml.safe_load(cleaned_yaml)

                persona_filename = f"outputs/personas/{sanitize_filename(product_name)}_persona.yml"
                with open(persona_filename, 'w', encoding='utf-8') as f:
                    yaml.dump(persona_data, f, allow_unicode=True)
                print(f"  -> Saved persona to {persona_filename}")

            except Exception as e:
                print(f"  -> Error generating persona for {product_name}: {e}")

if __name__ == '__main__':
    main(dry_run=False)
