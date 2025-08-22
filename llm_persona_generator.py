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
    Please generate consumer persona number {persona_number} for the following product. This persona should be a typical consumer who is highly likely to purchase this specific product, considering its unique features and characteristics. Ensure this persona is distinct from other personas you might generate for this product, highlighting a primary purchasing motivation or characteristic.

    Product Name: {product_name}
    Product Category: {category_1} > {category_2} > {category_3}
    Product Features: {features}

    The persona must have the following attributes. Assign a value (e.g., a specific age, a description, or a rating 1-10) and a weight (1-10) to each attribute, indicating its importance in the purchase decision and how it is influenced by the product's features. For '평균 구매 수량', please provide a numerical value (e.g., 1, 2, 3) representing the typical quantity purchased, usually between 1 and 5.

    **Crucially, you must also generate the '제품 적합도' and '구매 결정 핵심 이유' attributes.**
    - For '제품 적합도', evaluate all other persona attributes to determine how likely this persona is to purchase this specific product, and provide a score from 1 (not at all likely) to 10 (extremely likely).
    - For '구매 결정 핵심 이유', provide a brief, clear rationale for the '제품 적합도' score, linking the persona's traits to the product's features.

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

def main(dry_run=False, num_personas_to_generate=1):
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

        # Loop to generate multiple personas
        for i in range(num_personas_to_generate):
            print(f"  Generating persona {i+1}/{num_personas_to_generate} for {product_name}")

            # Modify chain_input to encourage diversity
            chain_input = {
                "product_name": product_name,
                "category_1": product_info['category_level_1'],
                "category_2": product_info['category_level_2'],
                "category_3": product_info['category_level_3'],
                "features": product_info['product_feature'],
                "attributes": attributes_str,
                "yaml_attributes": yaml_attributes_str,
                "persona_number": i + 1 # Add persona number to prompt
            }

            if dry_run:
                # In dry run, just create a dummy prompt for inspection
                # The actual prompt construction happens via LangChain's chain.invoke
                # We'll print the input to the chain here for inspection
                print(f"\n--- Dry Run: Chain Input for {product_name} (Persona {i+1}) ---\n")
                for key, value in chain_input.items():
                    if key == "attributes" or key == "yaml_attributes":
                        # For long strings, print only a snippet or indicate content
                        print(f"{key}: (content from persona_attributes.yml)")
                    else:
                        print(f"{key}: {value}")
                print("--------------------------------------------------\n")

                # Still write a placeholder file to indicate generation
                prompt_filename = f"outputs/prompts/llm_prompt_{sanitize_filename(product_name)}_persona_{i+1}.txt"
                with open(prompt_filename, 'w', encoding='utf-8') as f:
                    f.write(f"System: (See llm_persona_generator.py system_message)\nUser: (See llm_persona_generator.py user_message_template with input below)\n\nChain Input for {product_name} (Persona {i+1}):\n{chain_input}")
            else:
                # In a real run, invoke the chain and save the persona
                try:
                    llm_output = persona_chain.invoke(chain_input)
                    cleaned_yaml = llm_output.strip().replace("```yaml", "").replace("```", "").strip()
                    persona_data = yaml.safe_load(cleaned_yaml)

                    # Modify persona_filename for multiple personas
                    persona_filename = f"outputs/personas/{sanitize_filename(product_name)}_persona_{i+1}.yml"
                    with open(persona_filename, 'w', encoding='utf-8') as f:
                        yaml.dump(persona_data, f, allow_unicode=True)
                    print(f"  -> Saved persona to {persona_filename}")

                except Exception as e:
                    print(f"  -> Error generating persona {i+1} for {product_name}: {e}")

if __name__ == '__main__':
    main(dry_run=False, num_personas_to_generate=10)
