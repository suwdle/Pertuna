import pandas as pd
import yaml
import os
import re
import numpy as np

def sanitize_filename(name):
    """Sanitizes a string to be used as a filename."""
    return re.sub(r'[^a-zA-Z0-9가-힣\s-]', '', name).replace(' ', '_')

def load_personas(product_name, num_personas_to_load=1):
    """
    Loads persona data for a given product.
    """
    safe_name = sanitize_filename(product_name)
    loaded_personas = []
    for i in range(1, num_personas_to_load + 1):
        persona_file = f"outputs/personas/{safe_name}_persona_{i}.yml"
        if os.path.exists(persona_file):
            with open(persona_file, 'r', encoding='utf-8') as f:
                persona_data = yaml.safe_load(f)
                loaded_personas.append(persona_data)
        else:
            print(f"Warning: Persona file not found for {product_name} (Persona {i}). Using a dummy persona for this slot.")
            loaded_personas.append({
                'name': f'dummy_persona_{i}',
                'attributes': [{'name': '평균 구매 수량', 'value': 1}]
            })
    return loaded_personas

def simulate_monthly_demand(persona):
    """
    Simulates monthly demand based solely on the persona's '평균 구매 수량'.
    """
    def safe_float(value, default=0.0):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    persona_attrs = {attr['name']: attr.get('value') for attr in persona.get('attributes', [])}
    avg_purchase_quantity = safe_float(persona_attrs.get('평균 구매 수량', 1.0), 1.0)
    
    # Ensure lambda is non-negative
    lambda_val = max(0.0, avg_purchase_quantity)
    
    return np.random.poisson(lam=lambda_val)

if __name__ == '__main__':
    with open('configs/simulator.yml', 'r', encoding='utf-8') as f:
        sim_config = yaml.safe_load(f)

    num_simulations_per_persona = sim_config.get('N_trials', 10000)
    market_multiplier = sim_config.get('pop_scale', 1.0)
    num_personas_to_load = sim_config.get('K', 10)

    product_df = pd.read_csv('data/product_info.csv')
    submission_df = pd.read_csv('data/sample_submission.csv')
    submission_df = submission_df.set_index('product_name')

    for index, product_info in product_df.iterrows():
        product_name = product_info['product_name']
        print(f"Simulating for: {product_name} ({num_simulations_per_persona} sims per persona)")

        loaded_personas = load_personas(product_name, num_personas_to_load=num_personas_to_load)
        if not loaded_personas:
            print(f"  -> No personas found for {product_name}. Skipping.")
            continue

        monthly_product_sales = [0] * 12

        for month_idx in range(12):
            monthly_purchases_sum_for_product = 0
            for persona in loaded_personas:
                for _ in range(num_simulations_per_persona):
                    monthly_purchases_sum_for_product += simulate_monthly_demand(persona)
            
            # Scale the sum of simulations by the market multiplier
            scaled_sales = round(monthly_purchases_sum_for_product * market_multiplier)
            monthly_product_sales[month_idx] = scaled_sales

        submission_df.loc[product_name] = monthly_product_sales

    output_path = 'outputs/submission_baseline.csv'
    submission_df.reset_index().to_csv(output_path, index=False)

    print(f"Simulation complete. Baseline submission file saved to {output_path}")
