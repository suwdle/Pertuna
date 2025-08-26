import pandas as pd
import yaml
import os
import re
import numpy as np
import datetime

def sanitize_filename(name):
    """Sanitizes a string to be used as a filename."""
    return re.sub(r'[^a-zA-Z0-9가-힣\s-]', '', name).replace(' ', '_')

def load_personas(product_name, num_personas_to_load=1):
    """
    Loads persona data for a given product.
    It looks for files named after the product in the personas directory.
    If not found, it returns a list containing a dummy persona.
    """
    safe_name = sanitize_filename(product_name)
    
    loaded_personas = []
    for i in range(1, num_personas_to_load + 1):
        persona_file = f"outputs/personas/{safe_name}_persona_{i}.yml"
        if os.path.exists(persona_file):
            with open(persona_file, 'r', encoding='utf-8') as f:
                try:
                    persona_data = yaml.safe_load(f)
                    if persona_data and 'attributes' in persona_data:
                        loaded_personas.append(persona_data)
                    else:
                        print(f"Warning: Persona file {persona_file} is missing attributes. Skipping.")
                except yaml.YAMLError as e:
                    print(f"Warning: Error parsing {persona_file}. Skipping. Error: {e}")
        else:
            print(f"Warning: Persona file not found: {persona_file}. Skipping.")
    return loaded_personas

def load_external_data():
    """
    Loads or generates mock external market data based on report.md insights.
    """
    # Dates for the 12-month prediction period (July 2024 - June 2025)
    dates = [datetime.date(2024, 7, 1) + datetime.timedelta(days=30*i) for i in range(12)]

    # Mock data for indices (simplified for demonstration)
    external_data = {
        "dates": dates,
        "food_cpi_impact": [1.0, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.89, 0.88], # Higher CPI means lower impact (less purchasing power)
        "marketing_boost_index": [1.0, 1.05, 1.1, 1.05, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], # Example: boost in first few months
        "health_premium_index": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], # Placeholder, could vary by product category
        "home_cooking_index": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], # Placeholder
        "seasonal_index": [1.2, 1.3, 1.1, 1.0, 0.9, 0.8, 0.9, 1.0, 1.1, 1.2, 1.1, 1.0], # Example: higher in summer/winter
        "holiday_index": [1.2, 1.2, 1.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5], # Example: boost in Nov (Chuseok) and Dec (year-end)
        "total_market_size_factor": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] # Placeholder for overall market growth/shrinkage
    }
    return external_data

def simulate_monthly_demand(persona, product_info, external_data, month_idx, coeffs):
    """
    Simulates the monthly demand for a single persona and product.
    The simulation uses a Poisson distribution based on an expected purchase quantity (lambda),
    which is calculated from persona attributes and external market factors using a log-linear model.
    """
    # Helper to safely convert values to float
    def safe_float(value, default=0.0):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    # Extract persona attributes
    persona_attrs = {
        attr['name']: {
            'value': attr.get('value'),
            'weight': attr.get('weight', 0)
        }
        for attr in persona.get('attributes', [])
    }

    # Base propensity from '제품 적합도'
    product_fit_score = safe_float(persona_attrs.get('제품 적합도', {'value': 5})['value'], 5.0)
    log_propensity = np.log(product_fit_score / 10.0)

    # Price Effect
    price_sensitivity = safe_float(persona_attrs.get('가격 민감도', {'value': 5})['value'], 5.0) / 10.0
    price_effect = (external_data['food_cpi_impact'][month_idx] - 1) * price_sensitivity * coeffs.get('gamma', 1.0)
    log_propensity += price_effect

    # Marketing Effect
    marketing_influence = safe_float(persona_attrs.get('마케팅 영향력', {'value': 5})['value'], 5.0) / 10.0
    ad_effect = 0
    # A simple way to check for ad campaigns from product features
    for model, boost in coeffs.get('delta_ad_map', {}).items():
        if f"광고모델: {model}" in product_info['product_feature']: # Example: specific marketing boost
            ad_effect += boost
    marketing_effect = (external_data['marketing_boost_index'][month_idx] - 1 + ad_effect) * marketing_influence
    log_propensity += marketing_effect

    # Seasonal Effect
    seasonal_pref = safe_float(persona_attrs.get('계절성 구매 성향', {'value': 5})['value'], 5.0) / 10.0
    seasonal_effect = (external_data['seasonal_index'][month_idx] - 1) * seasonal_pref * coeffs.get('beta_season', 1.0)
    log_propensity += seasonal_effect

    # Other Effects
    # Health Consciousness (example for yogurt)
    if "발효유" in product_info['category_level_2'] and "건강 관심도" in persona_attrs:
        health_interest = safe_float(persona_attrs['건강 관심도']['value'], 5.0) / 10.0
        health_effect = (external_data['health_premium_index'][month_idx] - 1) * health_interest
        log_propensity += health_effect

    # Home Cooking (example for seasoning)
    if "조미료" in product_info['category_level_2'] and "가정식 선호도" in persona_attrs:
        home_cooking_pref = safe_float(persona_attrs['가정식 선호도']['value'], 5.0) / 10.0
        home_cooking_effect = (external_data['home_cooking_index'][month_idx] - 1) * home_cooking_pref
        log_propensity += home_cooking_effect

    # Holiday Influence (example for canned ham)
    if "축산캔" in product_info['category_level_1'] and "선물 구매 성향" in persona_attrs:
        gift_tendency = safe_float(persona_attrs['선물 구매 성향']['value'], 5.0) / 10.0
        holiday_effect = (external_data['holiday_index'][month_idx] - 1) * gift_tendency
        log_propensity += holiday_effect

    # Convert log-propensity back to propensity
    adjusted_propensity = np.exp(log_propensity)
    adjusted_propensity = max(0.0, adjusted_propensity) # Ensure non-negative

    # Get the persona's average purchase quantity
    avg_purchase_quantity = int(safe_float(persona_attrs.get('평균 구매 수량', {'value': 1})['value'], 1.0))

    # Calculate lambda (expected purchase quantity) for the Poisson distribution
    lambda_val = adjusted_propensity * avg_purchase_quantity

    # Sample from the Poisson distribution to get the simulated purchase quantity
    return np.random.poisson(lam=lambda_val)


if __name__ == '__main__':
    # 설정 파일 로드
    with open('configs/simulator.yml', 'r', encoding='utf-8') as f:
        sim_config = yaml.safe_load(f)

    # 시뮬레이션 파라미터 설정
    num_simulations_per_persona = sim_config.get('N_trials', 10000)
    market_multiplier = sim_config.get('pop_scale', 1.0)
    num_personas_to_load = sim_config.get('K', 10)
    coeffs = sim_config.get('global_coeffs', {})

    product_df = pd.read_csv('data/product_info.csv')
    submission_df = pd.read_csv('data/sample_submission.csv')
    submission_df = submission_df.set_index('product_name')

    external_market_data = load_external_data() # Load external data once

    for index, product_info in product_df.iterrows():
        product_name = product_info['product_name']
        print(f"Simulating for: {product_name} ({num_simulations_per_persona} sims per persona)")

        # Load all personas for this product
        loaded_personas = load_personas(product_name, num_personas_to_load=num_personas_to_load)
        if not loaded_personas:
            print(f"  -> No personas found for {product_name}. Skipping.")
            continue

        monthly_product_sales = [0] * 12 # Sales for this specific product over 12 months

        for month_idx in range(12): # Iterate through each month
            monthly_purchases_sum_for_product = 0

            # Simulate for each loaded persona
            for persona in loaded_personas:
                for _ in range(num_simulations_per_persona):
                    purchase_quantity = simulate_monthly_demand(persona, product_info, external_market_data, month_idx, coeffs)
                    monthly_purchases_sum_for_product += purchase_quantity
            
            # Scale the sum of simulations by the market multiplier
            scaled_sales = round(monthly_purchases_sum_for_product * market_multiplier)
            monthly_product_sales[month_idx] = scaled_sales

        submission_df.loc[product_name] = monthly_product_sales

    submission_df.reset_index().to_csv('outputs/submission.csv', index=False)

    print("Simulation complete. Submission file saved to outputs/submission.csv")