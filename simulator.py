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

def load_external_data(product_info):
    """
    Generates and loads external market data tailored to a specific product.
    This includes category-specific seasonality and event-based marketing boosts.
    """
    # --- 1. Define Base Data ---
    # Dates for the 12-month prediction period (July 2024 - June 2025)
    dates = [datetime.date(2024, 7, 1) + datetime.timedelta(days=30*i) for i in range(12)]
    # More realistic mock Food CPI: slight recovery then stable
    food_cpi_impact = [0.98, 0.97, 0.97, 0.98, 0.99, 1.0, 1.0, 1.0, 0.99, 0.99, 0.98, 0.98]

    # --- 2. Category-Specific Seasonality & Holiday Effects ---
    # Default indices (for products without strong seasonal patterns)
    seasonal_index = np.ones(12)
    holiday_index = np.ones(12)
    
    category_1 = product_info.get('category_level_1', '')
    category_2 = product_info.get('category_level_2', '')

    # (Months: 0-11 correspond to Jul-Jun)
    if category_2 == '커피': # Summer peak for coffee
        seasonal_index = np.array([1.3, 1.4, 1.1, 1.0, 0.9, 0.8, 0.8, 0.9, 1.0, 1.1, 1.1, 1.2])
    elif category_1 in ['참치', '축산캔']: # Holiday peaks for canned goods (Chuseok in Sep, New Year in Jan/Feb)
        holiday_index = np.array([1.0, 1.1, 1.8, 1.1, 1.0, 1.0, 1.4, 1.3, 1.0, 1.0, 1.0, 1.0]) # Chuseok: Sep(2), New Year: Jan(6)/Feb(7)
    elif category_2 == '발효유': # Slight health trend boost in summer/New Year
        seasonal_index = np.array([1.1, 1.1, 1.0, 1.0, 0.9, 0.9, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0])

    # --- 3. Event-Based Marketing Boost ---
    marketing_boost_index = np.ones(12)
    product_features = product_info.get('product_feature', '')
    
    # Find patterns like "X-Y월 광고" (e.g., "6-7월 TV/Youtube/SNS 광고 진행")
    matches = re.findall(r'(\d+)-(\d+)월', product_features)
    for start_month, end_month in matches:
        start_month, end_month = int(start_month), int(end_month)
        # Convert calendar month to simulation month index (0-11 for Jul-Jun)
        for month in range(start_month, end_month + 1):
            sim_month_idx = (month - 7 + 12) % 12
            marketing_boost_index[sim_month_idx] = 1.2 # Apply a 20% boost

    # --- 4. Assemble Data ---
    external_data = {
        "dates": dates,
        "food_cpi_impact": food_cpi_impact,
        "marketing_boost_index": marketing_boost_index,
        "seasonal_index": seasonal_index,
        "holiday_index": holiday_index,
        # Placeholders for other potential factors
        "health_premium_index": np.ones(12),
        "home_cooking_index": np.ones(12),
        "total_market_size_factor": np.ones(12)
    }
    return external_data

def simulate_monthly_demand_probabilistic(personas, product_info, external_data, month_idx, coeffs):
    """
    Simulates monthly demand based on a purchase probability model.
    Each persona makes a probabilistic decision to buy, and if so, buys their average quantity.
    """
    # Helper to safely convert values to float
    def safe_float(value, default=0.0):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    # --- 1. Extract Attributes into Numpy Arrays ---
    all_attr_names = set(attr['name'] for p in personas for attr in p.get('attributes', []))
    attributes_dict = {name: [] for name in all_attr_names}
    for p in personas:
        persona_attrs = {attr['name']: attr.get('value') for attr in p.get('attributes', [])}
        for name in all_attr_names:
            attributes_dict[name].append(safe_float(persona_attrs.get(name)))
    for key, value in attributes_dict.items():
        attributes_dict[key] = np.array(value)

    num_personas = len(personas)

    # --- 2. Calculate Final Purchase Probability for each Persona ---
    
    # Start with the base probability from '제품 적합도'
    base_prob = attributes_dict.get('제품 적합도', np.full(num_personas, 5.0)) / 10.0

    # Calculate modifier effects
    price_sensitivity = attributes_dict.get('가격 민감도', np.full(num_personas, 5.0)) / 10.0
    price_modifier = (external_data['food_cpi_impact'][month_idx] - 1) * price_sensitivity * coeffs.get('gamma', 1.0)

    marketing_influence = attributes_dict.get('마케팅 영향력', np.full(num_personas, 5.0)) / 10.0
    ad_effect = 0
    for model, boost in coeffs.get('delta_ad_map', {}).items():
        if f"광고모델: {model}" in product_info['product_feature']:
            ad_effect += boost
    marketing_modifier = (external_data['marketing_boost_index'][month_idx] - 1 + ad_effect) * marketing_influence

    seasonal_pref = attributes_dict.get('계절성 구매 성향', np.full(num_personas, 5.0)) / 10.0
    seasonal_modifier = (external_data['seasonal_index'][month_idx] - 1) * seasonal_pref * coeffs.get('beta_season', 1.0)

    # Combine base probability with modifiers
    # This model assumes modifiers are additive to a multiplier factor
    final_prob = base_prob * (1 + price_modifier + marketing_modifier + seasonal_modifier)
    
    # Clip probability to be between 0 and 1
    final_prob = np.clip(final_prob, 0, 1)

    # --- 3. Simulate Purchase Event and Quantity ---
    
    # Perform a Bernoulli trial for each persona to see if they purchase
    does_purchase = np.random.binomial(1, final_prob)
    
    # If they purchase, their quantity is '평균 구매 수량', otherwise 0
    avg_purchase_quantity = attributes_dict.get('평균 구매 수량', np.ones(num_personas))
    purchase_quantities = does_purchase * avg_purchase_quantity

    return purchase_quantities


if __name__ == '__main__':
    # --- 1. Load Configs and Data ---
    with open('configs/simulator.yml', 'r', encoding='utf-8') as f:
        sim_config = yaml.safe_load(f)

    num_personas_to_load = sim_config.get('K', 15)
    coeffs = sim_config.get('global_coeffs', {})

    product_df = pd.read_csv('data/product_info.csv')
    submission_df = pd.read_csv('data/sample_submission.csv')
    submission_df = submission_df.set_index('product_name')

    # --- 2. Define Market Size (TAM) based on MARKET_ANALYSIS.md ---
    tam_by_category = {
        '커피': 20.8,
        '발효유': 0.55,
        '참치캔': 10.4,
        '조미료': 3.4,
        '축산캔': 10.4,
        'default': 5.0
    }

    # --- 3. Main Simulation Loop ---
    for index, product_info in product_df.iterrows():
        product_name = product_info['product_name']
        print(f"Simulating for: {product_name}...")

        # --- 3a. Load Personas ---
        loaded_personas = load_personas(product_name, num_personas_to_load=num_personas_to_load)
        if not loaded_personas:
            print(f"  -> No personas found for {product_name}. Skipping.")
            continue

        # --- 3b. Get Product-Specific TAM ---
        product_cat_l2 = product_info.get('category_level_2')
        product_cat_l1 = product_info.get('category_level_1')
        product_tam = tam_by_category.get(product_cat_l2, tam_by_category.get(product_cat_l1, tam_by_category['default']))
        print(f"  -> Using {len(loaded_personas)} personas to represent a TAM of {product_tam}M people.")

        # --- 3c. Run Monthly Simulation with Purchase Probability Model ---
        external_market_data = load_external_data(product_info)
        monthly_product_sales = [0] * 12
        for month_idx in range(12):
            # This function now returns the simulated purchase quantities for the sample
            purchase_quantities_sample = simulate_monthly_demand_probabilistic(loaded_personas, product_info, external_market_data, month_idx, coeffs)
            
            # Total items bought by the K personas in our sample
            total_purchases_in_sample = np.sum(purchase_quantities_sample)

            # Scale this up to the total market size
            num_valid_personas = len(loaded_personas)
            market_segment_size = (product_tam * 1_000_000) / num_valid_personas
            
            # The total sales are the sales from the sample, scaled up by the segment size each persona represents.
            total_monthly_sales = total_purchases_in_sample * market_segment_size

            monthly_product_sales[month_idx] = round(total_monthly_sales)

        submission_df.loc[product_name] = monthly_product_sales

    # --- 4. Save Results ---
    submission_df.reset_index().to_csv('outputs/submission.csv', index=False)
    print("\nSimulation complete. Submission file saved to outputs/submission.csv")
