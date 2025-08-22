import pandas as pd
import yaml
import os
import re
import numpy as np
import datetime

def sanitize_filename(name):
    """Sanitizes a string to be used as a filename."""
    return re.sub(r'[^a-zA-Z0-9가-힣\s-]', '', name).replace(' ', '_')

def load_personas(product_name):
    """
    Loads persona data for a given product.
    It looks for a file named after the product in the personas directory.
    If not found, it returns a dummy persona.
    """
    safe_name = sanitize_filename(product_name)
    persona_file = f"outputs/personas/{safe_name}_persona.yml"

    if not os.path.exists(persona_file):
        # If a specific persona file doesn't exist, return a dummy persona
        # This dummy persona now matches the expected structure
        return {
            'name': 'dummy_persona',
            'attributes': [{'name': attr, 'value': 'dummy', 'weight': 5} for attr in ['나이', '성별', '건강 관심도', '가격 민감도', '마케팅 영향력', '가정식 선호도', '계절성 구매 성향', '선물 구매 성향']]
        }

    with open(persona_file, 'r', encoding='utf-8') as f:
        persona_data = yaml.safe_load(f)
    
    # The LLM now generates a single persona dictionary, not a list of attributes
    # So, we return the loaded persona_data directly
    return persona_data

def load_external_data():
    """
    Loads or generates mock external market data based on report.md insights.
    For now, this is mock data. In a real scenario, this would load from CSVs.
    """
    # Dates for the 12-month prediction period (July 2024 - June 2025)
    dates = [datetime.date(2024, 7, 1) + datetime.timedelta(days=30*i) for i in range(12)]

    # Mock data for indices (simplified for demonstration)
    # These values would ideally come from actual data or more complex models
    external_data = {
        "dates": dates,
        "food_cpi_impact": [1.0, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.89, 0.88], # Higher CPI means lower impact (less purchasing power)
        "marketing_boost_index": [1.0, 1.05, 1.1, 1.05, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], # Example: boost in first few months
        "health_premium_index": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], # Placeholder, could vary by product category
        "home_cooking_index": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], # Placeholder
        "seasonal_index": [1.2, 1.3, 1.1, 1.0, 0.9, 0.8, 0.9, 1.0, 1.1, 1.2, 1.1, 1.0], # Example: higher in summer/winter
        "holiday_index": [1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.3], # Example: boost in Nov (Chuseok) and Dec (year-end)
        "total_market_size_factor": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] # Placeholder for overall market growth/shrinkage
    }
    return external_data

def simulate_monthly_demand(persona, product_info, external_data, month_idx):
    """
    Simulates the monthly demand for a single persona and product,
    incorporating persona attributes and external market factors.
    """
    # Extract persona attributes
    persona_attrs = {attr['name']: {'value': attr['value'], 'weight': attr['weight']}
                     for attr in persona['attributes']}

    # Base purchase probability (can be refined)
    base_purchase_score = sum(attr['weight'] for attr in persona['attributes'])
    base_purchase_probability = min(base_purchase_score / 120.0, 1.0) # Max score 10*12 = 120

    # Apply external factors and persona sensitivities
    adjusted_probability = base_purchase_probability

    # Price Sensitivity
    price_sensitivity = persona_attrs.get('가격 민감도', {'value': 5, 'weight': 5})['value'] / 10.0 # Scale 0-1
    adjusted_probability *= (1 + (external_data['food_cpi_impact'][month_idx] - 1) * price_sensitivity)

    # Marketing Influence
    marketing_influence = persona_attrs.get('마케팅 영향력', {'value': 5, 'weight': 5})['value'] / 10.0
    if "광고모델: 안유진" in product_info['product_feature']: # Example: specific marketing boost
        adjusted_probability *= (1 + (external_data['marketing_boost_index'][month_idx] - 1) * marketing_influence)

    # Health Consciousness (example for yogurt)
    if "발효유" in product_info['category_level_2'] and "건강 관심도" in persona_attrs:
        health_interest = persona_attrs['건강 관심도']['value'] / 10.0
        adjusted_probability *= (1 + (external_data['health_premium_index'][month_idx] - 1) * health_interest)

    # Home Cooking (example for seasoning)
    if "조미료" in product_info['category_level_2'] and "가정식 선호도" in persona_attrs:
        home_cooking_pref = persona_attrs['가정식 선호도']['value'] / 10.0
        adjusted_probability *= (1 + (external_data['home_cooking_index'][month_idx] - 1) * home_cooking_pref)

    # Seasonal Preference (example for cup coffee)
    if "커피-CUP" in product_info['category_level_3'] and "계절성 구매 성향" in persona_attrs:
        seasonal_pref = persona_attrs['계절성 구매 성향']['value'] / 10.0
        adjusted_probability *= (1 + (external_data['seasonal_index'][month_idx] - 1) * seasonal_pref)

    # Holiday Influence (example for canned ham)
    if "축산캔" in product_info['category_level_1'] and "선물 구매 성향" in persona_attrs:
        gift_tendency = persona_attrs['선물 구매 성향']['value'] / 10.0
        adjusted_probability *= (1 + (external_data['holiday_index'][month_idx] - 1) * gift_tendency)

    # Ensure probability is within [0, 1]
    adjusted_probability = max(0.0, min(1.0, adjusted_probability))

    # Determine if a purchase occurs in this month
    # This is a simplified binary purchase. Could be extended to quantity.
    if np.random.rand() < adjusted_probability:
        return 1
    else:
        return 0


if __name__ == '__main__':
    product_df = pd.read_csv('data/product_info.csv')
    submission_df = pd.read_csv('data/sample_submission.csv')
    submission_df = submission_df.set_index('product_name')

    external_market_data = load_external_data() # Load external data once

    total_market_size_base = 10000 # Base market size
    num_personas_per_product = 100 # Number of simulations per product

    for index, product_info in product_df.iterrows():
        product_name = product_info['product_name']
        print(f"Simulating for: {product_name}")
        
        persona_template = load_personas(product_name)
        if not persona_template: # Check if persona_template is empty (e.g., if dummy persona is empty)
            print(f"  -> No valid persona found for {product_name}. Skipping.")
            continue

        monthly_product_sales = [0] * 12 # Sales for this specific product over 12 months

        for month_idx in range(12): # Iterate through each month
            # Apply total market size factor for the month
            current_market_size = total_market_size_base * external_market_data['total_market_size_factor'][month_idx]
            
            monthly_purchases_sum_for_product = 0
            for _ in range(num_personas_per_product): # Simulate for each persona instance
                # In a more advanced setup, you'd sample from a pool of diverse personas
                # For now, we use the template and simulate purchase for this month
                purchase_occurred = simulate_monthly_demand(persona_template, product_info, external_market_data, month_idx)
                monthly_purchases_sum_for_product += purchase_occurred
            
            # Scale up the simulated purchases to the current market size
            # This is a very simplified scaling. A more complex model would distribute purchases.
            scaled_sales = round(monthly_purchases_sum_for_product * (current_market_size / num_personas_per_product))
            monthly_product_sales[month_idx] = scaled_sales

        submission_df.loc[product_name] = monthly_product_sales

    submission_df.reset_index().to_csv('outputs/submission.csv', index=False)

    print("Simulation complete. Submission file saved to outputs/submission.csv")
