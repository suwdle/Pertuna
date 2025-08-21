import numpy as np
from .purchase_model import persona_month_mu, sample_negbin

ORDER = ["2024-07","2024-08","2024-09","2024-10","2024-11","2024-12",
         "2025-01","2025-02","2025-03","2025-04","2025-05","2025-06"]
def _idx(m): return ORDER.index(m)

def simulate_product(bag, policy, calendar, coeffs, weights, N, k, months,
                     N_trials=300, stochastic=True, seed=42):
    rng = np.random.default_rng(seed)
    basket = policy.get("basket_size_by_channel", {"대형마트":1.1,"편의점":1.0,"온라인":1.3})
    paths = []
    for _ in range(N_trials):
        path = []
        for m in months:
            price, promo, ads, z = calendar[m]["price"], calendar[m]["promo"], calendar[m]["ad_on"], calendar[m]["vector"]
            mu_sum = 0.0
            for w_i, P in zip(weights, bag.personas):
                mu_i = persona_month_mu(
                    pi0=P.purchase_prob_12[_idx(m)], fi0=P.expected_freq_12[_idx(m)],
                    season=float(P.seasonality[m[-2:]]), promo=promo, ad_dict=ads,
                    price=price, ref_price=policy["ref_price"],
                    beta_season=coeffs["beta_season"], gamma=coeffs["gamma"], delta_ad_map=coeffs["delta_ad_map"], eta=P.price_sensitivity,
                    theta_season=coeffs["theta_season"], lam=coeffs["lambda"], phi=coeffs["phi"],
                    z_vec=z, w_vec=coeffs["w_vec"], u_vec=coeffs["u_vec"],
                    channel_pref=P.channel_pref, basket_map=basket
                )
                mu_sum += w_i * mu_i
            mu = float(N * mu_sum)
            y = sample_negbin(mu, k, rng) if stochastic else mu
            path.append(y)
        paths.append(path)
    arr = np.array(paths)
    return {"months": months,
            "mean":   arr.mean(0).tolist(),
            "median": np.median(arr,0).tolist(),
            "p05":    np.percentile(arr,5,axis=0).tolist(),
            "p95":    np.percentile(arr,95,axis=0).tolist()}
