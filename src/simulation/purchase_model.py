import numpy as np

def _logit(p): return np.log((p+1e-12)/(1-p+1e-12))
def _inv(x):   return 1/(1+np.exp(-x))

def persona_month_mu(pi0, fi0, season, promo, ad_dict, price, ref_price,
                     beta_season, gamma, delta_ad_map, eta,
                     theta_season, lam, phi, z_vec, w_vec, u_vec,
                     channel_pref, basket_map):
    # 확률 보정
    x = _logit(pi0) + beta_season*np.log(season+1e-12) + gamma*promo \
        - eta*((price/ref_price)-1.0) + float(np.dot(w_vec, z_vec))
    for m,on in ad_dict.items(): x += delta_ad_map.get(m, 0.0) * on
    p = _inv(x)

    # 빈도 보정
    ln_f = np.log(max(fi0, 1e-6)) + theta_season*np.log(season+1e-12) \
         + lam*promo - phi*((price/ref_price)-1.0) + float(np.dot(u_vec, z_vec))
    f = np.exp(ln_f)

    # 채널 장바구니
    b = sum(channel_pref.get(c,0.0)*basket_map.get(c,1.0) for c in basket_map.keys())
    return float(p * f * b)

def sample_negbin(mean, k, rng):
    p = k/(k+mean+1e-12); r = k
    return float(rng.negative_binomial(r, p))
