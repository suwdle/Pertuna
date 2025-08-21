import json, yaml, typer
from pathlib import Path
from src.personas.generator import generate_persona_bag
from src.personas.schema import PersonaBag
from src.features.build_calendar import build_calendar
from src.simulation.runner import simulate_product
from src.utils.audit import assert_single_call_per_product

app = typer.Typer()

@app.command()
def generate_personas(config_products: str="configs/products.yml",
                      simulator_cfg: str="configs/simulator.yml"):
    prods = yaml.safe_load(open(config_products,"r",encoding="utf-8"))["products"]
    sim = yaml.safe_load(open(simulator_cfg,"r",encoding="utf-8"))
    for pm in prods:
        generate_persona_bag(pm, K=sim["K"], submit=sim["submit"])
    assert_single_call_per_product(sim["submit"])
    typer.echo("OK: product-level single-turn personas generated.")

@app.command()
def simulate(product_id: str,
             config_products: str="configs/products.yml",
             calendar_cfg: str="configs/calendar.yml",
             simulator_cfg: str="configs/simulator.yml"):
    prods = {p["id"]: p for p in yaml.safe_load(open(config_products,"r",encoding="utf-8"))["products"]}
    sim = yaml.safe_load(open(simulator_cfg,"r",encoding="utf-8")); pm = prods[product_id]

    bag_json = json.loads(Path(f"outputs/personas_json/{product_id}.json").read_text(encoding="utf-8"))
    bag = PersonaBag.model_validate(bag_json)

    calendar, months = build_calendar(calendar_cfg, product_id, pm["ref_price"])
    policy = {"ref_price": pm["ref_price"], "basket_size_by_channel": {"대형마트":1.1,"편의점":1.0,"온라인":1.3}}

    # 페르소나 가중치: 균등 또는 Dirichlet(옵션)
    K = len(bag.personas)
    if sim.get("weights","uniform") == "uniform":
        weights = [1.0/K]*K
    else:
        import numpy as np
        weights = (np.random.default_rng(42).dirichlet([1.0]*K)).tolist()

    coeffs = sim["global_coeffs"]
    out = simulate_product(
        bag=bag, policy=policy, calendar=calendar,
        coeffs={"beta_season": coeffs["beta_season"], "theta_season": coeffs["theta_season"],
                "gamma": coeffs["gamma"], "lambda": coeffs["lambda"], "phi": coeffs["phi"],
                "w_vec": coeffs["w_vec"], "u_vec": coeffs["u_vec"], "delta_ad_map": coeffs["delta_ad_map"]},
        weights=weights, N=sim["pop_scale"], k=sim["dispersion_k"],
        months=months, N_trials=sim["N_trials"], stochastic=sim["stochastic"], seed=42
    )

    out_path = Path(f"outputs/forecasts/{product_id}.json"); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"OK: forecasts → {out_path}")

if __name__ == "__main__":
    app()
