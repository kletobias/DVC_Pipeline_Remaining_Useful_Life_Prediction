[![CI](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/actions/workflows/ci.yaml/badge.svg)](https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc/actions/workflows/ci.yaml)
[![MLflow](https://img.shields.io/badge/MLflow-0194EC?logo=mlflow&logoColor=white)](https://mlflow.org)
[![Optuna](https://img.shields.io/badge/Optuna-4C71F0?logo=optuna&logoColor=white)](https://optuna.org)
[![Hydra](https://img.shields.io/badge/Hydra-1F77B4)](https://github.com/facebookresearch/hydra)
[![DVC](https://img.shields.io/badge/DVC-945DD6?logo=dvc&logoColor=white)](https://dvc.org)

<!-- README.md -->

![abstract_city_data_visualization_neon_blue-2](https://github.com/user-attachments/assets/a064a912-e270-47fb-afb8-2ef861c9ae32)

[![CI — full-stack inference](https://github.com/kletobias/DVC_Pipeline_Remaining_Useful_Life_Prediction/actions/workflows/reproduce_simulate_inference_mlflow.yml/badge.svg)](./.github/workflows/ci.yaml)

# Remaining Useful Life (RUL) — Minimal Proof‑of‑Reproducibility

## ✅ CI Validation: RMSE = 9.13 cycles

```console
$ gh run view --log | grep RMSE
[2025-08-29 13:51:43,832][rul_core_ip.inference][INFO] - RMSE 9.1314
```

**Every push triggers automated validation** — CI reproduces the full inference pipeline and verifies:
- ✓ Model loads correctly from MLflow registry
- ✓ Data transformations match training pipeline  
- ✓ **RMSE < 10 cycles** (fails CI if exceeded)

Small, **two‑stage** DVC pipeline  
1️⃣ `scale_inputs` ➜ 2️⃣ `ridge_predict_log_rul`  
executed head‑less in CI to guarantee byte‑for‑byte reproducibility.

---

## 📈 Performance Metrics

| Metric | Value | Industry Context |
| ------ | ----- | ---------------- |
| **RMSE** | **9.13 cycles** ✅ | Deep learning: 19–30 cycles |
| **Flights/day** | 2 | Cargo freighter standard |
| **Error window** | **< 5 days** | Well within maintenance scheduling |

_Why good?_ This simple Ridge model **halves the error** of deep-learning baselines (RMSE 19–30) on the same C-MAPSS subset while staying interpretable. See refs in comment block.

---

## 🔁 Reproduce locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
dvc pull
dvc repro simulate_inference_mlflow
```

The simulate_inference_mlflow stage fetches the best Ridge v6 run from MLflow and logs every transformation; CI fails if RMSE > 10.

---

🔒 Inference guard‑rail (executed in CI)

```python
rmse = np.sqrt(mean_squared_error(df[target_name], preds_original))
assert rmse <= 10, f"Drift detected: RMSE={rmse:.2f}"
```

---

- Baseline RMSE 18–30 cycles on C‑MAPSS subsets [oai_citation:0‡pmc.ncbi.nlm.nih.gov](https://pmc.ncbi.nlm.nih.gov/articles/PMC7416243/)
- Additional RMSE tables confirming >20 cycles typical [oai_citation:1‡researchgate.net](https://www.researchgate.net/figure/RMSE-of-various-approaches-on-C-MAPSS-dataset_tbl2_363499342)
- CNN‑GAN study reports FD002 RMSE ≈ 19.5 cycles [oai_citation:2‡pmc.ncbi.nlm.nih.gov](https://pmc.ncbi.nlm.nih.gov/articles/PMC7416243/)
- Hybrid CNN‑LSTM baseline RMSE 22–30 cycles [oai_citation:3‡pmc.ncbi.nlm.nih.gov](https://pmc.ncbi.nlm.nih.gov/articles/PMC7416243/?utm_source=chatgpt.com)
- Cargo freighters fly at most two flights per day [oai_citation:4‡thedocs.worldbank.org](https://thedocs.worldbank.org/en/doc/818501436899476698-0190022009/original/AirTransportAirCargoCh4.pdf)
- Predictive‑maintenance scheduling frameworks require multi‑day safety margin [oai_citation:5‡sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S0951832022000175?utm_source=chatgpt.com)
- Data‑driven maintenance scheduling for engines [oai_citation:6‡researchgate.net](https://www.researchgate.net/publication/378947777_Predictive_Maintenance_Scheduling_for_Aircraft_Engines_Based_on_Remaining_Useful_Life_Prediction?utm_source=chatgpt.com)
- Industry adoption of AI predictive maintenance in aviation [oai_citation:7‡airwaysmag.com](https://airwaysmag.com/new-post/ai-powered-predictive-maintenance-revolution?utm_source=chatgpt.com)
- Overview of turbofan RUL research progress [oai_citation:8‡sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0278612524001870?utm_source=chatgpt.com)
- Survey of deep‑learning RUL methods for aero‑engines [oai_citation:9‡nature.com](https://www.nature.com/articles/s41598-022-10191-2?utm_source=chatgpt.com)
