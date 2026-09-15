# Subscription Churn: Risk Ranking and Retention Targeting

A portfolio case study in churn modeling **as a targeting and business-decision problem**, not just a classification exercise: leakage-free evaluation, ranking-quality metrics, probability calibration, and a transparent, assumption-driven campaign-economics simulator — built and audited to the standard a Product Analyst / ML Product Analyst / Data Analyst role should be judged on.

## 1. Problem

> **Given customer information available before churn, rank customers by churn risk and decide whom a retention campaign should target under limited resources.**

This is **predictive churn modeling**, not a causal analysis:

- The dataset (public Telco Customer Churn, 7,043 customers, one static snapshot) contains **no randomized retention treatment and no record of any past retention outreach**. It cannot support causal claims about what *causes* churn or what a retention intervention would *do*.
- Everything downstream that involves dollars (Section 6) is **scenario-based decision analysis under explicit, labeled assumptions** — not a measured campaign effect.

## 2. Data audit and EDA

Full audit in [`notebooks/01_data_and_eda.ipynb`](notebooks/01_data_and_eda.ipynb). Headline findings:

- **Size / balance:** 7,043 customers, 20 candidate features, churn rate **26.5%** — imbalanced, so accuracy is a weak signal throughout this project.
- **Missing values:** `TotalCharges` is stored as text and contains 11 blank values that `isna()` on the raw column does not catch (blank strings aren't `NaN`). After numeric coercion, all 11 blanks correspond to customers with `tenure == 0` (not yet billed) — encoded as `TotalCharges = 0`, with the `tenure == 0` precondition asserted in code so a future data refresh that breaks the assumption fails loudly.
- **Duplicates / leakage:** No duplicate rows, no duplicate `customerID`. `customerID` is a bare identifier and is dropped; no other column encodes post-outcome information, so no additional leakage columns were found.
- **Churn correlates:** short tenure, month-to-month contracts, fiber-optic internet, electronic check payment, and absence of `OnlineSecurity`/`TechSupport` add-ons are all associated with materially higher churn (see notebook for exact rates by segment).

## 3. Evaluation protocol (leakage-free)

A **stratified 60/20/20 train / validation / test split**, fixed seed (`random_state=42`), computed **exactly once** in notebook 01 and persisted to `data/processed/{train,val,test}.csv`. Every other notebook reads those files — no re-splitting anywhere.

| Split | Role |
|---|---|
| Train (n=4,225) | Fit both models |
| Validation (n=1,409) | CatBoost early stopping (`eval_set`), hyperparameter selection, class-weighting comparisons, probability calibration, threshold/economics decisions |
| Test (n=1,409) | Opened once, in `03_final_test_evaluation.ipynb`, purely for final reporting and diagnostics |

Test is **never** used as CatBoost's `eval_set`, **never** used to pick a threshold, **never** used for model selection, and **never** used to fit calibration. All preprocessing (`StandardScaler`, `OneHotEncoder`) is fit on train only, inside an sklearn `Pipeline`.

> The original version of this project fit CatBoost with `eval_set=test_pool` and selected its decision threshold by scanning F1 **on the test set** — both are textbook train/test leakage. Both are fixed in this rebuild (see Section 12, "what changed").

## 4. Models

Two models, both built for interpretability of *how* they're constructed, not just their output:

- **Logistic Regression** — `ColumnTransformer` (StandardScaler + OneHotEncoder) inside a single sklearn `Pipeline`. `class_weight` (`None` vs. `"balanced"`) was compared on validation PR-AUC rather than assumed; `"balanced"` won narrowly (0.6427 vs. 0.6422) and was kept.
- **CatBoost** — native categorical handling (no manual encoding), trained on train with validation as the early-stopping `eval_set` (`eval_metric="PRAUC"`). A small `depth` x `learning_rate` grid (4 combinations) was evaluated by validation PR-AUC:

  | depth | learning_rate | best_iteration | val PR-AUC |
  |---:|---:|---:|---:|
  | **4** | **0.05** | **192** | **0.6520** |
  | 6 | 0.1 | 78 | 0.6504 |
  | 8 | 0.1 | 31 | 0.6450 |
  | 6 | 0.05 | 73 | 0.6445 |

  Class weighting was likewise tested, not assumed: weighting the positive class (`class_weights=[1.0, 2.77]`) improved validation PR-AUC marginally (0.6520 → 0.6533) and was kept.

## 5. Model comparison (final test-set numbers)

Computed once, in `03_final_test_evaluation.ipynb`. Threshold 0.5 here is a **fixed statistical reporting point only** — it is explicitly *not* the business decision (see Section 6).

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.843 | 0.634 | 0.504 | 0.781 | 0.613 | 748 | 287 | 82 | 292 |
| **CatBoost (calibrated)** | **0.845** | **0.671** | 0.650 | 0.567 | 0.606 | 921 | 114 | 162 | 212 |

CatBoost is the model of record: it matches Logistic Regression on ROC-AUC and clearly exceeds it on **PR-AUC** — the primary model-selection signal here, since churners are the minority class and this is fundamentally a ranking problem.

### Ranking quality (Recall / Precision / Lift @ Top-K%, test set)

| Model | Top-K% | Contacted | Churners captured | Precision@K | Recall@K | Lift@K |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 5% | 70 | 53 | 0.757 | 0.142 | 2.85x |
| Logistic Regression | 10% | 141 | 107 | 0.759 | 0.286 | 2.86x |
| Logistic Regression | 20% | 282 | 187 | 0.663 | 0.500 | 2.50x |
| **CatBoost** | **5%** | 70 | 60 | **0.857** | 0.160 | **3.23x** |
| **CatBoost** | **10%** | 141 | 111 | **0.787** | 0.297 | **2.97x** |
| **CatBoost** | **20%** | 282 | 194 | **0.688** | 0.519 | **2.59x** |

CatBoost out-ranks Logistic Regression at every Top-K cut tested. Contacting the top 10% of customers by CatBoost score captures ~30% of all churners in the test set at ~3x the hit rate of a same-sized random sample — this result holds on held-out test data, not just validation.

## 6. Calibration

Sigmoid (Platt) calibration was **fit on validation only**, compared against the uncalibrated model on validation Brier score, and adopted because it won:

| | Brier score | Log loss |
|---|---:|---:|
| Raw CatBoost (validation) | 0.1659 | 0.4918 |
| Calibrated CatBoost (validation) | 0.1376 | 0.4264 |
| Raw CatBoost (test) | 0.1649 | 0.4867 |
| **Calibrated CatBoost (test)** | **0.1364** | **0.4215** |

The improvement generalizes to test, which matters directly for Section 6's economics: expected-value calculations multiply predicted probabilities by dollar amounts, so calibrated probabilities are a precondition for those numbers to mean anything even as *scenario* estimates.

## 7. Retention decision layer (campaign economics — scenario analysis, not a causal claim)

**The threshold is not chosen by maximum F1.** It is chosen by maximizing *expected net campaign value* under explicit, labeled assumptions, selected on **validation** and only then applied to test for reporting.

**Base-scenario assumptions** (illustrative placeholders — a real deployment substitutes actual CLV/margin figures):
- Cost per contact: **$25**
- Value of a retained customer: **$600**
- Assumed retention success rate among contacted true churners: **30%**

| | Max-F1 threshold (validation) | Economics-optimal threshold (validation) |
|---|---:|---:|
| Threshold | 0.40 | **0.11** |
| F1 | 0.641 | — |

These land in very different places — a concrete illustration of why "optimize F1, call it the business threshold" is the wrong move: at 0.11, the model recommends contacting ~56% of customers, because contact cost is cheap relative to the assumed value of a save.

**Applying the validation-chosen threshold (0.11) to test:**

| Metric | Value |
|---|---:|
| Customers contacted | 795 (56.4%) |
| True churners reached | 341 |
| Expected customers retained (30% success rate) | 102.3 |
| Campaign cost | $19,875 |
| Expected value preserved | $61,380 |
| **Expected net value** | **$41,505** |
| ROI | 2.09x |

### Sensitivity analysis

Sweeping cost ($10/$25/$50/$100), success rate (15%/30%/50%), and customer value ($300/$600/$1,000) on validation shows the recommended policy is **highly sensitive to these assumptions**:

- **Cheap, high-conviction scenario** (cost $10, success 50%, value $1,000): optimal threshold → 0.0, i.e. contact everyone.
- **Expensive, low-conviction scenario** (cost $100 or $50, success 15%, value $300): optimal threshold rises so high that **zero customers qualify** — the campaign is not worth running at all under those assumptions, a conclusion the economics model surfaces that a purely statistical threshold never would.
- With a **20%-of-base capacity constraint** (a call-center/budget cap), the constraint binds at ~19.4% of customers in every scenario tested where the campaign is worth running at all — the decision question shifts from "what threshold" to "which 20%," i.e. the Top-K framing below.

Full grids: `reports/tables/economics_sensitivity_grid.csv` and `..._capped20.csv`.

## 8. Budget-constrained targeting

*"If the retention team can contact only N% of customers, whom should it contact?"* — evaluated on test, model-based Top-K vs. random targeting of the same size:

| Top-K% | Contacted | Churners captured (model) | Churners captured (random, expected) | Lift vs. random |
|---:|---:|---:|---:|---:|
| 5% | 70 | 60 | 18.7 | **3.21x** |
| 10% | 141 | 111 | 37.4 | **2.97x** |
| 20% | 282 | 194 | 74.8 | **2.59x** |

This is the assumption-free headline result: **regardless of exactly how the economics are priced, risk-based targeting captures 2.6x-3.2x as many churners as random outreach of the same size.**

## 9. Segment diagnostics (test set — correlational, not causal)

| Segment | n | Churn rate | Model recall (@0.5) |
|---|---:|---:|---:|
| Contract: Month-to-month | 773 | 42.6% | 64.4% |
| Contract: One year | 300 | 12.0% | **0.0%** |
| Contract: Two year | 336 | 2.7% | **0.0%** |
| Tenure: 0–12 months | 449 | 47.9% | 71.6% |
| Tenure: 13–36 months | 369 | 27.1% | 50.0% |
| Tenure: 37+ months | 591 | 10.0% | 13.6% |

**Honestly-reported weak spot:** the model's recall among one- and two-year contract customers is **0% at the standard reporting threshold** — contract type dominates the score so strongly that long-contract churners (36 and 9 customers in this test set, respectively) essentially never cross 0.5, even though they do churn. Long-tenured customers show the same pattern at a smaller scale. This is a genuine model limitation worth flagging to stakeholders, not a data-sparsity artifact to wave away.

## 10. Explainability and actionability

SHAP (`TreeExplainer`) on the uncalibrated CatBoost model (`reports/figures/shap_summary.png`) confirms the EDA-driven feature story (`Contract`, `tenure`, `InternetService`, `TotalCharges`, `MonthlyCharges`, `PaymentMethod`, `OnlineSecurity` dominate). Features are split into:

| Category | Examples | Why |
|---|---|---|
| Actionable levers | `Contract`, `OnlineSecurity`, `TechSupport`, `PaymentMethod` | The business can offer term incentives, bundle add-ons, or nudge payment method |
| Predictive, not directly actionable | `tenure`, `TotalCharges`, `MonthlyCharges` | Strong signal, but not a free lever — changing price has its own revenue cost |
| Non-actionable profile variables | `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Limited/no legitimate intervention path; use `SeniorCitizen` carefully to avoid discriminatory targeting |

**What the model suggests the retention team should investigate:** *why* month-to-month, no-add-on, fiber-optic, electronic-check customers churn more (price? service quality? competitive offers? switching friction?) — SHAP shows association strength, not mechanism. See Section 11 for how to actually test a lever.

## 11. Final recommendation

- **Use CatBoost, calibrated on validation.** It wins on PR-AUC and every Top-K ranking cut, and calibration meaningfully improves the Brier score on held-out test data.
- **Don't hardcode a single threshold.** Feed the business's actual cost/value/success-rate assumptions into the sensitivity model (Section 7); where capacity is fixed, a Top-10% to Top-20% risk-ranked list is a robust, assumption-light starting point (2.6x-3.2x lift over random).
- **Investigate the long-contract blind spot** (Section 9) before relying on this model for that segment.
- **What cannot be concluded here:** any causal claim that contacting a customer would prevent their churn, or the dataset's "true" retention success rate — there is no randomized treatment in this data.
- **What to test next:** a real A/B test.

### A/B test design (proposed — no result is reported or implied)

- **Unit of randomization:** individual customer.
- **Eligible population:** Top 20% by CatBoost risk score (concentrates power where the intervention plausibly matters most).
- **Control:** no proactive retention contact.
- **Treatment:** proactive contact + an offer informed by the actionable levers above (contract terms, security/support bundle, payment-method nudge).
- **Primary metric:** churn rate over one billing cycle, treatment vs. control (two-proportion test, report absolute + relative risk reduction with a CI).
- **Guardrails:** cost per contact vs. budget, complaint/opt-out rate, short-term revenue impact of any discount.
- **Major validity risks:** contamination/spillover between treated and control customers; novelty effects from a one-off contact; results generalize only to "high-risk as scored by this model" given risk-based eligibility; long-contract segments will be underpowered given the model's low recall there (Section 9) and need a separate, deliberately oversampled arm; seasonality.

## 12. What changed from the original version of this project

An audit of the pre-existing notebook found several methodological issues, all fixed in this rebuild:

1. **CatBoost was fit with `eval_set=test_pool, use_best_model=True`** — the test set drove early stopping. Fixed: validation is the only `eval_set` used anywhere.
2. **The decision threshold (0.35) was chosen by scanning F1 on the test set** and described in the README as a "business" choice. Fixed: the threshold is chosen on validation, by maximizing *expected economic value* under explicit assumptions (Section 7), with an explicit contrast against (and rejection of) max-F1 thresholding.
3. **No train/validation split** — only a single train/test split, so every "tuning" decision (including the threshold) implicitly used test. Fixed: a proper 60/20/20 stratified split with test opened exactly once.
4. **No PR-AUC, no ranking metrics (Recall@K/Precision@K/Lift@K), no calibration check, no campaign economics, no segment diagnostics, no SHAP** — all added here.
5. **Logistic Regression used raw `pd.get_dummies` with no scaling**, done outside any pipeline (risk of train/test column mismatch and no leakage protection for future refits). Fixed: `ColumnTransformer` + `Pipeline`.
6. **Reported numbers were not reproducible from a fresh run** (no pinned dependencies, no scripted pipeline). Fixed: pinned `requirements.txt`, `src/` modules, and four notebooks that execute cleanly end-to-end from a fresh kernel.

## Repository structure

```text
subscription-churn-model/
├── src/
│   ├── data.py          # loading, cleaning, the one and only train/val/test split
│   ├── features.py      # column definitions, sklearn preprocessing pipelines
│   ├── evaluation.py     # classification / ranking / calibration / segment metrics
│   ├── modeling.py       # Logistic Regression + CatBoost fitting, calibration
│   └── economics.py      # campaign-economics simulator, sensitivity analysis
├── notebooks/
│   ├── 01_data_and_eda.ipynb            # audit, EDA, freezes the train/val/test split
│   ├── 02_modeling_and_validation.ipynb # fit + tune on train/val only, calibrate
│   ├── 03_final_test_evaluation.ipynb   # test opened once: metrics, ranking, SHAP, segments
│   └── 04_retention_decisioning.ipynb   # campaign economics, sensitivity, A/B design
├── reports/
│   ├── figures/          # generated PNGs referenced above
│   └── tables/           # generated CSVs backing every number in this README
├── data/                 # gitignored; populated by running notebook 01
├── models/               # gitignored; populated by running notebook 02
├── requirements.txt      # pinned to the versions used for this run
└── README.md
```

## Reproduction

```bash
git clone https://github.com/avgrosheva/subscription-churn-model.git
cd subscription-churn-model
python -m venv .venv && source .venv/Scripts/activate  # or .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

Download the dataset (public Telco Customer Churn) to `data/telco_churn.csv` — e.g. from Kaggle ("Telco Customer Churn") or the IBM sample-data mirror — then run the notebooks **in order**, from a fresh kernel:

```text
notebooks/01_data_and_eda.ipynb            # writes data/processed/{train,val,test}.csv
notebooks/02_modeling_and_validation.ipynb # writes models/*.pkl
notebooks/03_final_test_evaluation.ipynb
notebooks/04_retention_decisioning.ipynb
```

Each notebook reads only the artifacts written by the notebooks before it — there is no hidden state and no notebook re-derives the train/val/test split.
