# Subscription Churn: Risk Ranking and Retention Targeting

A portfolio case study in churn modeling **as a targeting and business-decision problem**, not just a classification exercise: leakage-free evaluation, ranking-quality metrics, cross-validated probability calibration, and a transparent, analytically-derived campaign-economics simulator — built and audited to the standard a Product Analyst / ML Product Analyst / Data Analyst role should be judged on.

## 1. Problem

> **Given customer information available before churn, rank customers by churn risk and decide whom a retention campaign should target under limited resources.**

This is **predictive churn modeling**, not a causal analysis:

- The dataset (public Telco Customer Churn, 7,043 customers, one static snapshot) contains **no randomized retention treatment and no record of any past retention outreach**. It cannot support causal claims about what *causes* churn or what a retention intervention would *do*.
- Everything downstream that involves dollars (Section 7) is **scenario-based expected-value analysis under explicit, labeled assumptions** — never a measured or realized campaign effect.

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
| Train (n=4,225) | Fit both models; CatBoost's final calibration is also fit here (see below) |
| Validation (n=1,409) | CatBoost early stopping (`eval_set`) and hyperparameter/class-weighting selection — used **exactly once**, for that purpose — plus post-hoc model-comparison diagnostics |
| Test (n=1,409) | Opened once, in `03_final_test_evaluation.ipynb`, purely for final reporting and diagnostics |

Test is **never** used as CatBoost's `eval_set`, **never** used to pick a threshold, **never** used for model selection, and **never** used to fit calibration. All preprocessing (`StandardScaler`, `OneHotEncoder`) is fit on train only, inside an sklearn `Pipeline`.

**Calibration does not reuse the validation labels used for model selection.** After hyperparameters (including the early-stopped iteration count) are frozen, CatBoost's probability calibration is fit by 5-fold cross-validation **on train only**: each fold trains a fresh model on the other folds and predicts on its held-out fold, producing out-of-fold probabilities; a sigmoid map is fit on those OOF probabilities vs. true labels; the final model is refit once on all of train. Validation is then used only to *evaluate* (never fit) the calibration, as a diagnostic — see `src/modeling.py::TrainCVCalibratedCatBoost`.

> The original version of this project fit CatBoost with `eval_set=test_pool` and selected its decision threshold by scanning F1 **on the test set** — both are textbook train/test leakage. An intermediate revision fixed the test leakage but calibrated CatBoost on the same validation labels also used for model comparison — a milder double-use of data. Both issues are fixed in this version (see Section 12).

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

  Class weighting was likewise tested, not assumed: weighting the positive class (`class_weights=[1.0, 2.77]`) improved validation PR-AUC marginally (0.6520 → 0.6533) and was kept. Its early-stopped iteration count (303) was then frozen as a fixed hyperparameter — see Section 6.

## 5. Model comparison (final test-set numbers)

Computed once, in `03_final_test_evaluation.ipynb`. Threshold 0.5 here is a **fixed statistical reporting point only** — it is explicitly *not* the business decision (see Section 7). Both models are compared on classification, ranking, **and** probability-quality metrics — not just ROC-AUC/F1 — before recommending either one.

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Brier score | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.843 | 0.634 | 0.504 | 0.781 | 0.613 | 0.1675 | 0.4930 |
| **CatBoost (calibrated)** | **0.845** | **0.671** | 0.650 | 0.567 | 0.606 | **0.1364** | **0.4216** |

**On ROC-AUC alone the two models are close enough to call a tie (0.843 vs. 0.845).** CatBoost's advantage widens on every metric that actually matters for a minority-class targeting problem: PR-AUC (+0.037), Brier score (-0.031), log loss (-0.071), and every Top-K ranking cut below. We are not forcing CatBoost to be the winner here — if Logistic Regression had matched it on PR-AUC and Lift@K, the added complexity of CatBoost (cross-validated calibration, a less directly interpretable score) would not have been worth it, and this project would recommend the simpler model instead. It doesn't: CatBoost wins on the metrics this specific use case is built around.

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

## 6. Probability calibration

Sigmoid (Platt) calibration was fit by **5-fold cross-validation on train only** (Section 3) — never on validation or test — and evaluated as a diagnostic on both:

| | Brier score | Log loss |
|---|---:|---:|
| Raw CatBoost, validation (diagnostic) | 0.1659 | 0.4918 |
| Calibrated CatBoost, validation (diagnostic) | 0.1376 | 0.4264 |
| Raw CatBoost, test | 0.1649 | 0.4867 |
| **Calibrated CatBoost, test** | **0.1364** | **0.4216** |

Class weighting (Section 4) distorts CatBoost's raw probabilities away from the true churn rate — expected, since the loss function was reweighted, so raw outputs are no longer a direct likelihood estimate. Calibration corrects this, and the improvement generalizes to test. The calibrated model is used as the final model regardless of this diagnostic's outcome (calibration is part of the fixed protocol, not a validation-scored choice), but the result is reported honestly either way: it helped, clearly.

## 7. Retention decision layer (scenario-based expected economics — not causal, not realized)

**The threshold is derived analytically from business assumptions, not by scanning validation or test outcomes.** For a customer with predicted churn probability *p*, contacting them has expected scenario value

```
EV(p) = p × success_rate × value_per_retained − cost_per_contact
```

which is positive iff `p > cost_per_contact / (success_rate × value_per_retained) =: p*`. This break-even probability depends **only on the stated assumptions** — never on outcome labels.

**Base-scenario assumptions** (illustrative placeholders — a real deployment substitutes actual CLV/margin figures):
- Cost per contact: **$25**
- Value of a retained customer: **$600**
- Assumed retention success rate among contacted true churners: **30%**

`p* = 25 / (0.30 × 600) ≈ 0.139`

A max-F1 threshold (validation, diagnostic only) lands at 0.390 — a very different, and irrelevant, number: F1 has no connection to the dollar assumptions above, so it is kept only as a modeling diagnostic, never as the business policy. As a further sanity check, a hindsight scan of validation *outcomes* for whatever threshold would have maximized realized net value lands at 0.080 — lower than the analytical 0.139, but in the same broad "target aggressively" region and nowhere near the max-F1 or default-0.5 thresholds, which is a reasonable (not exact) corroboration that the calibrated probabilities are usable inputs to the formula. **That hindsight-scanned number is never itself used as policy**, for the same reason the max-F1 threshold isn't — it is shown only as a diagnostic.

**Applying `p* ≈ 0.139` to test (scenario / expected figures — no campaign was actually run):**

| Metric | Value |
|---|---:|
| Customers contacted | 752 (53.4%) |
| True churners reached | 333 |
| Expected customers retained (30% assumed success rate) | 99.9 |
| Campaign cost | $18,800 |
| Expected value preserved | $59,940 |
| **Expected net value** | **$41,140** |
| Expected ROI | 2.19x |

### Sensitivity analysis

Because the threshold is now a direct function of the three assumptions, its movement is fully explainable rather than an empirical artifact: doubling the cost doubles *p\**; doubling the value or success rate halves it. Sweeping cost ($10/$25/$50/$100), success rate (15%/30%/50%), and customer value ($300/$600/$1,000) on validation:

- **Cheap, high-conviction scenario** (cost $10, success 50%, value $1,000): *p\** = 0.02 — contact nearly the entire base (100% cleared this bar in validation).
- **Expensive, low-conviction scenario** (cost $100, success 15%, value $300): *p\** = 2.22, **which exceeds 1** — no customer's probability can clear this bar, so the rational policy is **"target nobody"** — the campaign is not worth running at all under these assumptions. Two other combinations in the grid (cost $50/success 15%/value $300, and cost $100/success 15%/value $600) also land in "target nobody" territory (*p\** = 1.11).

There is no single "correct" threshold independent of these business inputs — full grid in `reports/tables/economics_sensitivity_grid.csv`.

## 8. Budget-constrained targeting

*"If the retention team can contact only N% of customers, whom should it contact?"* — a separate decision framework from the analytical threshold above, used when outreach **capacity** rather than a per-contact economic bar is the binding constraint. Evaluated on test, model-based Top-K vs. random targeting of the same size:

| Top-K% | Contacted | Churners captured (model) | Churners captured (random, expected) | Lift vs. random |
|---:|---:|---:|---:|---:|
| 5% | 70 | 60 | 18.7 | **3.21x** |
| 10% | 141 | 111 | 37.4 | **2.97x** |
| 20% | 282 | 194 | 74.8 | **2.59x** |

This is the assumption-free headline result: **regardless of exactly how the economics are priced, risk-based targeting captures 2.6x-3.2x as many churners as random outreach of the same size.**

## 9. Segment diagnostics (test set — correlational, not causal)

Every segment finding below is reported with its **absolute churner count and absolute captured count**, under three explicitly-named policies — a recall figure is meaningless, and easy to mis-headline, without both.

| Contract | Churners (n) | Recall @ 0.5 | Recall @ economics threshold (0.139) | Recall @ Top-20% (cutoff 0.536) |
|---|---:|---:|---:|---:|
| Month-to-month | 329 | 64.7% (213/329) | 94.8% (312/329) | 59.0% (194/329) |
| One year | **36** | 0.0% (0/36) | **58.3% (21/36)** | 0.0% (0/36) |
| Two year | **9** | 0.0% (0/9) | 0.0% (0/9) | 0.0% (0/9) |

**The "0% recall" finding from an earlier draft of this project was reported against the wrong policy.** At the arbitrary 0.5 statistical threshold, one- and two-year contract recall is indeed 0%. But at the analytical economics threshold this project actually recommends (~0.139), one-year-contract recall jumps to 58.3% (21 of 36 churners captured) — the model was never structurally blind to these customers; 0.5 was simply too conservative a cutoff to apply to them. Top-20% targeting (cutoff 0.536, more conservative than 0.5) reproduces the same 0% pattern for the same reason.

Two-year contracts remain at 0% recall under every policy tested — but with only **9 actual churners** in the entire test set, this is a **sample-limited, unstable finding**, not a demonstrated structural blind spot: moving 2-3 customers would swing the rate by 20+ points. It is worth monitoring, not treating as settled.

Tenure shows a related but more statistically stable pattern (59-215 churners per band, larger than either contract segment): long-tenured customers (37+ months) are recalled at 13.6% (0.5 threshold) vs. 62.7% (economics threshold) vs. 8.5% (Top-20%) — directionally the same "conservative-threshold-hides-recall" story, with enough churners in the denominator to trust the direction of the effect.

## 10. Explainability and actionability

SHAP (`TreeExplainer`) on the uncalibrated, frozen CatBoost model (`reports/figures/shap_summary.png`) confirms the EDA-driven feature story (`Contract`, `tenure`, `InternetService`, `TotalCharges`, `MonthlyCharges`, `PaymentMethod`, `OnlineSecurity` dominate). SHAP values describe **predictive association strength within this model**, not a causal mechanism. Features are split into:

| Category | Examples | Why |
|---|---|---|
| Actionable levers | `Contract`, `OnlineSecurity`, `TechSupport`, `PaymentMethod` | The business can offer term incentives, bundle add-ons, or nudge payment method |
| Predictive, not directly actionable | `tenure`, `TotalCharges`, `MonthlyCharges` | Strong signal, but not a free lever — changing price has its own revenue cost |
| Non-actionable profile variables | `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Limited/no legitimate intervention path; use `SeniorCitizen` carefully to avoid discriminatory targeting |

**What the model suggests the retention team should investigate:** *why* month-to-month, no-add-on, fiber-optic, electronic-check customers churn more (price? service quality? competitive offers? switching friction?) — SHAP shows association strength, not mechanism. See Section 11 for how to actually test a lever.

## 11. Final recommendation

- **Use CatBoost, with calibration cross-validated on train.** It is close to Logistic Regression on ROC-AUC alone, but pulls clearly ahead on PR-AUC, Brier score/log loss, and every Top-K ranking cut — the metrics that matter for a minority-class targeting problem, and the basis on which CatBoost is preferred here despite its added complexity.
- **Use the analytical break-even threshold, `p* = cost / (success × value)`,** with the business's real assumptions plugged in — never a threshold tuned to any observed outcome, and never max-F1. Where capacity is fixed instead of a per-contact economic bar, a Top-10% to Top-20% risk-ranked list is a robust, assumption-light alternative (2.6x-3.2x lift over random).
- **The recommended threshold is highly sensitive to cost/value/success-rate assumptions** — cheap, high-conviction campaigns should target broadly; expensive or lower-confidence campaigns should target narrowly, or in some plausible scenarios (see Section 7), not run at all. Communicate this sensitivity to stakeholders rather than presenting a single number as fixed truth.
- **Revisit the long-contract segments with the right nuance:** one-year-contract recall is far from zero once the correct (economics) threshold is used; the two-year gap is real in this test set but rests on only 9 churners and should not be treated as a settled structural finding.
- **What cannot be concluded here:** any causal claim that contacting a customer would prevent their churn, or the dataset's "true" retention success rate — there is no randomized treatment in this data, and no dollar figure in Section 7 is a realized or observed value.
- **What to test next:** a real A/B test.

### A/B test design (proposed — no result is reported or implied)

- **Unit of randomization:** individual customer.
- **Eligible population:** Top 20% by CatBoost risk score (concentrates power where the intervention plausibly matters most).
- **Control:** no proactive retention contact.
- **Treatment:** proactive contact + an offer informed by the actionable levers above (contract terms, security/support bundle, payment-method nudge).
- **Primary metric:** churn rate over one billing cycle, treatment vs. control (two-proportion test, report absolute + relative risk reduction with a CI).
- **Guardrails:** cost per contact vs. budget, complaint/opt-out rate, short-term revenue impact of any discount.
- **Major validity risks:** contamination/spillover between treated and control customers; novelty effects from a one-off contact; results generalize only to "high-risk as scored by this model" given risk-based eligibility; the one-/two-year contract segments (36 and 9 churners respectively in this test set) will be badly underpowered and need a separate, deliberately oversampled arm; seasonality.

## 12. What changed from the original version of this project

An audit of the pre-existing notebook found several methodological issues; a first rebuild fixed the most severe ones, and a focused follow-up cleanup (this version) fixed two more subtle ones surfaced in review:

1. **CatBoost was fit with `eval_set=test_pool, use_best_model=True`** — the test set drove early stopping. Fixed: validation is the only `eval_set` used anywhere.
2. **The decision threshold (0.35) was chosen by scanning F1 on the test set** and described in the README as a "business" choice. Fixed: the threshold is now derived **analytically** from stated cost/value/success-rate assumptions (`p* = cost / (success × value)`), with an explicit contrast against (and rejection of) both max-F1 thresholding and hindsight-scanning outcomes for the "best" threshold.
3. **No train/validation split** — only a single train/test split, so every "tuning" decision implicitly used test. Fixed: a proper 60/20/20 stratified split with test opened exactly once.
4. **Calibration was fit on validation** in an intermediate revision, the same labels used for model comparison and diagnostics — a milder double-use of data than test leakage, but still worth removing. Fixed: calibration is now fit by 5-fold cross-validation on **train only** (`src/modeling.py::TrainCVCalibratedCatBoost`); validation is reserved purely for model comparison/diagnostics.
5. **A "0% recall" segment finding was headlined without its denominator or the policy it referred to**, at one point implying a structural blind spot. Fixed: segment tables now report absolute churner/capture counts under three named policies; the finding is now correctly described as "the 0.5 threshold was too conservative for this segment" (one-year contracts) vs. a genuinely small-sample, unstable result (two-year contracts, 9 churners).
6. **No PR-AUC, no ranking metrics (Recall@K/Precision@K/Lift@K), no calibration check, no campaign economics, no segment diagnostics, no SHAP** — all added.
7. **Logistic Regression used raw `pd.get_dummies` with no scaling**, done outside any pipeline. Fixed: `ColumnTransformer` + `Pipeline`.
8. **Reported numbers were not reproducible from a fresh run** (no pinned dependencies, no scripted pipeline). Fixed: pinned `requirements.txt`, `src/` modules, and four notebooks that execute cleanly end-to-end from a fresh kernel.

## Repository structure

```text
subscription-churn-model/
├── src/
│   ├── data.py          # loading, cleaning, the one and only train/val/test split
│   ├── features.py      # column definitions, sklearn preprocessing pipelines
│   ├── evaluation.py     # classification / ranking / calibration / segment metrics
│   ├── modeling.py       # Logistic Regression + CatBoost fitting, train-CV calibration
│   └── economics.py      # analytical break-even threshold, sensitivity analysis
├── notebooks/
│   ├── 01_data_and_eda.ipynb            # audit, EDA, freezes the train/val/test split
│   ├── 02_modeling_and_validation.ipynb # fit + tune on train/val, freeze + calibrate on train
│   ├── 03_final_test_evaluation.ipynb   # test opened once: metrics, ranking, SHAP, segments
│   └── 04_retention_decisioning.ipynb   # analytical economics, sensitivity, A/B design
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
