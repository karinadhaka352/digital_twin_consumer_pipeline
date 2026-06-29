# Automated Meta-Analysis Engine for Synthetic Consumer Digital Twins

An end-to-end automated statistical framework designed to ingest, harmonize, and synthesize decentralized multi-study empirical data evaluating consumer choice dynamics. This analytical engine provides the rigorous empirical ground-truth baseline required to calibrate multi-agent Synthetic Consumer Digital Twins.

## 🛠️ Core Engineering Highlights
* **Data Engineering & Ingestion:** Built robust data pipelines using `pandas` and `openpyxl` to extract, clean, and resolve conflicting schemas across 123 independent empirical studies representing over 100,000+ consumer subjects.
* **Statistical Modeling & Calibration:** Programmed core algorithms executing variance-stabilizing Fisher’s $z$ transformations, Cochran’s $Q$ diagnostics, and DerSimonian-Laird Random-Effects pooling to isolate and correct severe cross-study contextual noise ($I^2 > 75\%$).
* **Fidelity & Bias Verification:** Integrated an automated quality-control layer using Egger’s Linear Regression to detect human publication bias ($p < 0.05$) while automating publication-ready summary `matplotlib` forest plots for executive reporting.

## 📂 Project Structure
* `pipeline/` — Data cleaning, harmonization math, and statistical synthesis engines.
* `outputs/` — Generated summary charts and analytical evaluation metrics.
* `.gitignore` — Strictly manages and isolates confidential source data to guarantee absolute local privacy.
