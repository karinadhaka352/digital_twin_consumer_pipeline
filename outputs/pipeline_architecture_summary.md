# Technical Architecture Summary: Consumer Digital Twin Synthesis Pipeline

## 🔄 1. Multi-Stage Ingestion Pipeline
* **Raw Ingestion Module (`pipeline/05_raw_survey_parser.py`):** Ingests decentralized consumer survey responses, isolates variable construct families, and generates foundational $33 \times 33$ Pearson product-moment correlation matrices ($r$) along with precise baseline sample sizes ($n$).
* **Mathematical Extraction Module (`pipeline/06_extract_meta_rows.py`):** Automates variance-stabilizing Fisher’s $z$ transformations to pool micro-item metrics safely into higher-level macro behavioral constructs without encountering mathematical scale biases.

## 📊 2. Global Synthesis Engine (`pipeline/04_meta_analysis_synthesis.py`)
* Executes inverse-variance weighting under a DerSimonian-Laird Random-Effects model to combine disparate studies into a unified behavioral distribution.
* Calculates exact cross-study heterogeneity indices ($I^2$ and $\tau^2$) to verify simulation boundaries and runs Egger's linear regression intercept audits to flag systemic publication bias.

## 🛡️ 3. Security & Governance Compliance
* Automated exclusion filters via `.gitignore` guarantee that no confidential raw spreadsheets or proprietary source data can leak to public version control platforms, protecting local data integrity completely.
