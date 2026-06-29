
import os, requests, zipfile, io
import pandas as pd
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.progress import track

console = Console()
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════
# UCI Online Retail II
# ═══════════════════════════════════════════════
def download_uci():
    out = RAW_DIR / "online_retail.csv"
    if out.exists():
        console.print(f"[green]✓ UCI already downloaded[/]")
        return str(out)

    console.print("[blue]Downloading UCI Online Retail II...[/]")
    url = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"

    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            # Find the xlsx/csv inside
            names = z.namelist()
            console.print(f"  Files in zip: {names}")
            for name in names:
                if name.endswith(".xlsx") or name.endswith(".csv"):
                    z.extract(name, RAW_DIR)
                    extracted = RAW_DIR / name
                    if name.endswith(".xlsx"):
                        df = pd.read_excel(extracted, sheet_name=0)
                        df.to_csv(out, index=False)
                        extracted.unlink()
                    else:
                        extracted.rename(out)
                    console.print(f"[green]✓ UCI saved to {out}[/]")
                    return str(out)
    except Exception as e:
        console.print(f"[yellow]UCI download failed ({e}). Generating realistic demo data.[/]")

    # Fallback: generate realistic synthetic UCI-style data
    _generate_realistic_uci(out)
    return str(out)


def _generate_realistic_uci(out: Path):
    """Generate realistic UCI-style transaction data if download fails."""
    np.random.seed(42)
    n = 10000
    countries = ["United Kingdom","Germany","France","Netherlands","Australia",
                 "USA","Spain","Belgium","Sweden","Japan"]
    categories = ["Home Decor","Kitchen","Clothing","Electronics","Books",
                  "Toys","Garden","Health","Sports","Office"]

    dates = pd.date_range("2020-01-01", "2023-12-31", periods=n)
    customer_ids = np.random.randint(10000, 18000, n)

    df = pd.DataFrame({
        "InvoiceNo":   [f"INV{i:06d}" for i in range(n)],
        "StockCode":   [f"SC{np.random.randint(1000,9999)}" for _ in range(n)],
        "Description": np.random.choice(categories, n),
        "Quantity":    np.random.randint(1, 50, n),
        "InvoiceDate": dates,
        "UnitPrice":   np.round(np.random.exponential(12, n).clip(0.5, 300), 2),
        "CustomerID":  customer_ids,
        "Country":     np.random.choice(countries, n, p=[0.45,0.1,0.1,0.07,0.07,
                                                          0.06,0.05,0.04,0.03,0.03])
    })
    df.to_csv(out, index=False)
    console.print(f"[green]✓ Generated realistic UCI demo data: {n} transactions, "
                  f"{df['CustomerID'].nunique()} customers, {df['Country'].nunique()} countries[/]")


# ═══════════════════════════════════════════════
# Amazon Reviews — Realistic Sample
# ═══════════════════════════════════════════════
def generate_amazon_sample():
    out = RAW_DIR / "amazon_reviews_sample.csv"
    if out.exists():
        console.print(f"[green]✓ Amazon sample already exists[/]")
        return str(out)

    console.print("[blue]Generating realistic Amazon Reviews sample...[/]")
    np.random.seed(123)
    n = 5000

    categories  = ["Electronics","Books","Home","Clothing","Sports",
                   "Beauty","Toys","Kitchen","Office","Health"]
    sentiments  = {5: "excellent great amazing love perfect",
                   4: "good nice satisfied happy recommend",
                   3: "okay average decent acceptable fine",
                   2: "disappointed poor slow bad mediocre",
                   1: "terrible awful broken waste horrible"}

    user_ids    = [f"USER_{i:06d}" for i in np.random.randint(0, 1500, n)]
    ratings     = np.random.choice([1,2,3,4,5], n, p=[0.05,0.08,0.15,0.32,0.40])
    texts       = [f"This product is {np.random.choice(sentiments[r].split())}. "
                   f"Would {'recommend' if r>=4 else 'not recommend'}." for r in ratings]

    df = pd.DataFrame({
        "user_id":   user_ids,
        "asin":      [f"B{np.random.randint(10000000,99999999)}" for _ in range(n)],
        "rating":    ratings,
        "text":      texts,
        "category":  np.random.choice(categories, n),
        "price":     np.round(np.random.lognormal(3.2, 0.8, n).clip(1, 500), 2),
        "verified":  np.random.choice([True, False], n, p=[0.85, 0.15]),
        "timestamp": pd.date_range("2020-01-01", "2024-01-01", periods=n)
    })

    df.to_csv(out, index=False)
    console.print(f"[green]✓ Amazon sample: {n} reviews, "
                  f"{df['user_id'].nunique()} users, {df['category'].nunique()} categories[/]")
    return str(out)


# ═══════════════════════════════════════════════
# Global Consumer Attitudes (WVS-style)
# ═══════════════════════════════════════════════
def generate_global_consumer_data():
    out = RAW_DIR / "global_consumer_attitudes.csv"
    if out.exists():
        console.print(f"[green]✓ Global consumer data already exists[/]")
        return str(out)

    console.print("[blue]Generating global consumer attitudes dataset...[/]")
    np.random.seed(99)
    n = 2000

    # Realistic distributions by country/region
    country_profiles = {
        "USA":          {"income": 0.70, "price_sens": 0.40, "brand": 0.65, "digital": 0.80},
        "UK":           {"income": 0.65, "price_sens": 0.45, "brand": 0.60, "digital": 0.75},
        "Germany":      {"income": 0.68, "price_sens": 0.50, "brand": 0.55, "digital": 0.72},
        "France":       {"income": 0.62, "price_sens": 0.52, "brand": 0.62, "digital": 0.70},
        "Japan":        {"income": 0.60, "price_sens": 0.55, "brand": 0.70, "digital": 0.78},
        "India":        {"income": 0.35, "price_sens": 0.75, "brand": 0.50, "digital": 0.65},
        "Brazil":       {"income": 0.40, "price_sens": 0.70, "brand": 0.55, "digital": 0.68},
        "Nigeria":      {"income": 0.30, "price_sens": 0.80, "brand": 0.45, "digital": 0.55},
        "Australia":    {"income": 0.72, "price_sens": 0.38, "brand": 0.58, "digital": 0.82},
        "Canada":       {"income": 0.68, "price_sens": 0.42, "brand": 0.60, "digital": 0.79},
    }

    rows = []
    per_country = n // len(country_profiles)
    for country, profile in country_profiles.items():
        for _ in range(per_country):
            age = np.random.choice(["Gen-Z","Millennial","Gen-X","Boomer"],
                                    p=[0.20, 0.35, 0.30, 0.15])
            rows.append({
                "country":          country,
                "age_band":         age,
                "income_proxy":     np.clip(np.random.normal(profile["income"], 0.15), 0, 1),
                "price_sensitivity":np.clip(np.random.normal(profile["price_sens"], 0.12), 0, 1),
                "brand_loyalty":    np.clip(np.random.normal(profile["brand"], 0.15), 0, 1),
                "digital_affinity": np.clip(np.random.normal(profile["digital"], 0.12), 0, 1),
                "life_satisfaction":np.clip(np.random.normal(0.65, 0.18), 0, 1),
                "sustainability_concern": np.clip(np.random.normal(0.55, 0.20), 0, 1),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    console.print(f"[green]✓ Global consumer data: {len(df)} respondents, "
                  f"{df['country'].nunique()} countries[/]")
    return str(out)


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    console.print("\n[bold cyan]═══ Digital Twin Consumer — Data Download ═══[/]\n")

    uci_path    = download_uci()
    amazon_path = generate_amazon_sample()
    wvs_path    = generate_global_consumer_data()

    console.print("\n[bold green]✓ All datasets ready![/]")
    console.print(f"  UCI Retail  : {uci_path}")
    console.print(f"  Amazon      : {amazon_path}")
    console.print(f"  Global WVS  : {wvs_path}")
    console.print("\n[bold]Next step:[/] Run [cyan]python pipeline/01_persona_extraction.py[/]")
