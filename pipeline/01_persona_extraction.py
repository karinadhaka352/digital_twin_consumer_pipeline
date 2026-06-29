
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from rich.console import Console

console = Console()
DATA_DIR = Path("data")
RAW_DIR  = DATA_DIR / "raw"
OUT_PATH = DATA_DIR / "persona_vectors.csv"


def extract_uci(path="data/raw/online_retail.csv"):
    console.print("[blue]Extracting UCI personas...[/]")
    df = pd.read_csv(path, encoding="ISO-8859-1", low_memory=False)
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(str)
    df["Quantity"]   = pd.to_numeric(df["Quantity"],  errors="coerce")
    df["UnitPrice"]  = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["TotalValue"]  = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    snap = df["InvoiceDate"].max()
    rfm = df.groupby("CustomerID").agg(
        frequency    = ("InvoiceNo",   "nunique"),
        monetary     = ("TotalValue",  "sum"),
        avg_basket   = ("TotalValue",  "mean"),
        n_categories = ("Description", "nunique"),
        country      = ("Country",     lambda x: x.mode()[0])
    ).reset_index()

    sc = MinMaxScaler()
    rfm["price_elasticity"]   = 1 - sc.fit_transform(rfm[["monetary"]])
    rfm["purchase_frequency"] = sc.fit_transform(rfm[["frequency"]])
    rfm["basket_size"]        = sc.fit_transform(rfm[["avg_basket"]])
    rfm["category_diversity"] = sc.fit_transform(rfm[["n_categories"]])
    rfm["brand_loyalty"]      = 0.5   # not in UCI
    rfm["digital_affinity"]   = 0.6
    rfm["income_proxy"]       = rfm["basket_size"]
    rfm["age_band"]           = "Unknown"
    rfm["source"]             = "uci"
    console.print(f"  [green]✓ {len(rfm)} UCI customers[/]")
    return rfm

def extract_amazon(path="data/raw/amazon_reviews_sample.csv"):
    console.print("[blue]Extracting Amazon personas...[/]")
    df = pd.read_csv(path)
    user = df.groupby("user_id").agg(
        avg_rating  = ("rating",   "mean"),
        n_reviews   = ("rating",   "count"),
        n_products  = ("asin",     "nunique"),
        avg_price   = ("price",    "mean") if "price" in df.columns else ("rating","count")
    ).reset_index().dropna()

    sc = MinMaxScaler()
    user["brand_loyalty"]    = 1 - sc.fit_transform(user[["n_products"]])
    user["digital_affinity"] = sc.fit_transform(user[["n_reviews"]])
    user["avg_satisfaction"] = sc.fit_transform(user[["avg_rating"]])
    user["purchase_frequency"] = sc.fit_transform(user[["n_reviews"]])
    user["basket_size"]        = 0.5
    user["price_elasticity"]   = 0.5
    user["category_diversity"] = 0.5
    user["income_proxy"]       = 0.5
    user["age_band"]           = "Unknown"
    user["country"]            = "USA"
    user["CustomerID"]         = user["user_id"]
    user["source"]             = "amazon"
    console.print(f"  [green]✓ {len(user)} Amazon users[/]")
    return user

def extract_global(path="data/raw/global_consumer_attitudes.csv"):
    console.print("[blue]Extracting global consumer personas...[/]")
    df = pd.read_csv(path)
    df["CustomerID"]         = "GBL_" + df.index.astype(str)
    df["purchase_frequency"] = 0.5
    df["basket_size"]        = df["income_proxy"]
    df["avg_satisfaction"]   = df["life_satisfaction"]
    df["category_diversity"] = 0.5
    df["price_elasticity"]   = df["price_sensitivity"]
    df["source"]             = "global_wvs"
    console.print(f"  [green]✓ {len(df)} global respondents, {df['country'].nunique()} countries[/]")
    return df

def merge_all(uci, amazon, gbl):
    COLS = ["CustomerID","country","age_band","source",
            "price_elasticity","purchase_frequency","basket_size",
            "brand_loyalty","digital_affinity","category_diversity",
            "income_proxy","avg_satisfaction"]
    frames = []
    for df in [uci, amazon, gbl]:
        for c in COLS:
            if c not in df.columns:
                df[c] = 0.5
        frames.append(df[COLS])

    out = pd.concat(frames, ignore_index=True)
    for c in COLS[4:]:   # numeric cols
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.5).clip(0, 1)

    out["composite_score"] = (
        out["price_elasticity"]   * 0.20 +
        out["brand_loyalty"]      * 0.25 +
        out["digital_affinity"]   * 0.20 +
        out["purchase_frequency"] * 0.15 +
        out["avg_satisfaction"]   * 0.20
    )
    return out.reset_index(drop=True)

if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    uci    = extract_uci()
    amazon = extract_amazon()
    gbl    = extract_global()
    pv     = merge_all(uci, amazon, gbl)
    pv.to_csv(OUT_PATH, index=False)
    console.print(f"\n[bold green]✓ Persona vectors saved → {OUT_PATH}[/]")
    console.print(f"  Total: {len(pv):,} | Sources: {pv['source'].value_counts().to_dict()}")
    console.print(f"  Countries: {pv['country'].nunique()} unique")
    console.print("\n[bold]Next:[/] python pipeline/02_build_rag_index.py")
