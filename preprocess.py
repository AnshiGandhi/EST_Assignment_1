# preprocess.py

import pandas as pd
import re
import numpy as np

# ---------- CONFIG ----------
PROD_FILE = "crop-area-and-production.xlsx"
IRRIG_FILE = "cropwise-irrigated-area.xlsx"
RAIN_FILE = "actual-and-annual-normal-rainfall.xlsx"
OUTPUT_FILE = "combined.csv"

# crop water requirement constants (m3/ha)
CWR_CONSTANTS = {
    "rice": 12000,
    "wheat": 5000,
    "sugarcane": 18000,
}

# region mapping (subregion -> state)
REGION_MAPPING = {
    "Saurashtra": "Gujarat",
    "Kutch & Diu": "Gujarat",
    "Gujarat Region": "Gujarat",
    "Coastal Andhra Pradesh": "Andhra Pradesh",
    "Rayalseema": "Andhra Pradesh",
    "Bihar Plains": "Bihar",
    "Bihar Plateau": "Bihar",
    "Haryana": "Haryana",
    "Chandigarh & Delhi": "Haryana",
    "Coastal Karnataka": "Karnataka",
    "North Interior Karnataka": "Karnataka",
    "South Interior Karnataka": "Karnataka",
    "MP East": "Madhya Pradesh",
    "MP West": "Madhya Pradesh",
    "Madhya Maharashtra": "Maharashtra",
    "Marathawada": "Maharashtra",
    "Vidarbha": "Maharashtra",
    "Rajasthan East": "Rajasthan",
    "Rajasthan West": "Rajasthan",
    "UP East": "Uttar Pradesh",
    "UP West": "Uttar Pradesh",
    "Gangetic West Bengal": "West Bengal",
    "Sub-Himalayan West Bengal & Sikkim": "West Bengal",
    "Assam and Meghalaya": "Assam and Meghalaya"
}

def clean_year(val):
    """Extract first 4-digit year from a string or number."""
    if pd.isna(val):
        return None
    val = str(val)
    m = re.search(r"\d{4}", val)
    if m:
        return int(m.group(0))
    return None

# ---------- HELPERS ----------
def normalize_state(name: str) -> str:
    if pd.isna(name):
        return name
    name = str(name).strip()
    return REGION_MAPPING.get(name, name)

# ---------- READERS ----------
def read_production(path):
    df = pd.read_excel(path, sheet_name=0, header=4)  # skip metadata rows
    df.columns = [c.strip().upper() for c in df.columns]

    keep = ["ST_CODE", "STATE", "YEAR", "RICE_TA", "RICE_TQ",
            "WHT_TA", "WHT_TQ", "SCAN_TA", "SGUR_TQ"]
    df = df[keep].copy()

    # convert units ('000 ha, '000 t)
    df["RICE_TA"] *= 1000
    df["RICE_TQ"] *= 1000
    df["WHT_TA"] *= 1000
    df["WHT_TQ"] *= 1000
    df["SCAN_TA"] *= 1000
    df["SGUR_TQ"] *= 1000

    # reshape long
    records = []
    for _, row in df.iterrows():
        year = clean_year(row["YEAR"])
        state = normalize_state(row["STATE"])
        records += [
            {"state": state, "year": year, "crop": "rice",
             "crop_area_ha": row["RICE_TA"], "production_tonnes": row["RICE_TQ"]},
            {"state": state, "year": year, "crop": "wheat",
             "crop_area_ha": row["WHT_TA"], "production_tonnes": row["WHT_TQ"]},
            {"state": state, "year": year, "crop": "sugarcane",
             "crop_area_ha": row["SCAN_TA"], "production_tonnes": row["SGUR_TQ"]},
        ]
    return pd.DataFrame(records)

def read_irrigated(path):
    df = pd.read_excel(path, sheet_name=0, header=4)
    df.columns = [c.strip().upper() for c in df.columns]

    keep = ["ST_CODE", "STATENAME", "YEAR",
            "RICE_TAI", "WHT_TAI", "SCAN_TAI"]
    df = df[keep].copy()

    # convert '000 ha → ha
    df["RICE_TAI"] *= 1000
    df["WHT_TAI"] *= 1000
    df["SCAN_TAI"] *= 1000

    # reshape long
    records = []
    for _, row in df.iterrows():
        year = clean_year(row["YEAR"])
        state = normalize_state(row["STATENAME"])
        records += [
            {"state": state, "year": year, "crop": "rice",
             "irrigated_area_ha": row["RICE_TAI"]},
            {"state": state, "year": year, "crop": "wheat",
             "irrigated_area_ha": row["WHT_TAI"]},
            {"state": state, "year": year, "crop": "sugarcane",
             "irrigated_area_ha": row["SCAN_TAI"]},
        ]
    return pd.DataFrame(records)

def read_rainfall(path):
    df = pd.read_excel(path, sheet_name=0, header=0)
    df.columns = [c.strip().upper() for c in df.columns]

    df = df.rename(columns={
        "STATE/REGION": "STATE",
        "ACTUAL RAINFALL": "ACTUAL_RAINFALL_MM",
        "NORMAL RAINFALL": "NORMAL_RAINFALL_MM"
    })

    df = df[["STATE", "YEAR", "ACTUAL_RAINFALL_MM", "NORMAL_RAINFALL_MM"]].copy()

    df["YEAR"] = df["YEAR"].apply(clean_year)
    df["STATE"] = df["STATE"].apply(normalize_state)
    df["ACTUAL_RAINFALL_MM"] = pd.to_numeric(df["ACTUAL_RAINFALL_MM"], errors="coerce")
    df["NORMAL_RAINFALL_MM"] = pd.to_numeric(df["NORMAL_RAINFALL_MM"], errors="coerce")

    return df.rename(columns={"STATE": "state", "YEAR": "year","ACTUAL_RAINFALL_MM" : "actual_rainfall_mm", "NORMAL_RAINFALL_MM" : "normal_rainfall_mm"})

def fix_crop_area(df):
    """Fix NaN/negative/zero crop_area_ha and production_tonnes values."""
    def fix_group(group):
        group = group.sort_values("year").reset_index(drop=True)

        # pre-compute global mean for this state+crop (excluding NaN and <=0)
        mean_area = group.loc[(group["crop_area_ha"].notna()) & (group["crop_area_ha"] > 0), "crop_area_ha"].mean()
        mean_prod = group.loc[(group["production_tonnes"].notna()) & (group["production_tonnes"] > 0), "production_tonnes"].mean()

        for i, row in group.iterrows():
            for col, mean_val in [("crop_area_ha", mean_area), ("production_tonnes", mean_prod)]:
                val = row[col]
                if pd.isna(val) or val <= 0:
                    prev_val, next_val = None, None

                    # look backward
                    if i > 0:
                        pv = group.at[i-1, col]
                        if not pd.isna(pv) and pv > 0:
                            prev_val = pv

                    # look forward
                    if i < len(group)-1:
                        nv = group.at[i+1, col]
                        if not pd.isna(nv) and nv > 0:
                            next_val = nv

                    # replacement logic
                    if prev_val is not None and next_val is not None:
                        new_val = (prev_val + next_val) / 2
                    elif prev_val is not None:
                        new_val = prev_val
                    elif next_val is not None:
                        new_val = next_val
                    else:
                        new_val = mean_val  # fallback global mean

                    group.at[i, col] = new_val
        return group

    df = df.groupby(["state", "crop"], group_keys=False).apply(fix_group)

    # Drop groups where still no valid values exist
    df = df.dropna(subset=["crop_area_ha", "production_tonnes"], how="all")
    df = df[df["crop_area_ha"].notna() & df["production_tonnes"].notna()]  # stricter cleanup

    return df

def fix_irrigated_area(df):
    """Fix NaN/negative/zero irrigated_area_ha values by group mean, drop if still missing."""
    
    def fix_group(group):
        mean_irrig = group.loc[
            (group["irrigated_area_ha"].notna()) & (group["irrigated_area_ha"] > 0),
            "irrigated_area_ha"
        ].mean()

        for i, row in group.iterrows():
            val = row["irrigated_area_ha"]
            if pd.isna(val) or val <= 0:
                group.at[i, "irrigated_area_ha"] = mean_irrig
        return group

    df = df.groupby(["state", "crop"], group_keys=False).apply(fix_group)

    # Drop rows where still missing after replacement
    df = df.dropna(subset=["irrigated_area_ha"])
    
    return df

def fix_rainfall(df):
    """Fix missing rainfall values with state/year averages and fallbacks."""

    # Step 1: build a unique state-year table (avoid crop triplication)
    rainfall_state_year = (
        df.groupby(["state", "year"], as_index=False)
        .agg({"actual_rainfall_mm": "mean"})
    )

    # Step 2: state-level averages (across years)
    state_avg = rainfall_state_year.groupby("state")["actual_rainfall_mm"].mean()

    # Step 3: year-level averages (across states)
    year_avg = rainfall_state_year.groupby("year")["actual_rainfall_mm"].mean()

    # Step 4: global average
    global_avg = rainfall_state_year["actual_rainfall_mm"].mean()

    # Replacement function
    def replace_actual(row):
        val = row["actual_rainfall_mm"]
        if pd.isna(val):
            # try state avg
            state_mean = state_avg.get(row["state"], np.nan)
            if not pd.isna(state_mean):
                return state_mean

            # try year avg
            year_mean = year_avg.get(row["year"], np.nan)
            if not pd.isna(year_mean):
                return year_mean

            # fallback: global avg
            return global_avg
        return val

    # Apply fixes
    df["actual_rainfall_mm"] = df.apply(replace_actual, axis=1)

    # Step 5: fix normal_rainfall_mm (replace NaN, 0, or negative)
    mask = (df["normal_rainfall_mm"].isna()) | (df["normal_rainfall_mm"] <= 0)
    df.loc[mask, "normal_rainfall_mm"] = df.loc[mask, "actual_rainfall_mm"]

    return df

# ---------- BUILD ----------
def build_combined():
    prod = read_production(PROD_FILE)
    irrig = read_irrigated(IRRIG_FILE)
    rain = read_rainfall(RAIN_FILE)

    # merge step by step
    df = prod.merge(irrig, on=["state", "year", "crop"], how="left")
    df = df.merge(rain, on=["state", "year"], how="left")

    df = df[df["state"].str.lower() != "tripura"]

    # --------- FIX NEGATIVE/0 VALUES ----------
    df = fix_crop_area(df)
    df = fix_irrigated_area(df)
    df = fix_rainfall(df)

    # derived
    df["yield_t_ha"] = df["production_tonnes"] / df["crop_area_ha"]
    df["irrigation_fraction"] = df["irrigated_area_ha"] / df["crop_area_ha"]

    # CWR constants
    df["cwr_m3_per_ha"] = df["crop"].map(CWR_CONSTANTS)

    # water demand
    df["water_demand_m3"] = df["crop_area_ha"] * df["cwr_m3_per_ha"]

    # drought flag
    df["drought_flag"] = (
        df["actual_rainfall_mm"] < 0.75 * df["normal_rainfall_mm"]
    ).astype(int)

    df["notes"] = "ok"

    # group by key and average all numeric columns
    df = df.groupby(["state", "year", "crop"], as_index=False).mean(numeric_only=True)

    return df

# ---------- MAIN ----------
if __name__ == "__main__":
    df = build_combined()
    print("Final combined rows:", len(df))
    df.to_csv(OUTPUT_FILE, index=False)
    print("Saved:", OUTPUT_FILE)
