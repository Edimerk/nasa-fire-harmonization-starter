"""
Módulo de cálculo analítico y validaciones para el Sprint 4.
NASA Space Apps Challenge 2026 - Harmonization of MODIS and VIIRS Hot Spots.

Proporciona funciones puras, deterministas y testeables para:
- Validación de coordenadas y rangos temporales.
- Filtrado espacial y temporal de detecciones armonizadas.
- Construcción de matriz de calendario (año x mes).
- Cálculo de mediana histórica mensual con imputación de ceros.
- Cálculo y categorización de anomalías de actividad de fuego.
- Sanitización de datos para exportación segura a CSV.
"""

from typing import Tuple, List, Optional, Union
import datetime
import numpy as np
import pandas as pd

MESES_ES = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"
}

NOMBRES_MESES = [MESES_ES[m] for m in range(1, 13)]

ANOMALY_THRESHOLD_HIGH = 1.5
ANOMALY_THRESHOLD_LOW = 0.5
MIN_YEARS_FOR_MEDIAN = 5


def validate_coordinates(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float
) -> Tuple[bool, str]:
    """
    Valida que las coordenadas geográficas estén dentro de rangos válidos WGS84
    y que los mínimos sean estrictamente menores que los máximos.
    """
    try:
        lat_min = float(lat_min)
        lat_max = float(lat_max)
        lon_min = float(lon_min)
        lon_max = float(lon_max)
    except (ValueError, TypeError):
        return False, "Las coordenadas deben ser valores numéricos válidos."

    if not (-90.0 <= lat_min <= 90.0 and -90.0 <= lat_max <= 90.0):
        return False, "La latitud debe estar comprendida entre -90.0° y 90.0°."

    if not (-180.0 <= lon_min <= 180.0 and -180.0 <= lon_max <= 180.0):
        return False, "La longitud debe estar comprendida entre -180.0° y 180.0°."

    if lat_min >= lat_max:
        return False, (
            f"La latitud mínima ({lat_min:.2f}°) debe ser menor que la "
            f"latitud máxima ({lat_max:.2f}°)."
        )

    if lon_min >= lon_max:
        return False, (
            f"La longitud mínima ({lon_min:.2f}°) debe ser menor que la "
            f"longitud máxima ({lon_max:.2f}°)."
        )

    return True, ""


def validate_dates(
    fi: Union[datetime.date, datetime.datetime, str],
    ff: Union[datetime.date, datetime.datetime, str],
    min_date: Optional[Union[datetime.date, datetime.datetime, str]] = None,
    max_date: Optional[Union[datetime.date, datetime.datetime, str]] = None
) -> Tuple[bool, str]:
    """
    Valida que el rango de fechas sea coherente (fi <= ff) y esté
    dentro del periodo disponible en el conjunto de datos.
    """
    try:
        fi_dt = pd.to_datetime(fi).date()
        ff_dt = pd.to_datetime(ff).date()
    except Exception:
        return False, "Formato de fecha inválido."

    if fi_dt > ff_dt:
        return False, (
            f"Rango de fechas invertido: La fecha 'Desde' ({fi_dt}) no puede ser "
            f"posterior a la fecha 'Hasta' ({ff_dt})."
        )

    if min_date is not None:
        min_dt = pd.to_datetime(min_date).date()
        if ff_dt < min_dt:
            return False, (
                f"El periodo seleccionado finaliza antes del inicio de datos disponibles ({min_dt})."
            )

    if max_date is not None:
        max_dt = pd.to_datetime(max_date).date()
        if fi_dt > max_dt:
            return False, (
                f"El periodo seleccionado inicia después del final de datos disponibles ({max_dt})."
            )

    return True, ""


def filter_bbox(df: pd.DataFrame, bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
    """
    Filtra detecciones por Bounding Box (w, s, e, n) -> (lon_min, lat_min, lon_max, lat_max).
    """
    if df.empty:
        return df.copy()
    w, s, e, n = bbox
    mask = (
        (df["longitude"] >= w) & (df["longitude"] <= e) &
        (df["latitude"] >= s) & (df["latitude"] <= n)
    )
    return df[mask].copy()


def filter_dates(
    df: pd.DataFrame,
    fi: Union[datetime.date, datetime.datetime, str],
    ff: Union[datetime.date, datetime.datetime, str]
) -> pd.DataFrame:
    """
    Filtra detecciones entre fecha inicial y fecha final (inclusiva hasta final del día).
    """
    if df.empty:
        return df.copy()
    ts_start = pd.to_datetime(fi)
    ts_end = pd.to_datetime(ff) + pd.Timedelta(days=1, microseconds=-1)
    mask = (df["timestamp"] >= ts_start) & (df["timestamp"] <= ts_end)
    return df[mask].copy()


def monthly_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el conteo de detecciones por año, mes numérico y nombre de mes.
    """
    if df.empty:
        return pd.DataFrame(columns=["year", "month", "month_name", "count"])
    res = df.groupby(["year", "month"]).size().reset_index(name="count")
    res["month_name"] = res["month"].map(MESES_ES)
    return res[["year", "month", "month_name", "count"]]


def calendar_matrix(
    df: pd.DataFrame,
    all_years: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Genera la matriz año x mes con los 12 meses cronológicos garantizados.
    Rellena con 0 los meses y años sin detecciones dentro del rango analizado.
    """
    if df.empty:
        if all_years:
            empty_df = pd.DataFrame(0, index=sorted(all_years), columns=range(1, 13))
            empty_df.columns = [MESES_ES[m] for m in empty_df.columns]
            return empty_df
        return pd.DataFrame(columns=NOMBRES_MESES)

    m = monthly_counts(df)
    pivot = m.pivot_table(index="year", columns="month", values="count", fill_value=0)

    # Asegurar que los 12 meses siempre existan como columnas
    for month_num in range(1, 13):
        if month_num not in pivot.columns:
            pivot[month_num] = 0

    pivot = pivot[sorted(pivot.columns)]
    pivot.columns = [MESES_ES[m] for m in pivot.columns]

    # Asegurar que todos los años esperados figuren en el índice
    if all_years:
        for y in all_years:
            if y not in pivot.index:
                pivot.loc[y] = 0
        pivot = pivot.loc[sorted(pivot.index)]

    return pivot.sort_index(ascending=True)


def historical_median(
    df: pd.DataFrame,
    baseline_years: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Calcula la mediana histórica mensual para la región actual.
    Metodología Sprint 3:
    - Utiliza los años históricos disponibles (por defecto, años anteriores al año máximo o lista explícita).
    - Imputa 0 para los meses donde un año de referencia no tuvo detecciones de fuego,
      evitando sobreestimar la mediana al promediar solo meses positivos.
    """
    if df.empty:
        return pd.DataFrame(columns=["month", "month_name", "median_historical", "n_years"])

    available_years = sorted(df["year"].unique())

    if baseline_years is not None:
        years_for_baseline = [y for y in baseline_years if y in available_years]
    else:
        # Por defecto, todos los años previos al año más reciente observado
        max_year = df["year"].max()
        years_for_baseline = [y for y in available_years if y < max_year]

    n_years = len(years_for_baseline)
    if n_years == 0:
        # Si no hay años previos (ej. solo hay 1 año), usar todos los años disponibles
        years_for_baseline = available_years
        n_years = len(years_for_baseline)

    # Filtrar solo registros de los años de línea base
    hist_df = df[df["year"].isin(years_for_baseline)]
    hist_counts = monthly_counts(hist_df)

    # Crear cuadrícula completa (año_base x mes) con imputación de 0
    records = []
    for m in range(1, 13):
        month_vals = []
        for y in years_for_baseline:
            match = hist_counts[(hist_counts["year"] == y) & (hist_counts["month"] == m)]
            cnt = int(match["count"].iloc[0]) if not match.empty else 0
            month_vals.append(cnt)
        med = float(np.median(month_vals)) if month_vals else 0.0
        records.append({
            "month": m,
            "month_name": MESES_ES[m],
            "median_historical": med,
            "n_years": n_years
        })

    return pd.DataFrame(records)


def compute_anomalies(
    df: pd.DataFrame,
    median_df: Optional[pd.DataFrame] = None,
    threshold_high: float = ANOMALY_THRESHOLD_HIGH,
    threshold_low: float = ANOMALY_THRESHOLD_LOW
) -> pd.DataFrame:
    """
    Calcula el ratio de anomalía por período (año-mes) comparado con la mediana histórica.
    Maneja con rigor matemático el caso de mediana histórica igual a 0.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "year", "month", "month_name", "count", "median_historical",
            "n_years", "anomaly_ratio", "anomaly_label", "year_month", "display_ratio"
        ])

    monthly = monthly_counts(df)

    if median_df is None or median_df.empty:
        median_df = historical_median(df)

    merged = monthly.merge(median_df, on="month", how="left")
    merged["month_name"] = merged["month"].map(MESES_ES)

    # Formato ordenable para el eje X
    merged["year_month"] = (
        merged["year"].astype(str) + "-" + merged["month"].astype(str).str.zfill(2)
    )

    ratios = []
    labels = []
    display_ratios = []

    for _, row in merged.iterrows():
        cnt = row["count"]
        med = row.get("median_historical", np.nan)
        n_yrs = row.get("n_years", 0)

        if pd.isna(med) or n_yrs == 0:
            ratios.append(np.nan)
            labels.append("sin_referencia")
            display_ratios.append("Sin referencia")
        elif med > 0:
            r = cnt / med
            ratios.append(r)
            display_ratios.append(f"{r:.2f}x")
            if r >= threshold_high:
                labels.append("alta")
            elif r <= threshold_low:
                labels.append("baja")
            else:
                labels.append("normal")
        else:
            # Mediana histórica es 0
            if cnt == 0:
                # 0 focos observados vs 0 de mediana histórica -> comportamiento normal esperado
                ratios.append(1.0)
                labels.append("normal")
                display_ratios.append("1.00x (esperado 0)")
            else:
                # Hubo fuegos donde históricamente la mediana era cero -> Anomalía alta
                # Se asigna un valor representativo para graficar sin provocar división por cero
                ratios.append(threshold_high * 1.5)
                labels.append("alta (mediana 0)")
                display_ratios.append(f"{cnt} focos (mediana 0)")

    merged["anomaly_ratio"] = ratios
    merged["anomaly_label"] = labels
    merged["display_ratio"] = display_ratios

    return merged.sort_values(["year", "month"]).reset_index(drop=True)


def sanitize_for_csv(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Sanitiza campos de texto para evitar ataques de CSV Injection (fórmulas maliciosas
    como =cmd|..., +cmd|..., etc.) al abrir el archivo en hojas de cálculo (Excel, LibreOffice).
    """
    if df.empty:
        return df.copy()

    export_df = df[columns].copy() if columns else df.copy()

    # Prefijos peligrosos que disparan ejecución de fórmulas en hojas de cálculo
    dangerous_prefixes = ("=", "+", "-", "@", "\t", "\r")

    for col in export_df.select_dtypes(include=["object", "string"]).columns:
        export_df[col] = export_df[col].astype(str).apply(
            lambda val: f"'{val}" if val.startswith(dangerous_prefixes) else val
        )

    return export_df
