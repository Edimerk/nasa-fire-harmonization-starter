"""Sprint 3: métricas reproducibles sobre detecciones armonizadas.

Los conteos representan DETECCIONES, no incendios individuales ni área quemada.
La mediana/anomalía solo se calcula con >=5 años de referencia explícitamente
marcados como completos; no se inventan ceros para meses sin cobertura.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

MIN_BASELINE_YEARS = 5


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    required = {"latitude", "longitude", "timestamp", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {', '.join(sorted(missing))}")
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out = out.loc[out.timestamp.notna() & out.latitude.between(-90, 90) & out.longitude.between(-180, 180)].copy()
    out["year"] = out.timestamp.dt.year.astype(int)
    out["month"] = out.timestamp.dt.month.astype(int)
    out["date"] = out.timestamp.dt.date
    return out


def filter_region_dates(df: pd.DataFrame, bbox=None, start=None, end=None) -> pd.DataFrame:
    """BBOX=(oeste,sur,este,norte), fechas inclusive en UTC."""
    out = prepare(df)
    if bbox is not None:
        if len(bbox) != 4:
            raise ValueError("bbox requiere oeste sur este norte")
        west, south, east, north = map(float, bbox)
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError("bbox inválido")
        out = out.loc[out.longitude.between(west, east) & out.latitude.between(south, north)].copy()
    if start is not None and end is not None and pd.Timestamp(start).date() > pd.Timestamp(end).date():
        raise ValueError("Fecha inicial posterior a fecha final")
    if start is not None:
        out = out.loc[out.date >= pd.Timestamp(start).date()].copy()
    if end is not None:
        out = out.loc[out.date <= pd.Timestamp(end).date()].copy()
    return out


def monthly_yearly_counts(df: pd.DataFrame):
    """Devuelve conteos mensual y anual por fuente, listos para visualización."""
    out = prepare(df)
    monthly = (out.groupby(["year", "month", "source"]).size()
               .rename("detections").reset_index().sort_values(["year", "month", "source"]))
    yearly = (out.groupby(["year", "source"]).size()
              .rename("detections").reset_index().sort_values(["year", "source"]))
    monthly["year_month"] = monthly.apply(lambda r: f"{int(r.year):04d}-{int(r.month):02d}", axis=1)
    return monthly, yearly


def monthly_total_grid(df: pd.DataFrame, years: list[int], *, full_coverage: bool) -> pd.DataFrame:
    """Cuadrícula año/mes. Los meses sin detección solo son cero con cobertura verificada."""
    out = prepare(df)
    agg = out.groupby(["year", "month"]).size().rename("detections")
    idx = pd.MultiIndex.from_product([sorted(set(years)), range(1, 13)], names=["year", "month"])
    grid = agg.reindex(idx).reset_index()
    if full_coverage:
        grid["detections"] = grid["detections"].fillna(0).astype(int)
    return grid


def historical_monthly_median(df: pd.DataFrame, baseline_years: list[int], *, full_coverage: bool = False) -> pd.DataFrame:
    """Requiere >=5 años distintos y cobertura completa explícitamente confirmada."""
    years = sorted(set(map(int, baseline_years)))
    if len(years) < MIN_BASELINE_YEARS:
        raise ValueError("Mediana no disponible: se requieren al menos 5 años de referencia")
    if not full_coverage:
        raise ValueError("Mediana no disponible: confirme cobertura completa de los años base")
    grid = monthly_total_grid(df, years, full_coverage=True)
    return (grid.groupby("month")["detections"].median()
            .rename("historical_median").reset_index())


def monthly_anomalies(df: pd.DataFrame, baseline_years: list[int], *, full_coverage: bool = False) -> pd.DataFrame:
    """Anomalía absoluta y porcentual mensual, con cero de base marcado no definido."""
    out = prepare(df)
    median = historical_monthly_median(out, baseline_years, full_coverage=full_coverage)
    counts = out.groupby(["year", "month"]).size().rename("detections").reset_index()
    res = counts.merge(median, on="month", how="left")
    res["anomaly_absolute"] = res.detections - res.historical_median
    res["anomaly_percent"] = (res.anomaly_absolute / res.historical_median.where(res.historical_median.ne(0))) * 100
    return res.sort_values(["year", "month"]).reset_index(drop=True)


def rankings(df: pd.DataFrame):
    out = prepare(df)
    years = out.groupby("year").size().rename("detections").reset_index()
    months = out.groupby(["year", "month"]).size().rename("detections").reset_index()
    return (years.sort_values(["detections", "year"], ascending=[False, True]).reset_index(drop=True),
            months.sort_values(["detections", "year", "month"], ascending=[False, True, True]).reset_index(drop=True))


def run(input_csv: Path, output_dir: Path, bbox=None, start=None, end=None,
        baseline_years=None, full_coverage=False):
    data = pd.read_csv(input_csv, low_memory=False)
    selected = filter_region_dates(data, bbox, start, end)
    monthly, yearly = monthly_yearly_counts(selected)
    rank_y, rank_m = rankings(selected)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_dir / "detecciones_filtradas.csv", index=False)
    monthly.to_csv(output_dir / "conteos_mensuales.csv", index=False)
    yearly.to_csv(output_dir / "conteos_anuales.csv", index=False)
    rank_y.to_csv(output_dir / "ranking_anual.csv", index=False)
    rank_m.to_csv(output_dir / "ranking_mensual.csv", index=False)
    state = "sin_referencia_suficiente"
    if baseline_years and len(set(baseline_years)) >= MIN_BASELINE_YEARS and full_coverage:
        # La referencia histórica se calcula sobre la REGIÓN, no sobre el periodo visualizado.
        region = filter_region_dates(data, bbox)
        baseline = region.loc[region.year.isin(baseline_years)]
        reference = historical_monthly_median(baseline, baseline_years, full_coverage=True)
        reference.to_csv(output_dir / "medianas_historicas.csv", index=False)
        observed = selected.groupby(["year", "month"]).size().rename("detections").reset_index()
        anomalies = observed.merge(reference, on="month", how="left")
        anomalies["anomaly_absolute"] = anomalies.detections - anomalies.historical_median
        anomalies["anomaly_percent"] = anomalies.anomaly_absolute.div(anomalies.historical_median.where(lambda s: s.ne(0))) * 100
        anomalies.to_csv(output_dir / "anomalias.csv", index=False)
        state = "calculadas_con_cobertura_declarada"
    report = {"input": str(input_csv), "selected_detections": int(len(selected)),
              "bbox": bbox, "start": start, "end": end, "baseline_years": baseline_years or [],
              "baseline_full_coverage_declared": bool(full_coverage),
              "anomalies_status": state,
              "note": "Conteos de detecciones, NO incendios únicos ni superficie quemada. Sin cobertura comprobada no se calculan anomalías."}
    (output_dir / "resumen_analitico.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sprint 3: estadísticas a partir de datos armonizados")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("OESTE", "SUR", "ESTE", "NORTE"))
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--baseline-years", type=int, nargs="+")
    parser.add_argument("--full-coverage", action="store_true", help="Solo si verificó cobertura mensual completa de los años base")
    args = parser.parse_args()
    run(args.input, args.output_dir, args.bbox, args.start, args.end, args.baseline_years, args.full_coverage)
