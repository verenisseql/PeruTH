"""Lectura + limpieza de CDC-MINSA: dengue por registro y malaria ya agregada.
No imputamos como cero las semanas SIN registros de dengue.
"""
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from .config import RAW, PROCESSED, REPORTS, CAPITALS
from .epi import normalize_region, add_epi_date


def dengue_clean(path: Path):
    usecols = ["departamento", "ano", "semana", "diagnostic", "tipo_dx"]
    pieces = []
    records_read = 0
    for batch in pd.read_csv(path, usecols=usecols, chunksize=150_000,
                             low_memory=False, encoding="utf-8-sig"):
        records_read += len(batch)
        batch["departamento"] = batch["departamento"].map(normalize_region)
        for c in ("ano", "semana"):
            batch[c] = pd.to_numeric(batch[c], errors="coerce")
        batch["tipo_dx"] = batch["tipo_dx"].astype("string").str.strip().str.upper()
        batch["diagnostic"] = batch["diagnostic"].astype("string").str.strip().str.upper()
        # Solo códigos A97.0/1/2, C=confirmado, P=probable. Una fila = un registro.
        good = batch["departamento"].isin(CAPITALS) & batch["ano"].between(2000,2025) & batch["semana"].between(1,53)
        good &= batch["tipo_dx"].isin(["C","P"]) & batch["diagnostic"].isin(["A97.0","A97.1","A97.2"])
        b = batch.loc[good].copy()
        b["confirmados"] = b["tipo_dx"].eq("C").astype(int)
        b["probables"] = b["tipo_dx"].eq("P").astype(int)
        pieces.append(b.groupby(["departamento","ano","semana"],as_index=False)[["confirmados","probables"]].sum())
    if not pieces:
        raise ValueError("No se encontraron registros de dengue válidos")
    weekly = pd.concat(pieces).groupby(["departamento","ano","semana"],as_index=False)[["confirmados","probables"]].sum()
    weekly["casos"] = weekly["confirmados"] + weekly["probables"]
    weekly["enfermedad"] = "DENGUE"
    weekly["origen"] = "notificaciones_dengue"
    weekly["observado_fuente"] = True
    return weekly, {"filas_crudas_dengue": records_read, "filas_agrupadas_dengue": len(weekly)}


def malaria_clean(folder: Path):
    parts = []
    audit = []
    files = sorted(folder.glob("malaria_2000-2025_*.csv"))
    if not files:
        raise FileNotFoundError(f"No hay CSV malaria_2000-2025_*.csv en {folder}")
    for path in files:
        region = normalize_region(path.stem.replace("malaria_2000-2025_", ""))
        if region not in CAPITALS:
            raise ValueError(f"Departamento malaria desconocido: {path.name} => {region}")
        frame = pd.read_csv(path, usecols=["ano","semana","provincia","malaria_total"],encoding="utf-8-sig")
        frame["ano"] = pd.to_numeric(frame["ano"],errors="coerce")
        frame["semana"] = pd.to_numeric(frame["semana"],errors="coerce")
        frame["malaria_total"] = pd.to_numeric(frame["malaria_total"],errors="coerce")
        valid = frame["ano"].between(2000,2025) & frame["semana"].between(1,53) & frame["malaria_total"].ge(0)
        audit.append({"archivo":path.name,"departamento":region,"filas":len(frame),"filas_invalidas":int((~valid).sum())})
        frame = frame.loc[valid].copy()
        frame["departamento"] = region
        parts.append(frame)
    full = pd.concat(parts,ignore_index=True)
    weekly = full.groupby(["departamento","ano","semana"],as_index=False).agg(casos=("malaria_total","sum"))
    weekly["confirmados"] = np.nan  # Fuente no distingue diagnóstico C/P
    weekly["probables"] = np.nan
    weekly["enfermedad"] = "MALARIA"
    weekly["origen"] = "totales_malaria_provincia"
    weekly["observado_fuente"] = True
    return weekly, pd.DataFrame(audit)


def run_etl():
    PROCESSED.mkdir(parents=True,exist_ok=True)
    REPORTS.mkdir(parents=True,exist_ok=True)
    dengue_files = sorted((RAW/"dengue").glob("*.csv"))
    if len(dengue_files) != 1:
        raise ValueError(f"Se esperaba un CSV de dengue en {RAW/'dengue'}; encontrados: {len(dengue_files)}")
    dengue, stats = dengue_clean(dengue_files[0])
    malaria, audit_m = malaria_clean(RAW/"malaria")
    union = pd.concat([dengue,malaria],ignore_index=True)
    union = add_epi_date(union)
    union["casos"] = union["casos"].astype(int)
    # Sin temporalización inventada: validamos y guardamos semanas no compatibles.
    rejected = union.loc[~union["calendario_valido"]].copy()
    accepted = union.loc[union["calendario_valido"]].copy()
    accepted = accepted.sort_values(["enfermedad","departamento","inicio_semana"])
    accepted.to_csv(PROCESSED/"epidemiologia_semanal.csv",index=False)
    rejected.to_csv(REPORTS/"semanas_epidemiologicas_pendientes.csv",index=False)
    audit_m.to_csv(REPORTS/"auditoria_archivos_malaria.csv",index=False)
    coverage = union.groupby(["enfermedad","departamento","ano"],as_index=False).agg(
        semanas_con_registro=("semana","nunique"),casos_registrados=("casos","sum"),
        semanas_fuera_calendario=("calendario_valido",lambda x:int((~x).sum())))
    coverage.to_csv(REPORTS/"cobertura_epi.csv",index=False)
    print("ETL dengue:",stats)
    print("Archivos malaria:",len(audit_m),"| semanas aceptadas:",len(accepted))
    print("Semanas pendientes:",len(rejected),"| casos pendientes:",int(rejected.casos.sum()))
    print("Departamentos con dengue:",accepted.loc[accepted.enfermedad.eq("DENGUE"),"departamento"].nunique())
    print("Departamentos con malaria:",accepted.loc[accepted.enfermedad.eq("MALARIA"),"departamento"].nunique())
    return accepted

if __name__ == "__main__":
    run_etl()
