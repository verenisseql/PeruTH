"""Calendario semanal PROVISIONAL (domingo-sábado; semana 1 contiene 4 de enero).

No es una garantía de concordancia histórica con todas las publicaciones del CDC Perú.
Semanas fuera de rango se auditan, nunca se reasignan automáticamente.
"""
import datetime as dt
import unicodedata
import re
import pandas as pd


def normalize_region(value):
    if pd.isna(value):
        return None
    s = unicodedata.normalize("NFKD", str(value))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s.strip()).upper()
    aliases = {"PROVINCIA CONSTITUCIONAL DEL CALLAO": "CALLAO", "LIMA PROVINCIAS": "LIMA", "LIMA METROPOLITANA": "LIMA", "SAN MARTIN": "SAN MARTIN"}
    return aliases.get(s, s) or None


def epi_start(year: int) -> pd.Timestamp:
    jan4 = pd.Timestamp(int(year), 1, 4)
    return jan4 - pd.Timedelta(days=(jan4.dayofweek + 1) % 7)


def epi_week_count(year: int) -> int:
    return (epi_start(year + 1) - epi_start(year)).days // 7


def add_epi_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ano"] = pd.to_numeric(out["ano"], errors="coerce")
    out["semana"] = pd.to_numeric(out["semana"], errors="coerce")
    valid_num = out["ano"].between(1900, 2100) & out["semana"].between(1, 53)
    out["inicio_semana"] = pd.NaT
    out["calendario_valido"] = False
    if valid_num.any():
        idx = out.index[valid_num]
        dates = [epi_start(int(y)) + pd.Timedelta(weeks=int(w) - 1)
                 for y, w in zip(out.loc[idx, "ano"], out.loc[idx, "semana"])]
        out.loc[idx, "inicio_semana"] = pd.to_datetime(dates).to_numpy()
        out.loc[idx, "calendario_valido"] = [int(w) <= epi_week_count(int(y))
                                              for y, w in zip(out.loc[idx, "ano"], out.loc[idx, "semana"])]
    return out


def week_of_date(value):
    day = pd.Timestamp(value).normalize()
    y = day.year
    if day < epi_start(y):
        y -= 1
    elif day >= epi_start(y + 1):
        y += 1
    return int(y), 1 + (day - epi_start(y)).days // 7
