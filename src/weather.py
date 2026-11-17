"""Open-Meteo histórico (archive) y pronóstico; caché local persistente."""
from datetime import date, timedelta
from pathlib import Path
import time
import numpy as np
import pandas as pd
import requests
from .config import CAPITALS, INTERIM, PROCESSED
from .epi import week_of_date

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HIST_VARIABLES = "temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum"


def get_json(url, params, retries=3):
    for intento in range(retries):
        try:
            respuesta = requests.get(
                url,
                params=params,
                timeout=120
            )

            if respuesta.status_code == 429:
                if intento == retries - 1:
                    raise RuntimeError(
                        "Open-Meteo mantiene el limite de solicitudes "
                        "(HTTP 429). Las descargas guardadas se conservaran. "
                        "Intenta continuar mas adelante."
                    )

                retry_after = respuesta.headers.get("Retry-After")

                try:
                    espera = max(1, min(float(retry_after), 300))
                except (TypeError, ValueError):
                    espera = 60 * (intento + 1)

                print(
                    f"Limite de Open-Meteo. "
                    f"Reintentando tras {espera:.0f} segundos..."
                )
                time.sleep(espera)
                continue

            respuesta.raise_for_status()
            datos = respuesta.json()

            if datos.get("error"):
                raise RuntimeError(
                    datos.get("reason", "Error de Open-Meteo")
                )

            return datos

        except requests.RequestException as error:
            if intento == retries - 1:
                raise RuntimeError(
                    f"No se pudo consultar Open-Meteo: {error}"
                ) from error

            espera = 5 * (2 ** intento)
            print(f"Error de conexion. Reintentando en {espera}s...")
            time.sleep(espera)

def read_daily(payload, depart):
    daily = payload.get("daily", {})
    if not daily or not daily.get("time"):
        raise ValueError("Respuesta de Open-Meteo sin fechas diarias")
    length = len(daily["time"])
    out = pd.DataFrame({"fecha":pd.to_datetime(daily["time"]),"departamento":depart})
    for key in ["temperature_2m_mean","temperature_2m_max","temperature_2m_min","precipitation_sum","precipitation_probability_max"]:
        raw = daily.get(key)
        out[key] = pd.to_numeric(pd.Series(raw if raw is not None else [np.nan]*length),errors="coerce")
    # La API de pronóstico no proporciona la temperatura media diaria directamente:
    # calculamos la media de valores horarios cuando está disponible.
    if "temperature_2m_mean" not in daily and payload.get("hourly",{}).get("time"):
        hourly = pd.DataFrame({"fecha":pd.to_datetime(payload["hourly"]["time"]).normalize(),
                               "temp":pd.to_numeric(payload["hourly"].get("temperature_2m",[]),errors="coerce")})
        means = hourly.groupby("fecha")["temp"].mean()
        out["temperature_2m_mean"] = out["fecha"].map(means)
    return out



def fetch_history(depart, year_start=2000, year_end=2025):
    city, lat, lon = CAPITALS[depart]

    INTERIM.mkdir(parents=True, exist_ok=True)

    frames = []

    for start in range(year_start, year_end + 1, 5):
        end = min(start + 4, year_end)

        archivo = INTERIM / (
            f"openmeteo_{depart.replace(' ', '_')}_"
            f"{start}_{end}.csv"
        )

        # Reutilizar bloques descargados anteriormente
        if archivo.exists():
            df = pd.read_csv(
                archivo,
                parse_dates=["fecha"]
            )
            print(f"  {depart} {start}-{end}: ya descargado")

        else:
            print(f"  Descargando {depart}: {start}-{end}")

            payload = get_json(
                ARCHIVE_URL,
                {
                    "latitude": lat,
                    "longitude": lon,
                    "daily": HIST_VARIABLES,
                    "start_date": f"{start}-01-01",
                    "end_date": f"{end}-12-31",
                    "timezone": "America/Lima"
                }
            )

            df = read_daily(payload, depart)

            # Guardar inmediatamente cada bloque completado
            df.to_csv(archivo, index=False)

            print(f"  Guardado: {archivo.name}")

            # Espaciar las consultas
            time.sleep(3)

        frames.append(df)

    resultado = pd.concat(
        frames,
        ignore_index=True
    )

    if resultado.duplicated("fecha").any():
        raise ValueError(
            f"Fechas duplicadas para {depart}"
        )

    return resultado


def download_history(year_start=2000,year_end=2025,regions=None):
    """Reanudable: si existe un archivo por departamento, lo reutiliza.
    Tras completar los 25 departamentos, combina todo en un solo CSV.
    """
    INTERIM.mkdir(parents=True,exist_ok=True)
    PROCESSED.mkdir(parents=True,exist_ok=True)
    regions = list(regions or CAPITALS)
    frames=[]
    for i,depart in enumerate(regions,1):
        fn = INTERIM/f"openmeteo_{depart.replace(' ','_')}_{year_start}_{year_end}.csv"
        if fn.exists():
            df=pd.read_csv(fn,parse_dates=["fecha"])
            print(f"[{i}/{len(regions)}] {depart}: caché")
        else:
            df=fetch_history(depart,year_start,year_end)
            df.to_csv(fn,index=False)
            print(f"[{i}/{len(regions)}] {depart}: descargado {len(df)} días")
        frames.append(df)
    result=pd.concat(frames,ignore_index=True)
    result.to_csv(PROCESSED/"clima_diario_openmeteo.csv",index=False)
    print("Guardado:",PROCESSED/"clima_diario_openmeteo.csv")
    return result


def aggregate_weeks(daily):
    df=daily.copy()
    df["fecha"]=pd.to_datetime(df["fecha"])
    df["inicio_semana"] = df["fecha"] - pd.to_timedelta((df["fecha"].dt.dayofweek+1)%7,unit="D")
    grouped=df.groupby(["departamento","inicio_semana"],as_index=False).agg(
        tmean=("temperature_2m_mean","mean"),
        tmax=("temperature_2m_max","mean"),
        tmin=("temperature_2m_min","mean"),
        precipitacion=("precipitation_sum","sum"),
        dias=("fecha","nunique"),
        n_tmean=("temperature_2m_mean","count"),
        n_tmax=("temperature_2m_max","count"),
        n_tmin=("temperature_2m_min","count"),
        n_precip=("precipitation_sum","count"),
    )
    grouped = grouped.loc[
        grouped["dias"].eq(7) & grouped[["n_tmean","n_tmax","n_tmin","n_precip"]].eq(7).all(axis=1)
    ].copy()
    return grouped[["departamento","inicio_semana","tmean","tmax","tmin","precipitacion"]]


def build_weekly_weather():
    p=PROCESSED/"clima_diario_openmeteo.csv"
    if not p.exists():
        raise FileNotFoundError("Falta clima_diario_openmeteo.csv. Ejecuta python -m src.weather")
    weekly=aggregate_weeks(pd.read_csv(p,parse_dates=["fecha"]))
    weekly.to_csv(PROCESSED/"clima_semanal_openmeteo.csv",index=False)
    print("Semanas meteorológicas completas:",len(weekly))
    return weekly


def live_weather(depart):
    """Pronóstico local de 16 días y hasta 42 días previos (datos del servicio, no CDC).
    No se extrapola el pronóstico más allá del horizonte de la API.
    """
    city,lat,lon=CAPITALS[depart]
    payload=get_json(FORECAST_URL,{
        "latitude":lat,"longitude":lon,"timezone":"America/Lima",
        "past_days":42,"forecast_days":16,
        "hourly":"temperature_2m",
        "daily":"temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max"
    })
    return read_daily(payload,depart)


if __name__ == "__main__":
    download_history()
    build_weekly_weather()
