"""Integración y rezagos sin utilizar variables futuras para entrenar."""
import numpy as np
import pandas as pd
from .config import PROCESSED
from .epi import week_of_date

CLIMATE = ["tmean","tmax","tmin","precipitacion"]
CLIMATE_LAGS = [f"{c}_lag{k}" for c in CLIMATE for k in range(1,5)]
CASE_LAGS = [f"casos_lag{k}" for k in range(1,5)]
BASE_FEATURES = ["departamento", "sin_semana", "cos_semana", "ano_tendencia"]
TRAVEL_FEATURES = BASE_FEATURES+CLIMATE_LAGS
FULL_FEATURES = TRAVEL_FEATURES+CASE_LAGS


def calendar_columns(df):
    out=df.copy()
    # La misma regla de calendarización para todas las fuentes.
    from .epi import week_of_date
    week_info=out["inicio_semana"].map(week_of_date)
    out["ano_epi"]=week_info.map(lambda x:x[0])
    out["semana_epi"]=week_info.map(lambda x:x[1])
    out["sin_semana"]=np.sin(2*np.pi*out["semana_epi"]/53)
    out["cos_semana"]=np.cos(2*np.pi*out["semana_epi"]/53)
    out["ano_tendencia"]=out["ano_epi"]-2000
    return out


def prepare_training():
    weather=pd.read_csv(PROCESSED/"clima_semanal_openmeteo.csv",parse_dates=["inicio_semana"])
    epi=pd.read_csv(PROCESSED/"epidemiologia_semanal.csv",parse_dates=["inicio_semana"])
    weather=weather.sort_values(["departamento","inicio_semana"])
    # Desplazamiento climático: generar rezagos en calendario METEOROLÓGICO completo
    # antes de filtrar semanas epidemiológicas observadas.
    for c in CLIMATE:
        for k in range(1,5):
            weather[f"{c}_lag{k}"]=weather.groupby("departamento")[c].shift(k)
    merged=[]
    for disease in ["DENGUE","MALARIA"]:
        source=epi.loc[epi.enfermedad.eq(disease)].copy()
        source=source.sort_values(["departamento","inicio_semana"])
        # No construir lags sobre filas de dengue faltantes: esas semanas NO son 0.
        # Exigimos que la semana anterior tenga fecha consecutiva (-7 días).
        for k in range(1,5):
            prev=source.groupby("departamento")["casos"].shift(k)
            prev_date=source.groupby("departamento")["inicio_semana"].shift(k)
            consecutive=(source["inicio_semana"]-prev_date).dt.days.eq(7*k)
            source[f"casos_lag{k}"]=prev.where(consecutive)
        combined=source.merge(weather,on=["departamento","inicio_semana"],how="left",validate="one_to_one")
        merged.append(combined)
    all_data=calendar_columns(pd.concat(merged,ignore_index=True))
    all_data=all_data.sort_values(["enfermedad","departamento","inicio_semana"])
    all_data.to_csv(PROCESSED/"dataset_modelado.csv",index=False)
    print("Registros etiquetados (sin rellenar como cero):",len(all_data))
    print("Disponibles para modelo de viajeros:",all_data[TRAVEL_FEATURES].dropna().shape[0])
    print("Disponibles con rezagos de casos:",all_data[FULL_FEATURES].dropna().shape[0])
    return all_data


def live_features(depart,chosen_date,daily):
    """Pronóstico de riesgo para semana seleccionada usando SOLO las 4 semanas COMPLETAS previas.
    No requiere casos 2026, que no están incluidos en el archivo CDC 2000-2025.
    """
    chosen=pd.Timestamp(chosen_date)
    target_start=chosen-pd.Timedelta(days=(chosen.dayofweek+1)%7)
    agg=aggregate_live_weeks(daily)
    row={"departamento":depart}
    for k in range(1,5):
        when=target_start-pd.Timedelta(weeks=k)
        sub=agg.loc[agg["inicio_semana"].eq(when)]
        if sub.empty:
            return None
        for c in CLIMATE:
            row[f"{c}_lag{k}"]=float(sub.iloc[0][c])
    year,week=week_of_date(target_start)
    row["sin_semana"]=float(np.sin(2*np.pi*week/53))
    row["cos_semana"]=float(np.cos(2*np.pi*week/53))
    row["ano_tendencia"]=year-2000
    return pd.DataFrame([row])


def aggregate_live_weeks(daily):
    from .weather import aggregate_weeks
    return aggregate_weeks(daily)


if __name__=="__main__":
    prepare_training()
