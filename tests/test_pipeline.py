import pandas as pd
from src.epi import normalize_region,epi_week_count,add_epi_date,week_of_date
from src.weather import aggregate_weeks,read_daily
from src.features import live_features


def test_accents():
    assert normalize_region('  San   Martín ') == 'SAN MARTIN'
    assert normalize_region('  Áncash  ') == 'ANCASH'


def test_provisional_calendar_flags_week53():
    d=add_epi_date(pd.DataFrame({'ano':[2009,2025],'semana':[53,53]}))
    assert d.calendario_valido.tolist()==[False,True]
    assert week_of_date('2025-01-01')==(2025,1)


def test_seven_days_only():
    d=pd.DataFrame({'fecha':pd.date_range('2025-01-05',periods=8),
                    'departamento':['PIURA']*8,'temperature_2m_mean':[28]*8,
                    'temperature_2m_max':[30]*8,'temperature_2m_min':[24]*8,
                    'precipitation_sum':[2]*8})
    result=aggregate_weeks(d)
    assert len(result)==1
    assert result.precipitacion.iloc[0]==14


def test_missing_week_no_imputation():
    d=pd.DataFrame({'fecha':pd.date_range('2025-01-05',periods=28),
                    'departamento':['PIURA']*28,'temperature_2m_mean':[28]*28,
                    'temperature_2m_max':[30]*28,'temperature_2m_min':[24]*28,
                    'precipitation_sum':[2]*28})
    assert live_features('PIURA','2025-02-02',d) is not None
    assert live_features('PIURA','2025-02-09',d) is None
