"""Dashboard para viajeros: Streamlit y Open-Meteo.
NO es un aviso clínico oficial ni probabilidad de infección individual.
La lógica epidemiológica permanece sin cambios; solo se rediseña la interfaz.
"""
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import escape

import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.config import CAPITALS, MODELS, PROCESSED
from src.weather import live_weather
from src.features import live_features
from src.train import inference

st.set_page_config(page_title="Perú | Travel Health", layout="wide", initial_sidebar_state="collapsed")

# Estilos encapsulados: no requiere editar otros archivos ni instalar paquetes.
st.markdown("""
<style>
:root { --page:#f3f5f0; --white:#ffffff; --ink:#151a15; --sub:#737b71;
        --lime:#b9ef39; --lime-deep:#95cd18; --soft:#eef8d9;
        --stroke:#e7ebe3; --shadow:0 15px 45px rgba(31,46,24,.045); }
.stApp, [data-testid="stAppViewContainer"] { background:var(--page); color:var(--ink); }
.block-container { max-width:1330px; padding:1.25rem 2.25rem 4rem; }
header[data-testid="stHeader"] { background:transparent; }
#MainMenu, footer, [data-testid="stDecoration"] { display:none!important; }
html, body, [data-testid="stAppViewContainer"] { font-family:Inter, 'Segoe UI', Arial, sans-serif; }
@keyframes fade-rise { from { opacity:0; transform:translateY(15px); } to { opacity:1; transform:translateY(0); } }
@keyframes bar-grow { from { width:0; } }
.topbar { display:flex; justify-content:space-between; align-items:center; gap:16px;
  padding:12px 19px; background:white; border:1px solid var(--stroke); border-radius:19px;
  margin-bottom:19px; animation:fade-rise .55s ease both; }
.brand { display:flex; align-items:center; gap:11px; font-size:14px; font-weight:850; letter-spacing:.04em; }
.brand-logo { display:inline-flex; justify-content:center; align-items:center; width:31px; height:31px;
  border-radius:10px; background:var(--lime); color:#16220b; font-size:17px; font-weight:900; }
.top-meta { font-size:11px; letter-spacing:.05em; color:#7c8675; font-weight:750; }
.hero { min-height:247px; padding:38px 41px; border:1px solid var(--stroke); border-radius:29px;
  background:radial-gradient(ellipse 51% 110% at 97% 2%,#e5f9bc 0%,transparent 89%),white;
  position:relative; overflow:hidden; display:flex; align-items:center; justify-content:space-between;
  gap:25px; box-shadow:var(--shadow); animation:fade-rise .62s .06s ease both; margin-bottom:20px; }
.eyebrow { display:inline-flex; align-items:center; gap:8px; color:#405c12; padding:8px 13px; border-radius:999px;
  font-size:11px; letter-spacing:.09em; text-transform:uppercase; font-weight:800; background:#eff9d9; }
.eyebrow:before { content:''; width:7px; height:7px; background:#8bbb0f; border-radius:50%; }
.hero h1 { color:var(--ink)!important; font-size:clamp(33px,4vw,52px); line-height:1.08;
  letter-spacing:-.055em; margin:17px 0 13px; font-weight:850; }
.hero p { color:#737b71; max-width:610px; line-height:1.65; font-size:15px; margin:0; }
.hero-art { position:relative; flex:0 0 285px; height:170px; }
.orb { width:153px; height:153px; border-radius:42px; background:linear-gradient(145deg,#d9fa84,#a8e924);
  transform:rotate(-13deg); position:absolute; top:5px; left:67px;
  box-shadow:0 17px 45px rgba(136,188,24,.23); }
.orb:after { content:'TH'; position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
  transform:rotate(13deg); font-size:48px; font-weight:900; letter-spacing:-.09em; color:#213212; }
.float-label { position:absolute; background:white; padding:12px 16px; border-radius:13px; border:1px solid #e6ecdf;
  font-size:11px; font-weight:850; color:#35412a; box-shadow:0 10px 23px rgba(30,45,16,.1); }
.float-label.one { right:1px; top:16px; transform:rotate(5deg); }
.float-label.two { left:0; bottom:18px; transform:rotate(-5deg); }
.section-head { display:flex; align-items:end; justify-content:space-between; gap:15px; margin:32px 0 15px;
  animation:fade-rise .55s .13s ease both; }
.section-head h2 { font-size:23px; line-height:1.2; letter-spacing:-.038em; font-weight:800; color:var(--ink);
  margin:0 0 5px; padding:0; }
.section-head p { margin:0; font-size:13px; color:var(--sub); }
.section-tag { font-size:10px; color:#68755e; font-weight:850; letter-spacing:.11em; text-transform:uppercase;
  border:1px solid #dee5d8; border-radius:999px; padding:8px 12px; white-space:nowrap; }
/* Los contenedores de Streamlit son tarjetas auténticas; los widgets siguen siendo nativos. */
[data-testid="stVerticalBlockBorderWrapper"] { border:1px solid var(--stroke)!important; border-radius:25px!important;
  background:white; box-shadow:var(--shadow); overflow:hidden; animation:fade-rise .55s .16s ease both;
  transition:transform .24s ease,box-shadow .24s ease; }
[data-testid="stVerticalBlockBorderWrapper"]:hover { box-shadow:0 17px 47px rgba(31,46,24,.075); }
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] { gap:.55rem; }
.card-head { font-size:16px; font-weight:820; color:#222b1b; letter-spacing:-.028em; margin:0; }
.card-desc { color:var(--sub); font-size:12px; line-height:1.65; margin:4px 0 9px; }
.card-number { font-size:13px; font-weight:750; color:#6f7867; letter-spacing:.06em; }
.status { padding:7px 11px; border-radius:100px; text-transform:uppercase; font-size:10px;
  letter-spacing:.07em; font-weight:850; white-space:nowrap; }
.status-line { display:flex; gap:10px; justify-content:space-between; align-items:center; }
.pred-value { font-size:47px; line-height:1.1; margin-top:15px; font-weight:850; letter-spacing:-.065em; color:#172113; }
.pred-value small { font-size:18px; font-weight:700; color:#7d8675; }
.pred-label { font-size:12px; color:#828b7a; margin-top:4px; }
.progress-head { display:flex; justify-content:space-between; font-size:12px; margin:18px 0 9px; color:#6a7463; }
.progress-head strong { color:#273123; }
.progress-track { background:#eef2e8; height:9px; border-radius:40px; overflow:hidden; }
.progress-fill { height:100%; border-radius:40px; animation:bar-grow .85s ease both; }
.mini-kpi { background:#f6f8f3; padding:17px 18px; border-radius:17px; margin-top:17px; }
.mini-kpi .value { display:block; color:#1a2517; font-size:27px; letter-spacing:-.04em; font-weight:850; }
.mini-kpi .caption { display:block; color:#778172; font-size:11px; line-height:1.5; }
.weather-card { background:#fff; border:1px solid var(--stroke); border-radius:25px; padding:23px 24px;
  min-height:167px; box-shadow:var(--shadow); animation:fade-rise .6s ease both;
  transition:transform .25s ease,box-shadow .25s ease; }
.weather-card:hover { transform:translateY(-3px); box-shadow:0 15px 40px rgba(30,45,18,.085); }
.weather-card .overline { color:#85907d; font-size:10px; font-weight:800; letter-spacing:.11em; text-transform:uppercase; }
.weather-card .metric { font-size:38px; font-weight:850; line-height:1.35; letter-spacing:-.05em; color:#1c2617; margin-top:12px; }
.weather-card .metric small { font-size:16px; color:#899382; font-weight:650; }
.weather-card .foot { font-size:12px; color:#7d8775; margin-top:8px; }
.notice { background:#f0f8df; border:1px solid #d9e9b8; border-radius:18px; padding:16px 19px;
  color:#4c6034; font-size:12px; line-height:1.7; margin-top:15px; }
.tip-row { display:flex; gap:15px; background:#f7f9f4; border-radius:15px; padding:15px 18px; margin:7px 0; }
.tip-no { display:flex; align-items:center; justify-content:center; flex:0 0 32px; height:32px; border-radius:10px;
  background:#e9f6cd; color:#537918; font-weight:850; font-size:12px; }
.tip-row b { display:block; color:#27311f; font-size:13px; margin-bottom:4px; }
.tip-row span { display:block; color:#788272; font-size:12px; line-height:1.55; }
.stButton > button { background:var(--lime); color:#23300c; border:1px solid var(--lime);
  font-weight:800; border-radius:999px; padding:10px 19px; min-height:42px;
  box-shadow:0 7px 16px rgba(154,203,39,.18); transition:all .2s ease; }
.stButton > button:hover { background:#a8e22c; color:#1e2b0a; transform:translateY(-2px); border-color:#a8e22c; }
.stButton > button:focus-visible { outline:3px solid #547d00; outline-offset:3px; }
[data-baseweb="select"] > div, [data-baseweb="input"] > div { border-radius:13px!important; background:#f7f9f5!important; }
[data-testid="stRadio"] [role="radiogroup"] { gap:12px; }
[data-testid="stExpander"] { border:1px solid var(--stroke)!important; border-radius:19px!important; background:white; }
[data-testid="stPlotlyChart"] { border-radius:18px; overflow:hidden; }
@media(max-width:800px) { .block-container{padding:1rem 1rem 3rem;} .hero{padding:28px 22px;min-height:unset;}
  .hero-art{display:none;} .hero h1{font-size:35px;} .top-meta{display:none;} .section-tag{display:none;} }
@media(prefers-reduced-motion:reduce) { *, *::before, *::after { animation:none!important; transition:none!important; } }
</style>
""", unsafe_allow_html=True)


def section(title, subtitle, tag=""):
    tag_markup = f'<div class="section-tag">{escape(tag)}</div>' if tag else ""
    st.markdown(
        f'<div class="section-head"><div><h2>{escape(title)}</h2>'
        f'<p>{escape(subtitle)}</p></div>{tag_markup}</div>',
        unsafe_allow_html=True,
    )


def weather_tile(label, value, unit, sub, delay=0):
    unit_markup = f"<small>{escape(unit)}</small>" if unit else ""
    st.markdown(
        f'<div class="weather-card" style="animation-delay:{int(delay)}ms">'
        f'<div class="overline">{escape(label)}</div>'
        f'<div class="metric">{escape(value)} {unit_markup}</div>'
        f'<div class="foot">{escape(sub)}</div></div>',
        unsafe_allow_html=True,
    )


st.markdown("""
<div class="topbar">
  <div class="brand"><span class="brand-logo">T</span> TRAVEL HEALTH <span style="color:#a2aaa0;font-weight:500">/ PERÚ</span></div>
  <div class="top-meta">INTELIGENCIA DE VIAJE &nbsp; / &nbsp; MODELO EXPERIMENTAL</div>
</div>
<div class="hero">
  <div><span class="eyebrow">Plataforma de información para viajeros</span>
    <h1>Viaja informado.<br>Explora con confianza.</h1>
    <p>Consulta el pronóstico climático y las señales experimentales de actividad histórica de dengue y malaria para tu destino en Perú.</p>
  </div>
  <div class="hero-art" aria-hidden="true"><div class="orb"></div>
    <div class="float-label one">CLIMA Y PRONÓSTICO</div>
    <div class="float-label two">DENGUE / MALARIA</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ------------------------ LÓGICA DEL PROYECTO: SIN CAMBIOS ------------------------
@st.cache_resource
def load_models():
    result={}
    for d in ["DENGUE","MALARIA"]:
        path=MODELS/f"{d.lower()}_travel.joblib"
        if path.exists():
            result[d]=joblib.load(path)
    return result


@st.cache_data(ttl=1800,show_spinner=False)
def weather_for(region):
    return live_weather(region)


def predict(region,selected,models):
    w=weather_for(region)
    selected_ts=pd.Timestamp(selected)
    day=w.loc[w.fecha.eq(selected_ts)]
    if day.empty:
        return w,None,{}
    feature=live_features(region,selected,w)
    if feature is None:
        return w,day.iloc[0],{}
    out={}
    for disease,model in models.items():
        if region not in model["thresholds"]:
            continue
        out[disease]=inference(model,feature)
    return w,day.iloc[0],out


def gauge(title,pred):
    """Componente visual. Las cifras vienen directamente de inference()."""
    if pred is None:
        st.markdown(f'<div class="card-head">{escape(title)}</div>', unsafe_allow_html=True)
        st.info("Sin datos suficientes para estimar la actividad de esta enfermedad.")
        return
    palette={"Bajo":("#218967","#e7f5ed"), "Moderado":("#b88319","#fff4d8"), "Alto":("#c64b42","#fdebe8")}
    level=str(pred["nivel"])
    color,soft=palette.get(level,("#66736a","#eff2ec"))
    confidence=max(0.0,min(100.0,float(pred["confianza_categoria"])*100))
    high=max(0.0,min(100.0,float(pred["prob_alto"])*100))
    st.markdown(
        f'<div class="status-line"><div class="card-head">{escape(title)}</div>'
        f'<span class="status" style="color:{color};background:{soft}">{escape(level)}</span></div>'
        '<div class="card-desc">Señal de actividad semanal departamental</div>'
        f'<div class="pred-value">{confidence:.1f}<small> %</small></div>'
        '<div class="pred-label">Confianza del modelo en la categoría mostrada</div>'
        f'<div class="progress-head"><span>Puntuación de actividad alta</span><strong>{high:.1f}%</strong></div>'
        f'<div class="progress-track"><div class="progress-fill" style="width:{high:.1f}%;background:{color}"></div></div>'
        f'<div class="mini-kpi"><span class="value">{pred["casos_esperados"]:.0f}</span>'
        '<span class="caption">Casos semanales departamentales estimados<br>Estimación experimental, no cifra oficial</span></div>',
        unsafe_allow_html=True,
    )


def advice(preds,precip):
    h_d=preds.get("DENGUE",{}).get("nivel")=="Alto"
    h_m=preds.get("MALARIA",{}).get("nivel")=="Alto"
    lines=["Repelente registrado: reaplica según la etiqueta.",
           "Usa ropa que cubra brazos y piernas.",
           "Prefiere habitaciones con mallas o mosquiteros."]
    if h_m:lines.append("Consulta medicina del viajero sobre prevención de malaria.")
    elif h_d or (pd.notna(precip) and precip>=50):lines.append("Evita zonas con agua acumulada y protege tu alojamiento.")
    else:lines.append("Revisa avisos sanitarios oficiales antes de salir.")
    return lines[:4]


models=load_models()
if len(models)<2:
    st.warning("Faltan modelos entrenados. En la terminal ejecuta `python -m src.pipeline` y luego vuelve a abrir esta aplicación. El clima en vivo sí puede consultarse.")

section("Planifica tu viaje", "Selecciona el departamento y la fecha de tu próxima visita.", "01 / destino")
with st.container(border=True):
    left,right=st.columns([1.7,1],gap="large")
    with left:
        region=st.selectbox("Destino en Perú (departamento)",list(CAPITALS),index=list(CAPITALS).index("PIURA"),
                            format_func=lambda x:f"{x.title()} · {CAPITALS[x][0]}")
    with right:
        selected=st.date_input("Fecha de viaje",value=date.today(),min_value=date.today(),
                               max_value=date.today()+timedelta(days=15),format="DD/MM/YYYY")
    st.caption("El clima corresponde a la capital departamental y no representa todos los microclimas de la región. Pronóstico: hasta 16 días desde hoy.")

try:
    with st.spinner("Consultando pronóstico Open-Meteo..."):
        daily,day,preds=predict(region,selected,models)
except Exception as exc:
    st.error(f"No se pudo consultar Open-Meteo: {exc}")
    st.stop()

if day is None:
    st.info("No hay pronóstico para esta fecha. Elige una fecha dentro del horizonte disponible.")
    st.stop()

section("Actividad estimada", "Indicadores experimentales para el destino seleccionado.", "02 / modelos")
col1,col2=st.columns(2,gap="large")
with col1:
    with st.container(border=True):
        gauge("Dengue",preds.get("DENGUE"))
with col2:
    with st.container(border=True):
        gauge("Malaria",preds.get("MALARIA"))
st.markdown('<div class="notice"><strong>Nota metodológica.</strong> Los porcentajes son puntuaciones no calibradas del modelo para categorías de actividad histórica, NO probabilidades de contagiarte. No equivalen a alertas MINSA ni a un diagnóstico. La base epidemiológica termina en 2025.</div>', unsafe_allow_html=True)

section("Condiciones climáticas", "Pronóstico diario de la capital departamental.", "03 / clima")
a,b,c=st.columns(3,gap="large")
with a:
    weather_tile("Temperatura media", f"{day.temperature_2m_mean:.1f}" if pd.notna(day.temperature_2m_mean) else "Sin dato", "°C" if pd.notna(day.temperature_2m_mean) else "", "Media pronosticada", 120)
with b:
    weather_tile("Temperatura máxima", f"{day.temperature_2m_max:.1f}" if pd.notna(day.temperature_2m_max) else "Sin dato", "°C" if pd.notna(day.temperature_2m_max) else "", "Máxima pronosticada", 200)
with c:
    weather_tile("Probabilidad de lluvia", f"{day.precipitation_probability_max:.0f}" if pd.notna(day.precipitation_probability_max) else "Sin dato", "%" if pd.notna(day.precipitation_probability_max) else "", "Probabilidad máxima diaria", 280)
if pd.notna(day.precipitation_sum):
    st.caption(f"Precipitación pronosticada: {day.precipitation_sum:.1f} mm ese día · Fuente: Open-Meteo")

# Mantener consultas voluntarias para evitar saturar la API.
@st.cache_data(ttl=1800,show_spinner=False)
def all_regions(daystr,disease):
    rows=[]
    def one(name):
        try:
            _,_,pr=predict(name,pd.Timestamp(daystr).date(),models)
            return name,pr.get(disease)
        except Exception:
            return name,None
    with ThreadPoolExecutor(max_workers=5) as executor:
        futs=[executor.submit(one,reg) for reg in CAPITALS]
        for fut in as_completed(futs):
            reg,pred=fut.result()
            city,lat,lon=CAPITALS[reg]
            rows.append({"departamento":reg,"capital":city,"lat":lat,"lon":lon,
                         "nivel":pred["nivel"] if pred else "Sin datos",
                         "prob_alto":round(pred["prob_alto"]*100,1) if pred else None})
    return pd.DataFrame(rows)

section("Explora el mapa", "Señales por capital departamental, no por distrito.", "04 / territorio")
with st.container(border=True):
    st.markdown('<div class="card-head">Mapa de actividad estimada</div><div class="card-desc">Consulta los departamentos de forma voluntaria para actualizar la visualización.</div>',unsafe_allow_html=True)
    mc1,mc2=st.columns([1.7,1],gap="large")
    with mc1:
        kind=st.radio("Enfermedad del mapa",["DENGUE","MALARIA"],horizontal=True)
    with mc2:
        run_map=st.button("Consultar departamentos",type="secondary",use_container_width=True)
    if run_map:
        with st.spinner("Calculando estimaciones regionales. La primera carga requiere consultas de clima..."):
            st.session_state["map_data"]={"date":str(selected),"kind":kind,"frame":all_regions(str(selected),kind)}
    mapstate=st.session_state.get("map_data")
    if mapstate and mapstate["date"]==str(selected) and mapstate["kind"]==kind:
        m=mapstate["frame"]
        color_map={"Bajo":"#168c65","Moderado":"#e4af34","Alto":"#d64b45","Sin datos":"#96a7b4"}
        fig=px.scatter_map(m,lat="lat",lon="lon",color="nivel",
            hover_name="departamento",hover_data={"capital":True,"prob_alto":True,"lat":False,"lon":False},
            color_discrete_map=color_map,zoom=4.5,center={"lat":-9.3,"lon":-75.1},
            map_style="open-street-map",height=470)
        fig.update_traces(marker={"size":16})
        fig.update_layout(margin=dict(l=0,r=0,t=10,b=0),legend_title_text="Actividad estimada",
                          paper_bgcolor="rgba(0,0,0,0)",font=dict(family="Arial",color="#526047"))
        st.plotly_chart(fig,use_container_width=True)
        st.caption("Los puntos corresponden a capitales departamentales; no son polígonos ni mapas de riesgo a nivel distrital. Gris: sin cobertura/modelo.")
    else:
        st.info("Selecciona la enfermedad y pulsa 'Consultar departamentos' para visualizar las señales.")

section("Antes de viajar", "Medidas generales para reducir la exposición.", "05 / prevención")
with st.container(border=True):
    titles=["Protección personal", "Ropa adecuada", "Alojamiento protegido"]
    for i,tip in enumerate(advice(preds,day.precipitation_probability_max)):
        # El texto de advice se presenta sin iconos ni emojis.
        clean_tip=tip
        if i==3:
            if preds.get("MALARIA",{}).get("nivel")=="Alto":
                title="Prevención de malaria"
            elif preds.get("DENGUE",{}).get("nivel")=="Alto" or (pd.notna(day.precipitation_probability_max) and day.precipitation_probability_max>=50):
                title="Entorno seguro"
            else:
                title="Fuentes oficiales"
        else:
            title=titles[i]
        st.markdown(f'<div class="tip-row"><span class="tip-no">{i+1:02d}</span>'
                    f'<div><b>{escape(title)}</b><span>{escape(clean_tip)}</span></div></div>',
                    unsafe_allow_html=True)

section("Acerca de la metodología", "Limitaciones importantes de las estimaciones.", "06 / información")
with st.expander("Cómo interpretar estos resultados"):
    st.write("""Los modelos se entrenan con notificaciones de dengue y totales provinciales de malaria agregados por departamento, y cuatro semanas previas de clima de una capital. Se usan umbrales estadísticos internos calculados sobre el periodo de entrenamiento, no umbrales oficiales de brote. La probabilidad indica confianza del clasificador sobre una categoría de actividad y **no** la posibilidad de infección personal. El pronóstico meteorológico tampoco es una señal causal de dengue o malaria. La cobertura y las definiciones de semana epidemiológica de las fuentes necesitan validación antes de decisiones de salud pública. Para prevenir malaria, la quimioprofilaxis depende del itinerario específico y de una consulta médica. Revisar avisos actualizados del MINSA y de medicina del viajero.""")
