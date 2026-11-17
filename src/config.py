"""Coordenadas aproximadas de CAPITALES departamentales, no centroides provinciales."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

# Etiqueta normalizada: (capital, lat, lon). Perú = 24 departamentos + Callao.
CAPITALS = {
    "AMAZONAS": ("Chachapoyas", -6.2317, -77.8690),
    "ANCASH": ("Huaraz", -9.5300, -77.5300),
    "APURIMAC": ("Abancay", -13.6370, -72.8810),
    "AREQUIPA": ("Arequipa", -16.4090, -71.5370),
    "AYACUCHO": ("Ayacucho", -13.1600, -74.2300),
    "CAJAMARCA": ("Cajamarca", -7.1638, -78.5003),
    "CALLAO": ("Callao", -12.0566, -77.1181),
    "CUSCO": ("Cusco", -13.5319, -71.9675),
    "HUANCAVELICA": ("Huancavelica", -12.7871, -74.9730),
    "HUANUCO": ("Huánuco", -9.9306, -76.2422),
    "ICA": ("Ica", -14.0678, -75.7286),
    "JUNIN": ("Huancayo", -12.0667, -75.2100),
    "LA LIBERTAD": ("Trujillo", -8.1116, -79.0288),
    "LAMBAYEQUE": ("Chiclayo", -6.7714, -79.8409),
    "LIMA": ("Lima", -12.0464, -77.0428),
    "LORETO": ("Iquitos", -3.7491, -73.2538),
    "MADRE DE DIOS": ("Puerto Maldonado", -12.5933, -69.1891),
    "MOQUEGUA": ("Moquegua", -17.1927, -70.9340),
    "PASCO": ("Cerro de Pasco", -10.6833, -76.2567),
    "PIURA": ("Piura", -5.1945, -80.6328),
    "PUNO": ("Puno", -15.8402, -70.0219),
    "SAN MARTIN": ("Moyobamba", -6.0342, -76.9716),
    "TACNA": ("Tacna", -18.0066, -70.2463),
    "TUMBES": ("Tumbes", -3.5669, -80.4515),
    "UCAYALI": ("Pucallpa", -8.3791, -74.5539),
}
