"""Ejecutar todos los pasos desde la raíz: python -m src.pipeline"""
from .etl import run_etl
from .weather import download_history,build_weekly_weather
from .features import prepare_training
from .train import run_training

def main():
    run_etl()
    download_history()
    build_weekly_weather()
    prepare_training()
    run_training()
    print("LISTO: ejecuta streamlit run app.py")

if __name__=="__main__":
    main()
