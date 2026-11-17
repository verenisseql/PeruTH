"""Entrenamiento reproducible con partición CRONOLÓGICA por año epidemiológico.
Modelos para viajeros usan únicamente clima rezagado y estacionalidad;
modelos 'completos' agregan casos pasados para análisis epidemiológico.
"""
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, classification_report, accuracy_score
from .config import MODELS, REPORTS, PROCESSED
from .features import TRAVEL_FEATURES, FULL_FEATURES


def make_pipeline(features,kind):
    nums=[x for x in features if x!="departamento"]
    transformer=ColumnTransformer([
        ("dept",OneHotEncoder(handle_unknown="ignore"),["departamento"]),
        ("num",SimpleImputer(strategy="median"),nums),
    ])
    if kind=="classifier":
        model=RandomForestClassifier(n_estimators=100,min_samples_leaf=8,
                                     max_features="sqrt",random_state=42,n_jobs=-1,
                                     class_weight="balanced_subsample")
    else:
        model=RandomForestRegressor(n_estimators=100,min_samples_leaf=8,
                                    max_features="sqrt",random_state=42,n_jobs=-1)
    return Pipeline([("pre",transformer),("model",model)])


def calculate_thresholds(train):
    """Umbrales internos por departamento basados SOLO en el periodo de entrenamiento.
    No son umbrales epidemiológicos oficiales; 60 y 90 percentiles de casos positivos.
    """
    out={}
    positive_train=train.loc[train.casos>0]
    for dept in train.departamento.unique():
        x=positive_train.loc[positive_train.departamento.eq(dept),"casos"]
        if len(x)==0:
            out[dept] = [1,2]
        else:
            lo=max(1,int(np.ceil(x.quantile(0.60))))
            hi=max(lo+1,int(np.ceil(x.quantile(0.90))))
            out[dept]=[lo,hi]
    return out


def classes_for(df,thresholds):
    def classify(r):
        lo,hi=thresholds[r.departamento]
        if r.casos<=lo:return 0
        if r.casos<=hi:return 1
        return 2
    return df.apply(classify,axis=1).to_numpy(dtype=int)


def train_one(disease,mode,data):
    features=TRAVEL_FEATURES if mode=="travel" else FULL_FEATURES
    df=data.loc[data.enfermedad.eq(disease)].dropna(subset=features+["casos"]).copy()
    training=df.loc[df.ano_epi.between(2000,2017)].copy()
    validation=df.loc[df.ano_epi.between(2018,2021)].copy()
    testing=df.loc[df.ano_epi.between(2022,2025)].copy()

    # identificar departamentos presentes en entrenamiento
    departamentos_entrenados = set(training["departamento"].unique())

    # detectar departamentos que aparecen solo en años posteriores
    for nombre, tabla in [
        ("Validacion", validation),
        ("Prueba", testing)
    ]:
        nuevos = set(tabla["departamento"]) - departamentos_entrenados
        if nuevos:
            print(
                f"{disease}/{mode} - {nombre}: "
                f"departamentos sin entrenamiento: {sorted(nuevos)}"
            )

    # evaluar solamente departamentos conocidos por el modelo
    validation = validation[
        validation["departamento"].isin(departamentos_entrenados)
    ].copy()
    testing = testing[
        testing["departamento"].isin(departamentos_entrenados)
    ].copy()

    if len(training)<150:
        raise ValueError(f"Muy pocas filas para {disease}/{mode}: {len(training)}")
    thresholds=calculate_thresholds(training)
    clf=make_pipeline(features,"classifier")
    reg=make_pipeline(features,"regressor")
    clf.fit(training[features],classes_for(training,thresholds))
    reg.fit(training[features],np.log1p(training.casos))
    report={"enfermedad":disease,"modo":mode,"n_train":len(training),
            "n_validacion":len(validation),"n_test":len(testing),
            "train_hasta":2017,"validacion": "2018-2021", "test": "2022-2025",
            "modelo_probabilidad":"RandomForestClassifier.predict_proba; NO calibrada, no es riesgo individual",
            "definicion_riesgo":"Bajo: casos <= percentil 60 de positivos; Moderado: hasta percentil 90; Alto: por encima, por departamento (solo entrenamiento)."}
    for name,part in [("validation",validation),("test",testing)]:
        if len(part)==0:
            report[name]={"observaciones":0}
            continue
        actual=classes_for(part,thresholds)
        pred=clf.predict(part[features])
        cases=np.expm1(reg.predict(part[features])).clip(min=0)
        report[name]={
            "observaciones":len(part),
            "accuracy":float(accuracy_score(actual,pred)),
            "mae_casos":float(mean_absolute_error(part.casos,cases)),
            "reporte_clasificacion":classification_report(actual,pred,
                labels=[0,1,2],target_names=["Bajo","Moderado","Alto"],zero_division=0,output_dict=True)
        }
    payload={"classifier":clf,"regressor":reg,"thresholds":thresholds,
             "features":features,"mode":mode,"disease":disease,"metadata":report}
    MODELS.mkdir(parents=True,exist_ok=True)
    REPORTS.mkdir(parents=True,exist_ok=True)
    joblib.dump(payload,MODELS/f"{disease.lower()}_{mode}.joblib")
    (REPORTS/f"metricas_{disease.lower()}_{mode}.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"{disease}/{mode}: train={len(training)}, val={len(validation)}, test={len(testing)}")
    if "mae_casos" in report.get("test",{}):
        print("  TEST MAE casos:",round(report["test"]["mae_casos"],2),
              "| accuracy:",round(report["test"]["accuracy"],3))
    return report


def run_training():
    f=PROCESSED/"dataset_modelado.csv"
    if not f.exists():
        raise FileNotFoundError("Ejecuta antes python -m src.features")
    data=pd.read_csv(f,parse_dates=["inicio_semana"])
    all_reports=[]
    for disease in ["DENGUE","MALARIA"]:
        for mode in ["travel","full"]:
            all_reports.append(train_one(disease,mode,data))
    return all_reports


def inference(payload,features):
    """Retorna probabilidad MODELADA de cada categoría y casos esperados.
    NO probabilidad de infección de un turista.
    """
    features=features[payload["features"]]
    clf=payload["classifier"]
    p=clf.predict_proba(features)[0]
    probs={int(cl):float(val) for cl,val in zip(clf.classes_,p)}
    three=np.array([probs.get(k,0.0) for k in [0,1,2]],dtype=float)
    best=int(three.argmax())
    expected=max(0,float(np.expm1(payload["regressor"].predict(features)[0])))
    return {"nivel":["Bajo","Moderado","Alto"][best],
            "confianza_categoria":float(three[best]),
            "prob_alto":float(three[2]),
            "casos_esperados":expected,"probabilidades":three.tolist()}


if __name__=="__main__":
    run_training()
