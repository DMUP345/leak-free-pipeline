"""EL PUNTO DE PARTIDA DEL PROYECTO. Se conserva a propósito, sin corregir.

Esta es la primera versión: un script de arriba abajo, con el caso Quincena clavado en
cada línea. De aquí salió `pipeline_agnostico.py`, y el contraste entre los dos archivos
es la mitad de lo que este repositorio quiere mostrar.

Lo que tenía mal, y que el módulo agnóstico resolvió:

- El preprocesamiento vivía FUERA del pipeline (`StandardScaler` aplicado aparte), así que
  el escalado y el one-hot se ajustaban con todos los datos, incluido el futuro.
- Las dummies se construían sobre la tabla completa antes de partir en entrenamiento y
  prueba, lo que contradice el corte temporal que el resto del archivo respeta.
- `if len(df) == len(prestamos)` como "chequeo de duplicados": con `how="left"` y
  `validate="many_to_one"` esa igualdad es siempre cierta, así que no podía detectar nada.
- `cuota_ingreso` no es una cuota, es la razón monto/ingreso. El nombre engaña.
- El `groupby` se guardaba en `g` antes de escribir columnas nuevas en el DataFrame.
- El número de árboles se elegía por el mínimo exacto de la curva de validación, que es
  ruido: en estos datos, 173 le gana a 167 por 0,00017.

Ninguno de esos errores lanza una excepción. Todos producen un resultado plausible.
Ver el README para lo que se hizo con cada uno.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display


DATOS = Path(__file__).resolve().parent / "datasets" / "quincena"

clientes = pd.read_csv(DATOS / "clientes.csv")
prestamos = pd.read_csv(DATOS / "prestamos.csv", parse_dates=["fecha_desembolso"])

print("clientes : ", clientes.shape)
print("prestamos :", prestamos.shape)

print("clientes con almenos un prestamo : ", prestamos.cliente_id.nunique())
print("tasa de mora actual :", f"{prestamos.mora.mean():.1%}")
print(clientes.head(4))
print(prestamos.head(4))

#datos del cliente al prestamo
df = prestamos.merge(clientes, on="cliente_id", how="left", validate="many_to_one")
#variables nuevas que solo existen al cruzar las 2 tablas
df["cuota_ingreso"] = df["monto"]/df["ingreso_mensual"]
#mirar solo prestamos anteriores del cliente
df = df.sort_values(["cliente_id","fecha_desembolso"])
g=df.groupby("cliente_id")
df["prestamos_previos"]=g.cumcount()
df["moras_previas"] = g["mora"].cumsum()-df["mora"]

print("filas despues del merge: ", len(df))
if len(df) == len(prestamos):
    print("No hay duplicados")
else:
    print("estan duplicados")
print(df[["prestamo_id", "cliente_id", "monto", "ingreso_mensual", "cuota_ingreso", "prestamos_previos", "moras_previas", "mora"]].head(10))

#revisar solo datos enero a julio, no mirar el futuro

eda = df[df.fecha_desembolso < "2025-08-01"] 
cortes=[0, .15, .25, .32, .40, .50, 1]
etiquetas = ["hasta 15 %", "15-25 %", "25-32 %", "32-40 %", "40-50 %", "más de 50 %"]
por_tramo = (eda.assign(tramo=pd.cut(eda.cuota_ingreso, cortes, labels=etiquetas))
                .groupby("tramo", observed=True)
                .agg(prestamos=("mora", "size"), moras=("mora", "sum"), tasa_mora=("mora", "mean")))
por_tramo["mora_%"] = (por_tramo.tasa_mora * 100).round(1)
por_tramo["veces_el_promedio"] = (por_tramo.tasa_mora / eda.mora.mean()).round(2)
print(por_tramo.drop(columns="tasa_mora"))

#matriz de variables y corte por tiempo

variables = ["edad", "ingreso_mensual", "meses_empleo", "monto", "plazo_dias", "cuota_ingreso", "prestamos_previos", "moras_previas", "tipo_contrato", "ciudad"]
X=pd.get_dummies(df[variables], columns=["tipo_contrato", "ciudad"]).astype(float)
y=df["mora"]
fecha=df["fecha_desembolso"]

#corte por tiempo: entrenado con el pasado, se prueba con el futuro

ent = fecha < "2025-08-01"
val = (fecha >= "2025-08-01") & (fecha < "2025-10-01")
pru = fecha >= "2025-10-01"

X_ent, y_ent = X[ent], y[ent]
X_val, y_val = X[val], y[val]
X_pru, y_pru = X[pru], y[pru]

resumen = pd.DataFrame([
    {"conjunto": nombre,
     "desde": fecha[m].min().date(), "hasta": fecha[m].max().date(),
     "prestamos": int(m.sum()), "moras": int(y[m].sum()),
     "mora_%": round(y[m].mean() * 100, 1),
     "moras_previas_prom": round(df.loc[m, "moras_previas"].mean(), 2)}
    for nombre, m in [("entrenamiento", ent), ("validación", val), ("prueba", pru)]
]).set_index("conjunto")
display(resumen)
print(type(resumen))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, log_loss

base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
base.fit(X_ent, y_ent)
p_base = base.predict_proba(X_pru)[:, 1]

print(f"Regresión logística  AUC {roc_auc_score(y_pru, p_base):.3f}   "
      f"log-loss {log_loss(y_pru, p_base):.3f}")

#gradient boosting + curva de validacion

from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=600,     # de sobra; luego vemos dónde parar
    learning_rate=0.05,   # cada árbol corrige solo un 5 %
    max_depth=3,          # árboles pequeños: máximo 3 preguntas seguidas
    min_samples_leaf=30,  # cada hoja necesita al menos 30 préstamos
    random_state=42,
)
gb.fit(X_ent, y_ent)

# error en validación después de cada árbol
curva_val = [log_loss(y_val, p[:, 1]) for p in gb.staged_predict_proba(X_val)]
curva_ent = [log_loss(y_ent, p[:, 1]) for p in gb.staged_predict_proba(X_ent)]
mejor_n = int(np.argmin(curva_val)) + 1

print("mejor número de árboles:", mejor_n)
print(f"log-loss validación en ese punto: {curva_val[mejor_n - 1]:.4f}")
print(f"log-loss validación con los 600:  {curva_val[-1]:.4f}")

puntos = sorted({1, 25, 50, 100, mejor_n, 250, 400, 600})
tabla_curva = pd.DataFrame({
    "arboles": puntos,
    "logloss_ent": [curva_ent[n - 1] for n in puntos],
    "logloss_val": [curva_val[n - 1] for n in puntos],
}).set_index("arboles")
tabla_curva["brecha"] = tabla_curva.logloss_val - tabla_curva.logloss_ent
display(tabla_curva.round(4))

#modelo final y comparacion

modelo = GradientBoostingClassifier(
    n_estimators=mejor_n, learning_rate=0.05, max_depth=3,
    min_samples_leaf=30, random_state=42,
).fit(X_ent, y_ent)

p_gb = modelo.predict_proba(X_pru)[:, 1]
p_nada = np.full(len(y_pru), y_ent.mean())   # "a todos la misma tasa": el piso

comparacion = pd.DataFrame([
    {"modelo": "Sin modelo (misma tasa a todos)", "AUC": 0.5, "log_loss": log_loss(y_pru, p_nada)},
    {"modelo": "Regresión logística", "AUC": roc_auc_score(y_pru, p_base), "log_loss": log_loss(y_pru, p_base)},
    {"modelo": "Gradient boosting", "AUC": roc_auc_score(y_pru, p_gb), "log_loss": log_loss(y_pru, p_gb)},
]).set_index("modelo")
display(comparacion.round(3))

#importancia
importancia = (pd.Series(modelo.feature_importances_, index=X.columns)
                 .sort_values(ascending=False))
tabla_imp = pd.DataFrame({"importancia_%": importancia * 100,
                          "acumulada_%": importancia.cumsum() * 100}).round(1)
display(tabla_imp.head(10))
print("las 5 ciudades juntas:", f"{importancia.filter(like='ciudad_').sum():.1%}")

#apertura de los arboles

# ¿en qué valores cortan los árboles? contamos cada pregunta de los 173 árboles
preguntas = [(X.columns[f], round(t, 3))
             for arbol in modelo.estimators_[:, 0]
             for f, t in zip(arbol.tree_.feature, arbol.tree_.threshold) if f >= 0]
cortes_usados = (pd.DataFrame(preguntas, columns=["variable", "corte"])
                   .value_counts().rename("veces").reset_index())
cortes_clave = (cortes_usados[cortes_usados.variable.isin(["cuota_ingreso", "meses_empleo"])]
                  .head(8).reset_index(drop=True))
display(cortes_clave)

# dependencia parcial "a mano": a todos les pongo el mismo valor y promedio el riesgo
def dependencia(modelo, X, col, valores):
    filas = []
    for v in valores:
        X_mod = X.copy()
        X_mod[col] = v
        filas.append({col: v, "riesgo_prom_%": modelo.predict_proba(X_mod)[:, 1].mean() * 100})
    t = pd.DataFrame(filas).set_index(col)
    t["cambio_pp"] = t["riesgo_prom_%"].diff()
    return t.round(1)

dep_cuota = dependencia(modelo, X_ent, "cuota_ingreso", [.10, .20, .27, .30, .33, .40, .49, .52, .60])
dep_meses = dependencia(modelo, X_ent, "meses_empleo", [3, 5, 6, 10, 13, 30, 50, 60, 100])
display(dep_cuota)
display(dep_meses)
