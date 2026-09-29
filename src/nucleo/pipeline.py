"""Pipeline agnóstico para problemas de "entidades + eventos".

La idea: todo lo que depende del caso (nombres de tablas, columnas, fechas de corte,
variables derivadas, parámetros) vive en objetos de configuración. El código de abajo
no menciona ni una sola columna de Quincena.

Piezas, en el orden en que se usan:
    Tabla, Relacion         -> qué archivos se cargan y cómo se unen
    Variable (interfaz)     -> Razon, ConteoPrevio, SumaPrevia: variables nuevas
    CorteTemporal           -> entrenamiento / validación / prueba por fecha
    Objetivo, ConfigModelo  -> qué se predice y con qué perillas
    Experimento             -> junta todo lo anterior y lo valida
    entrenar(exp)           -> devuelve un Resultado
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance



@dataclass(frozen=True)
class Tabla:
    """Un archivo CSV con su llave y sus columnas de fecha."""
    nombre: str
    ruta: str
    llave: str
    fechas: tuple[str, ...] = ()

    def cargar(self) -> pd.DataFrame:
        df = pd.read_csv(self.ruta, parse_dates=list(self.fechas))
        if self.llave not in df.columns:
            raise KeyError(f"{self.nombre}: no existe la llave '{self.llave}'")
        return df


@dataclass(frozen=True)
class Relacion:
    """Relación uno a muchos: cada evento pertenece a una entidad."""
    eventos: Tabla      # lado "muchos" (préstamos, compras, sesiones...)
    entidades: Tabla    # lado "uno"   (clientes, usuarios, tiendas...)
    llave: str

    def unir(self) -> pd.DataFrame:
        ev, en = self.eventos.cargar(), self.entidades.cargar()
        repetidas = en[self.llave].duplicated().sum()
        if repetidas:
            raise ValueError(f"{self.entidades.nombre}: {repetidas} llaves repetidas")
        return ev.merge(en, on=self.llave, how="left", validate="many_to_one")


@runtime_checkable
class Variable(Protocol):
    """Cualquier objeto con un nombre y un método calcular(df) -> Series sirve."""
    nombre: str

    def calcular(self, df: pd.DataFrame) -> pd.Series: ...


@dataclass(frozen=True)
class Razon:
    nombre: str 
    numerador: str
    denominador: str

    def calcular(self, df: pd.DataFrame) -> pd.Series:
        razon = df[self.numerador] / df[self.denominador]
        #un denominador en cero da inf, que sklearn rechaza: lo dejamos nulo
        #para que lo rellene el imputador del ColumnTransformer
        return razon.replace([np.inf, -np.inf], np.nan)


@dataclass(frozen=True)
class ConteoPrevio:
    """Cuántos eventos tuvo la misma entidad ANTES de este."""
    nombre: str
    llave: str
    fecha: str

    def calcular(self, df: pd.DataFrame) -> pd.Series:
        orden = df.sort_values([self.llave, self.fecha], kind="mergesort")
        return orden.groupby(self.llave).cumcount().reindex(df.index)


@dataclass(frozen=True)
class SumaPrevia:
    """Suma de una columna en los eventos ANTERIORES de la misma entidad."""
    nombre: str
    llave: str
    fecha: str
    columna: str

    def calcular(self, df: pd.DataFrame) -> pd.Series:
        orden = df.sort_values([self.llave, self.fecha], kind="mergesort")
        acumulado = orden.groupby(self.llave)[self.columna].cumsum() - orden[self.columna]
        return acumulado.reindex(df.index)


@dataclass(frozen=True)
class CorteTemporal:
    columna: str
    inicio_validacion: str
    inicio_prueba: str

    def __post_init__(self):
        #pd.Timestamp ya revienta si la fecha no se puede leer
        if pd.Timestamp(self.inicio_validacion) >= pd.Timestamp(self.inicio_prueba):
            raise ValueError("inicio_validacion debe ser anterior a inicio_prueba")

    def partir(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        if self.columna not in df.columns:
            raise KeyError(f"no existe la columna de fecha '{self.columna}'")
        f = df[self.columna]
        if not pd.api.types.is_datetime64_any_dtype(f):
            raise TypeError(f"'{self.columna}' no es de tipo fecha: "
                            f"declarala en Tabla(..., fechas=('{self.columna}',))")
        partes = {
            "ent": f < self.inicio_validacion,
            "val": (f >= self.inicio_validacion) & (f < self.inicio_prueba),
            "pru": f >= self.inicio_prueba,
        }
        #sin esto el error sale mucho despues, dentro del fit, y no dice cual parte fallo
        vacias = [nombre for nombre, m in partes.items() if not m.any()]
        if vacias:
            raise ValueError(
                f"particiones vacias {vacias}: los datos van de {f.min().date()} a "
                f"{f.max().date()}, pero los cortes son {self.inicio_validacion} y "
                f"{self.inicio_prueba}")
        return partes


@dataclass(frozen=True)
class Objetivo:
    columna: str


@dataclass(frozen=True)
class ConfigModelo:
    learning_rate: float = 0.05
    max_depth: int = 3
    min_samples_leaf: int = 30
    n_max: int = 600
    random_state: int = 42
    #con qué se rellenan los nulos de las numéricas: median, mean o constant
    imputacion: str = "median"
    #cuántos errores estándar de holgura sobre el mínimo de la curva; 0 = mínimo exacto
    tolerancia_se: float = 1.0
    #"exacto" = GradientBoosting, "histograma" = HistGradientBoosting (unas 100x mas rapido)
    motor: str = "exacto"
    #"log_loss" penaliza la calibracion; "auc" solo mira el orden y sobrevive a que
    #la tasa base cambie entre particiones
    criterio: str = "log_loss"
    #holgura para el criterio "auc", en puntos de AUC: no hay error estandar emparejado
    #porque el AUC no se descompone por fila
    tolerancia_auc: float = 0.002
    #0 silencioso, 1 progreso con tiempo restante, 2 cada arbol. Solo cubre los dos
    #ajustes: las pasadas de la curva son silenciosas
    verbose: int = 0

    def __post_init__(self):
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate debe estar entre 0 y 1")
        if self.max_depth < 1 or self.min_samples_leaf < 1:
            raise ValueError("max_depth y min_samples_leaf deben ser positivos")
        if self.imputacion not in ("median", "mean", "most_frequent", "constant"):
            raise ValueError(f"imputacion '{self.imputacion}' no es una estrategia valida")
        if self.tolerancia_se < 0:
            raise ValueError("tolerancia_se no puede ser negativa")
        if self.motor not in ("exacto", "histograma"):
            raise ValueError(f"motor '{self.motor}' no existe: usa 'exacto' u 'histograma'")
        if self.criterio not in ("log_loss", "auc"):
            raise ValueError(f"criterio '{self.criterio}' no existe: usa 'log_loss' o 'auc'")


@dataclass
class Experimento:
    relacion: Relacion
    corte: CorteTemporal
    objetivo: Objetivo
    numericas: list[str]
    categoricas: list[str]
    derivadas: list[Variable] = field(default_factory=list)
    modelo: ConfigModelo = field(default_factory=ConfigModelo)

    def __post_init__(self):
        nombres = self.numericas + [v.nombre for v in self.derivadas] + self.categoricas
        if len(nombres) != len(set(nombres)):
            raise ValueError("hay variables con el nombre repetido")
        if self.objetivo.columna in nombres:
            raise ValueError("el objetivo no puede ser también una variable de entrada")
        for v in self.derivadas:
            if not isinstance(v, Variable):
                raise TypeError(f"{v!r} no cumple la interfaz Variable")

    @property
    def columnas_numericas(self) -> list[str]:
        return self.numericas + [v.nombre for v in self.derivadas]


@dataclass
class Resultado:
    mejor_n: int
    curva_val: list[float]
    metricas: dict[str, float]
    pipeline: Pipeline
    #dónde estaba el mínimo exacto de la curva, para ver cuánto recortó la tolerancia
    n_minimo: int = 0
    #probabilidades del modelo final sobre validación, para elegir el umbral sin tocar prueba
    p_val: np.ndarray | None = None
    y_val: np.ndarray | None = None

    def guardar(self, ruta: str) -> None:
        joblib.dump(self.pipeline, ruta)

    def umbrales(self, fracciones=(0.05, 0.10, 0.15, 0.20, 0.30)) -> pd.DataFrame:
        """Qué se atrapa revisando cada fracción, medido sobre validación."""
        if self.p_val is None:
            raise ValueError("este Resultado no guardó las probabilidades de validación")
        return tabla_de_umbrales(self.y_val, self.p_val, fracciones)


def construir_tabla(exp: Experimento) -> pd.DataFrame:
    df = exp.relacion.unir()
    #antes de calcular nada: un typo en la config debe avisar aqui, no dentro del fit
    pedidas = set(exp.numericas) | set(exp.categoricas) | {exp.objetivo.columna}
    faltan = sorted(pedidas - set(df.columns))
    if faltan:
        raise KeyError(f"columnas que no existen tras unir las tablas: {faltan}. "
                       f"Disponibles: {sorted(df.columns)}")
    for v in exp.derivadas:
        df[v.nombre] = v.calcular(df)
    return df


def _perdida_por_fila(y_true: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Log-loss de cada fila por separado; su promedio es el log_loss de sklearn."""
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def elegir_n(curva: list[float], y_val, modelo, X_val, cfg) -> tuple[int, int]:
    """Devuelve (n elegido, n del punto óptimo exacto).

    Con criterio "log_loss" el óptimo es el mínimo de la curva; con "auc", el máximo.
    """
    curva = np.asarray(curva)
    if cfg.criterio == "auc":
        #el AUC mejora hacia arriba, y no se descompone por fila: la holgura es un
        #numero explicito de puntos de AUC, no un error estandar emparejado
        n_mejor = int(curva.argmax()) + 1
        if cfg.tolerancia_auc <= 0:
            return n_mejor, n_mejor
        umbral = curva[n_mejor - 1] - cfg.tolerancia_auc
        return int(np.argmax(curva >= umbral)) + 1, n_mejor

    n_min = int(curva.argmin()) + 1
    if cfg.tolerancia_se <= 0:
        return n_min, n_min

    y = np.asarray(y_val, dtype=float)
    p_min = next(p for i, p in enumerate(modelo.staged_predict_proba(X_val), start=1)
                 if i == n_min)
    perdida_min = _perdida_por_fila(y, p_min[:, 1])

    for n, p in enumerate(modelo.staged_predict_proba(X_val), start=1):
        diferencia = _perdida_por_fila(y, p[:, 1]) - perdida_min
        se = diferencia.std(ddof=1) / np.sqrt(len(diferencia))
        #se == 0 solo si n es el mínimo mismo, y ahí la diferencia también es 0
        if diferencia.mean() <= cfg.tolerancia_se * se:
            return n, n_min
    return n_min, n_min


def construir_pipeline(exp: Experimento, n_arboles: int) -> Pipeline:
    m = exp.modelo
    #la mediana se calcula solo con entrenamiento, porque vive dentro del Pipeline
    preparar = ColumnTransformer([
        ("num", SimpleImputer(strategy=m.imputacion), exp.columnas_numericas),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), exp.categoricas),
    ])
    if m.motor == "histograma":
        modelo = HistGradientBoostingClassifier(
            max_iter=n_arboles,           # aqui se llama max_iter, no n_estimators
            learning_rate=m.learning_rate, max_depth=m.max_depth,
            min_samples_leaf=m.min_samples_leaf, random_state=m.random_state,
            early_stopping=False,         # el corte lo decide elegir_n, no el modelo
            verbose=m.verbose,
        )
    else:
        modelo = GradientBoostingClassifier(
            n_estimators=n_arboles, learning_rate=m.learning_rate, max_depth=m.max_depth,
            min_samples_leaf=m.min_samples_leaf, random_state=m.random_state,
            verbose=m.verbose,
        )
    return Pipeline([("preparar", preparar), ("modelo", modelo)])

def tabla_de_umbrales(y_true, p, fracciones=(0.05, 0.10, 0.15, 0.20, 0.30)) -> pd.DataFrame:
    """Qué se atrapa al revisar cada fracción de los casos con mayor puntaje.

    La fracción es la capacidad: cuántos casos se pueden atender. El umbral que
    devuelve es el de ESTE lote y no debe fijarse para siempre — las probabilidades
    derivan con el tiempo aunque el orden se mantenga, así que un umbral congelado
    termina marcando muchos más (o muchos menos) casos de los previstos.
    """
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    positivos = y_true.sum()
    if positivos == 0:
        raise ValueError("no hay casos positivos: la tabla de umbrales no significa nada")
    tasa_base = positivos / len(y_true)
    orden = np.sort(p)[::-1]
    filas = []
    for f in fracciones:
        k = max(1, int(round(f * len(y_true))))
        umbral = orden[k - 1]
        #con empates pueden salir mas marcados que k, y se reporta lo que de verdad pasa
        marcados = p >= umbral
        atrapados = float(y_true[marcados].sum())
        filas.append({
            "fraccion": f,
            "umbral": float(umbral),
            "revisados": int(marcados.sum()),
            "atrapados": int(atrapados),
            "precision": atrapados / marcados.sum(),
            "cobertura": atrapados / positivos,
            "lift": (atrapados / marcados.sum()) / tasa_base,
        })
    return pd.DataFrame(filas).set_index("fraccion")


#4
def entrenar(exp: Experimento) -> Resultado:
    df = construir_tabla(exp)
    partes = exp.corte.partir(df)
    X = df[exp.columnas_numericas + exp.categoricas]
    y = df[exp.objetivo.columna]
    ent, val, pru = partes["ent"], partes["val"], partes["pru"]

    #una sola clase rompe el fit, el log_loss y el AUC, cada uno con un error distinto
    for nombre, m in (("ent", ent), ("val", val), ("pru", pru)):
        if y[m].nunique() < 2:
            raise ValueError(
                f"la particion '{nombre}' tiene una sola clase de "
                f"'{exp.objetivo.columna}' en sus {int(m.sum())} filas: "
                f"no se puede entrenar ni evaluar")

    # a) muchos árboles de sobra para ver dónde parar
    grande = construir_pipeline(exp, exp.modelo.n_max).fit(X[ent], y[ent])
    X_val = grande[:-1].transform(X[val])            # todos los pasos menos el modelo
    #con "auc" la curva sube en vez de bajar: elegir_n sabe interpretarla segun el criterio
    if exp.modelo.criterio == "auc":
        curva = [roc_auc_score(y[val], p[:, 1]) for p in grande[-1].staged_predict_proba(X_val)]
    else:
        curva = [log_loss(y[val], p[:, 1]) for p in grande[-1].staged_predict_proba(X_val)]
    mejor_n, n_minimo = elegir_n(curva, y[val], grande[-1], X_val, exp.modelo)

    # b) modelo final y una sola evaluación en prueba
    final = construir_pipeline(exp, mejor_n).fit(X[ent], y[ent])
    p = final.predict_proba(X[pru])[:, 1]
    metricas = {"auc": roc_auc_score(y[pru], p), "log_loss": log_loss(y[pru], p),
                "n_ent": int(ent.sum()), "n_val": int(val.sum()), "n_pru": int(pru.sum())}
    #el umbral se elige con validacion, nunca con prueba: por eso se guardan estas dos
    return Resultado(mejor_n=mejor_n, curva_val=curva, metricas=metricas, pipeline=final,
                     n_minimo=n_minimo,
                     p_val=final.predict_proba(X[val])[:, 1],
                     y_val=np.asarray(y[val], dtype=float))


#olist Config
@dataclass(frozen=True)
class DiferenciaDias:
    """Días entre dos fechas de la misma fila."""
    nombre: str
    desde: str
    hasta: str

    def calcular(self, df: pd.DataFrame) -> pd.Series:
        #restar fechas da un timedelta64, que sklearn no acepta:
        #.dt.days extrae la parte entera de dias y deja un numero
        return (df[self.hasta] - df[self.desde]).dt.days


@dataclass(frozen=True)
class ParteDeFecha:
    """Un componente de una fecha: dia de la semana, hora, dia del mes."""
    nombre: str
    fecha: str
    parte: str

    def calcular(self, df: pd.DataFrame) -> pd.Series:
        #getattr sobre el accesor .dt deja que la parte sea configurable sin ifs
        return getattr(df[self.fecha].dt, self.parte)




def importancias(res: Resultado, exp: Experimento, n_muestra: int = 50_000,
                 repeticiones: int = 3) -> pd.DataFrame:
    """Importancia de cada variable, por el metodo que el motor permita.

    Los dos metodos miden cosas distintas y no se comparan entre si, asi que la
    columna `metodo` dice cual salio:

    - "impureza": cuanta impureza redujo cada variable al construir los arboles. Es
      gratis porque se acumula al entrenar, pero solo la lleva el motor exacto, y
      viene en unidades arbitrarias que suman 1.
    - "permutacion": cuanto cae el AUC al barajar esa columna. Funciona con cualquier
      motor y se mide sobre VALIDACION, nunca sobre prueba. Cuesta una prediccion por
      variable y repeticion, de ahi la muestra.

    Ademas nombran cosas distintas: la permutacion da las columnas ORIGINALES, la
    impureza las expandidas por el one-hot, que hay que sumar para leer una
    categorica completa.
    """
    modelo = res.pipeline[-1]

    if hasattr(modelo, "feature_importances_"):
        s = pd.Series(modelo.feature_importances_,
                      index=res.pipeline[:-1].get_feature_names_out())
        metodo = "impureza"
    else:
        df = construir_tabla(exp)
        val = exp.corte.partir(df)["val"]
        X = df[exp.columnas_numericas + exp.categoricas]
        y = df[exp.objetivo.columna]
        filas = df[val]
        if n_muestra and len(filas) > n_muestra:
            filas = filas.sample(n=n_muestra, random_state=exp.modelo.random_state)
        imp = permutation_importance(
            res.pipeline, X.loc[filas.index], y.loc[filas.index],
            scoring="roc_auc",          # la exactitud es inutil con clases desbalanceadas
            n_repeats=repeticiones, random_state=exp.modelo.random_state, n_jobs=-1)
        s = pd.Series(imp.importances_mean, index=X.columns)
        metodo = "permutacion"

    return (pd.DataFrame({"importancia": s, "metodo": metodo})
              .sort_values("importancia", ascending=False))
