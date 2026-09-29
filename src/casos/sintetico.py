"""El mismo pipeline sobre un dominio que no tiene nada que ver con prÃ©stamos.

Genera un dataset de juguete de comercio electrÃ³nico (usuarios + compras, objetivo
`devolucion`) y lo entrena con el mismo `entrenar()` que usa Quincena. Si esto corre,
el pipeline es agnÃ³stico de verdad: ninguna columna se repite entre los dos casos.

La seÃ±al es no lineal a propÃ³sito, igual que en generar_datos.py, para que el AUC
quede claramente por encima de 0.5 y la corrida sirva como prueba y no solo como demo.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from nucleo import (ConfigModelo, ConteoPrevio, CorteTemporal, Experimento, Objetivo,
                    Razon, Relacion, SumaPrevia, Tabla, entrenar)

DATOS = Path(__file__).resolve().parent.parent / "datasets" / "ejemplo"
N_USUARIOS = 1200


def generar_datos():
    """Escribe usuarios.csv y compras.csv de juguete. Semilla fija: mismas filas en
    cada corrida, asÃ­ el resultado es reproducible."""
    rng = np.random.default_rng(11)
    planes = ["Free", "Plus", "Premium"]
    paises = ["Colombia", "MÃ©xico", "Chile", "PerÃº"]

    usuarios = pd.DataFrame({
        "usuario_id": [f"U{i:05d}" for i in range(1, N_USUARIOS + 1)],
        "plan": rng.choice(planes, N_USUARIOS, p=[0.55, 0.30, 0.15]),
        "pais": rng.choice(paises, N_USUARIOS, p=[0.45, 0.28, 0.15, 0.12]),
    })
    # gasto habitual mensual, sesgado a la derecha
    usuarios["gasto_promedio"] = np.clip(
        rng.lognormal(mean=np.log(180), sigma=0.5, size=N_USUARIOS), 40, 2000).round(2)

    # ---- compras ----------------------------------------------------------
    filas = []
    cid = 1
    inicio = pd.Timestamp("2025-01-01")
    for _, u in usuarios.iterrows():
        n = rng.choice([1, 2, 3, 4, 5, 6, 7],
                       p=[0.22, 0.21, 0.18, 0.15, 0.11, 0.08, 0.05])
        fechas = np.sort(rng.integers(0, 365, n))
        devoluciones_previas = 0
        for k, d in enumerate(fechas):
            express = int(rng.random() < 0.35)
            valor = round(float(u.gasto_promedio
                                * np.clip(rng.beta(2.0, 3.0) * 2.4, 0.05, 3.0)), 2)
            vs_habitual = valor / u.gasto_promedio

            # --- riesgo "verdadero" de devoluciÃ³n (el modelo solo ve las columnas) ---
            z = -2.40
            z += 1.20 * (vs_habitual > 1.0) + 0.85 * (vs_habitual > 1.8)
            z += 0.60 * (u.plan == "Free")
            z += 0.40 * (u.pais == "PerÃº")
            z -= 0.35 * express                                 # llega rÃ¡pido, se devuelve menos
            z += 1.05 * min(devoluciones_previas, 2)
            z -= 0.30 * (k >= 2 and devoluciones_previas == 0)  # buen historial ayuda
            # interacciÃ³n: compra grande Y plan Free es peor que la suma
            z += 0.75 * ((u.plan == "Free") and (vs_habitual > 1.0))
            z += rng.normal(0, 0.35)
            devolucion = int(rng.random() < 1 / (1 + np.exp(-z)))

            filas.append({
                "compra_id": f"K{cid:06d}",
                "usuario_id": u.usuario_id,
                "fecha": (inicio + pd.Timedelta(days=int(d))).date().isoformat(),
                "valor": valor,
                "envio_express": express,
                "devolucion": devolucion,
            })
            devoluciones_previas += devolucion
            cid += 1

    compras = pd.DataFrame(filas).sort_values("fecha").reset_index(drop=True)

    DATOS.mkdir(parents=True, exist_ok=True)
    usuarios.to_csv(DATOS / "usuarios.csv", index=False)
    compras.to_csv(DATOS / "compras.csv", index=False)
    print("usuarios:", usuarios.shape, " compras:", compras.shape,
          " tasa de devolucion:", f"{compras.devolucion.mean():.1%}")


# ---- el mismo pipeline, otra configuraciÃ³n --------------------------------
exp = Experimento(
    relacion=Relacion(
        eventos=Tabla("compras", str(DATOS / "compras.csv"), llave="compra_id",
                      fechas=("fecha",)),
        entidades=Tabla("usuarios", str(DATOS / "usuarios.csv"), llave="usuario_id"),
        llave="usuario_id",
    ),
    corte=CorteTemporal("fecha", inicio_validacion="2025-08-01", inicio_prueba="2025-10-01"),
    objetivo=Objetivo("devolucion"),
    numericas=["valor", "envio_express", "gasto_promedio"],
    categoricas=["plan", "pais"],
    derivadas=[
        Razon("valor_vs_habitual", numerador="valor", denominador="gasto_promedio"),
        ConteoPrevio("compras_previas", llave="usuario_id", fecha="fecha"),
        SumaPrevia("devoluciones_previas", llave="usuario_id", fecha="fecha",
                   columna="devolucion"),
    ],
    modelo=ConfigModelo(learning_rate=0.05, max_depth=3, min_samples_leaf=20),
)

if __name__ == "__main__":
    generar_datos()
    res = entrenar(exp)
    print("mejor nÃºmero de Ã¡rboles:", res.mejor_n)
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in res.metricas.items()})
