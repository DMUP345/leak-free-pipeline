"""Genera el dataset ficticio de Quincena (dos tablas relacionadas).

clientes.csv   -> una fila por cliente
prestamos.csv  -> una fila por préstamo (un cliente puede tener varios)

La variable objetivo `mora` vale 1 si el préstamo se pagó con más de 30 días de retraso.
El riesgo se construye con reglas no lineales a propósito (umbrales e interacciones),
que es justo el tipo de patrón que un gradient boosting sabe encontrar.
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
N_CLIENTES = 2500

ciudades = ["Bogotá", "Medellín", "Cali", "Barranquilla", "Bucaramanga"]
contratos = ["Indefinido", "Término fijo", "Prestación de servicios"]

clientes = pd.DataFrame({
    "cliente_id": [f"C{i:05d}" for i in range(1, N_CLIENTES + 1)],
    "edad": rng.integers(19, 62, N_CLIENTES),
    "ciudad": rng.choice(ciudades, N_CLIENTES, p=[0.38, 0.22, 0.16, 0.14, 0.10]),
    "tipo_contrato": rng.choice(contratos, N_CLIENTES, p=[0.45, 0.30, 0.25]),
})
# ingreso mensual en millones de COP, sesgado a la derecha
ingreso = rng.lognormal(mean=np.log(2.6), sigma=0.45, size=N_CLIENTES)
clientes["ingreso_mensual"] = np.clip(ingreso, 1.3, 14).round(2)
clientes["meses_empleo"] = np.clip(rng.gamma(2.0, 14, N_CLIENTES), 1, 180).astype(int)

# ---- préstamos ------------------------------------------------------------
filas = []
pid = 1
inicio = pd.Timestamp("2025-01-01")
for _, c in clientes.iterrows():
    n = rng.choice([1, 2, 3, 4, 5, 6], p=[0.28, 0.24, 0.18, 0.14, 0.10, 0.06])
    fechas = np.sort(rng.integers(0, 365, n))
    moras_previas = 0
    for k, d in enumerate(fechas):
        plazo = rng.choice([15, 30, 60], p=[0.35, 0.45, 0.20])
        # monto como fracción del ingreso: la mayoría pide poco, algunos piden mucho
        frac = np.clip(rng.beta(2.2, 5.5) * 0.9, 0.04, 0.85)
        monto = round(float(c.ingreso_mensual * frac), 2)
        cuota_ingreso = monto / c.ingreso_mensual

        # --- riesgo "verdadero" (el modelo no lo ve, solo ve las columnas) ---
        z = -2.75
        z += 1.35 * (cuota_ingreso > 0.32) + 0.9 * (cuota_ingreso > 0.5)
        z += 0.85 * (c.meses_empleo < 6)
        z += 0.55 * (c.tipo_contrato == "Prestación de servicios")
        z += 0.45 * (c.edad < 25)
        z += 0.30 * (plazo == 60)
        z += 1.10 * min(moras_previas, 2)
        z -= 0.35 * (k >= 2 and moras_previas == 0)       # buen historial ayuda
        # interacción: poca antigüedad Y cuota alta es peor que la suma
        z += 0.70 * ((c.meses_empleo < 12) and (cuota_ingreso > 0.32))
        z += rng.normal(0, 0.35)
        p = 1 / (1 + np.exp(-z))
        mora = int(rng.random() < p)

        filas.append({
            "prestamo_id": f"P{pid:06d}",
            "cliente_id": c.cliente_id,
            "fecha_desembolso": (inicio + pd.Timedelta(days=int(d))).date().isoformat(),
            "monto": monto,
            "plazo_dias": int(plazo),
            "mora": mora,
        })
        moras_previas += mora
        pid += 1

prestamos = pd.DataFrame(filas).sort_values("fecha_desembolso").reset_index(drop=True)

DATOS = Path(__file__).resolve().parent / "datasets" / "quincena"
DATOS.mkdir(parents=True, exist_ok=True)
clientes.to_csv(DATOS / "clientes.csv", index=False)
prestamos.to_csv(DATOS / "prestamos.csv", index=False)
print(clientes.shape, prestamos.shape, prestamos.mora.mean().round(3))
