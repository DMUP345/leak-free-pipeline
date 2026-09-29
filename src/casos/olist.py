"""Olist: predecir si un pedido llegarÃ¡ tarde.

El objetivo `tarde` no puede ser una variable derivada: construir_tabla() valida que la
columna objetivo exista justo despuÃ©s de unir las tablas, antes de calcular las derivadas.
Por eso el paso 1 es preparaciÃ³n de datos y vive fuera del pipeline.

------------------------------------------------------------------------------------
PASO 1  Preparar los eventos       -> deja un CSV con el objetivo dentro
PASO 2  Declarar las Tabla y la Relacion
PASO 3  Definir el Experimento
PASO 4  Entrenar e interpretar
------------------------------------------------------------------------------------
"""
from pathlib import Path

import pandas as pd

from nucleo import (ConfigModelo, ConteoPrevio, CorteTemporal, DiferenciaDias,
                    Experimento, Objetivo, ParteDeFecha, Relacion, SumaPrevia, Tabla,
                    entrenar)

RAIZ = Path(__file__).resolve().parent.parent.parent
DATOS = Path(__file__).resolve().parent.parent / "datasets" / "olist"
PEDIDOS_CRUDOS = DATOS / "olist_orders_dataset.csv"
CLIENTES = DATOS / "olist_customers_dataset.csv"
#lo escribe el paso 1, asi que vive junto a los crudos de su propio caso
PEDIDOS_LISTOS = DATOS / "olist_pedidos_preparados.csv"
# =====================================================================================
# PASO 1 â€” preparar los eventos
# =====================================================================================
FECHAS = ["order_purchase_timestamp",
          "order_delivered_customer_date",
          "order_estimated_delivery_date"]

def preparar_eventos():
    """Deja PEDIDOS_LISTOS con la columna objetivo dentro, que es lo que el pipeline
    exige: construir_tabla() valida que el objetivo exista antes de las derivadas."""
    orders = pd.read_csv(PEDIDOS_CRUDOS, parse_dates=FECHAS)
    print("raw orders :", orders.shape)

    #a) fuera los censurados: sin fecha de entrega no se sabe si llegaron tarde.
    #   Filtrar ANTES de comparar: en pandas NaT > fecha da False, no nulo, asi que
    #   quedarian marcados como puntuales y diluirian la tasa en silencio.
    entregados = orders[orders.order_delivered_customer_date.notna()].copy()
    print("censurados descartados :", len(orders) - len(entregados))
    #b) el objetivo: 1 si la entrega real ocurrio despues de la prometida
    entregados["tarde"] = (entregados.order_delivered_customer_date
                           > entregados.order_estimated_delivery_date).astype(int)
    #c) index=False: tras filtrar, el indice tiene huecos y no significa nada
    entregados.to_csv(PEDIDOS_LISTOS, index=False)
    print("eventos listos :", len(entregados),
          "  tasa de retraso :", f"{entregados.tarde.mean():.2%}")


# =====================================================================================
# PASO 2 â€” las dos tablas y su relaciÃ³n
# =====================================================================================
eventos = Tabla(nombre="pedidos", ruta=str(PEDIDOS_LISTOS), llave="order_id",
                fechas=("order_purchase_timestamp", "order_estimated_delivery_date"))
entidades = Tabla(nombre="clientes", ruta=str(CLIENTES), llave="customer_id")
relacion = Relacion(eventos=eventos, entidades=entidades, llave="customer_id")

# =====================================================================================
# PASO 3 â€” el experimento
# =====================================================================================
experimento = Experimento(
    relacion=relacion,
    corte=CorteTemporal(
        "order_purchase_timestamp",
        inicio_validacion="2018-04-01",
        inicio_prueba="2018-06-01",
    ),
    objetivo=Objetivo("tarde"),
    numericas=["customer_zip_code_prefix"],
    categoricas=["customer_state"],
    derivadas=[
        ParteDeFecha("dia_semana", 
                     fecha="order_purchase_timestamp", 
                     parte="dayofweek"),
        ParteDeFecha("hora",       
                     fecha="order_purchase_timestamp", 
                     parte="hour"),

        DiferenciaDias("dias_prometidos",
                       desde="order_purchase_timestamp",
                       hasta="order_estimated_delivery_date"),
        ConteoPrevio("compras_previas",
                     llave="customer_unique_id",
                     fecha="order_purchase_timestamp"),
        SumaPrevia("retrasos_previos",
                   llave="customer_unique_id",
                   fecha="order_purchase_timestamp",
                   columna="tarde"),
    ],
    modelo=ConfigModelo(learning_rate=0.05, max_depth=3, min_samples_leaf=30, n_max=600),
)


# =====================================================================================
# PASO 4 â€” entrenar e interpretar
# =====================================================================================
#lo de arriba son definiciones baratas: `from config_olist import experimento` no
#entrena nada. Lo de aqui abajo solo corre al ejecutar el archivo.
if __name__ == "__main__":
    preparar_eventos()

    res = entrenar(experimento)
    print("mejor nÃºmero de Ã¡rboles:", res.mejor_n, " (Ã³ptimo exacto:", res.n_minimo, ")")
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in res.metricas.items()})

    # de dÃ³nde viene la seÃ±al
    cols = res.pipeline[:-1].get_feature_names_out()
    imp = (pd.Series(res.pipeline[-1].feature_importances_, index=cols)
             .sort_values(ascending=False))
    print("\nimportancia (%):")
    print((imp * 100).round(1).head(10).to_string())
    print("estados juntos:", f"{imp.filter(like='customer_state').sum():.1%}")

    # dÃ³nde tocÃ³ fondo la curva de validaciÃ³n: si mejor_n queda pegado a n_max,
    # el nÃºmero no significa nada y hay que subir el presupuesto
    c = res.curva_val
    print("\ncurva:", {n: round(c[n - 1], 5) for n in (100, 200, 300, 400, 500, 600)})

    # quÃ© se atrapa segÃºn cuÃ¡ntos pedidos se pueda revisar, medido sobre validaciÃ³n
    print("\numbrales (sobre validaciÃ³n):")
    print(res.umbrales().round(3).to_string())

    res.guardar(str(RAIZ / "modelo_olist.joblib"))




