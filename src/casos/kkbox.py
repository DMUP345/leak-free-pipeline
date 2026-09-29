"""KKBox: predecir si una suscripciÃ³n se va a renovar.  >>> ESQUELETO PARA COMPLETAR <<<

KKBox fue un servicio de streaming de mÃºsica. Cada usuario paga una suscripciÃ³n y la
renueva (o no) cada cierto nÃºmero de dÃ­as. Este es el primer dataset del proyecto donde
una misma entidad aparece MUCHAS veces: en Olist el 96,8 % de los clientes comprÃ³ una
sola vez y `ConteoPrevio` y `SumaPrevia` no tuvieron nada que contar. AquÃ­ un usuario con
dos aÃ±os de suscripciÃ³n mensual tiene mÃ¡s de veinte transacciones.

Columnas reales de los archivos:

    members_v3.csv    msno Â· city Â· bd Â· gender Â· registered_via Â· registration_init_time
    transactions.csv  msno Â· payment_method_id Â· payment_plan_days Â· plan_list_price
                      actual_amount_paid Â· is_auto_renew Â· transaction_date
                      membership_expire_date Â· is_cancel
    train_v2.csv      msno Â· is_churn        (objetivo POR USUARIO, no por evento)

------------------------------------------------------------------------------------
PASO 1  Construir el objetivo a nivel de evento  -> deja un CSV listo para modelar
PASO 2  Declarar las Tabla y la Relacion
PASO 3  Definir el Experimento
PASO 4  Entrenar e interpretar
------------------------------------------------------------------------------------
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from nucleo import (ConfigModelo, ConteoPrevio, CorteTemporal, DiferenciaDias,
                    Experimento, Objetivo, ParteDeFecha, Razon, Relacion, SumaPrevia,
                    Tabla, entrenar, importancias)

RAIZ = Path(__file__).resolve().parent.parent.parent
DATOS = Path(__file__).resolve().parent.parent / "datasets" / "KKbox"

MIEMBROS = DATOS / "members_v3.csv"
TRANSACCIONES = DATOS / "transactions.csv"
ETIQUETAS_OFICIALES = DATOS / "train_v2.csv"
EVENTOS_LISTOS = DATOS / "kkbox_eventos_preparados.csv"
N_USUARIOS = 200_000
rng = np.random.default_rng(7)  


# =====================================================================================
# PASO 1 â€” construir el objetivo a nivel de evento
# =====================================================================================
FECHAS_TX = ["transaction_date", "membership_expire_date"]
FIN_VENTANA = pd.Timestamp("2016-12-01")


def preparar_eventos():
    """Deja EVENTOS_LISTOS con la columna `abandona` dentro.

    El is_churn oficial viene POR USUARIO, asi que no sirve: el pipeline necesita una
    etiqueta por evento. Se construye mirando si el mismo msno tiene otra transaccion
    dentro de los 30 dias posteriores al vencimiento de esta.
    """
    rng = np.random.default_rng(7)

    #se muestrea por USUARIO, nunca por fila: tomar filas al azar partiria los
    #historiales y destruiria justo lo que ConteoPrevio y SumaPrevia miden
    candidatos = pd.read_csv(ETIQUETAS_OFICIALES)["msno"].unique()
    muestra = set(rng.choice(candidatos, size=N_USUARIOS, replace=False))

    #el archivo son 1,65 GB: se recorre por trozos y solo se guarda la muestra
    trozos = [t[t.msno.isin(muestra)]
              for t in pd.read_csv(TRANSACCIONES, chunksize=2_000_000)]
    tx = pd.concat(trozos, ignore_index=True)

    #las fechas vienen como enteros 20150930. Sin format, to_datetime las lee como
    #nanosegundos desde 1970 y devuelve fechas validas pero absurdas, sin avisar
    for col in FECHAS_TX:
        tx[col] = pd.to_datetime(tx[col], format="%Y%m%d")
    print("transacciones de la muestra:", tx.shape)

    FIN_DATOS = tx["transaction_date"].max()      # deberia dar 2017-02-28

    #fuera el centinela de 1970 (1.776 filas): un vencimiento imposible nunca tendria
    #renovacion dentro de 30 dias y quedaria marcado como abandono falso
    tx = tx[tx.membership_expire_date != pd.Timestamp("1970-01-01")].copy()

    #shift trabaja sobre el orden de las FILAS, no sobre las fechas
    tx = tx.sort_values(["msno", "transaction_date"], kind="mergesort")
    siguiente = tx.groupby("msno")["transaction_date"].shift(-1)
    limite = tx["membership_expire_date"] + pd.Timedelta(days=30)
    renovo = siguiente.notna() & (siguiente <= limite)
    tx["abandona"] = (~renovo).astype(int)

    #dos filtros distintos, por dos razones distintas
    etiquetable = limite <= FIN_DATOS                      # se puede saber el resultado
    sin_sesgo = tx["transaction_date"] < FIN_VENTANA       # la composicion es comparable
    #el segundo hace falta porque la censura NO es uniforme: pasa del 1 % en noviembre
    #al 22,8 % en diciembre y al 98,8 % en enero, y descarta sobre todo planes largos
    observable = etiquetable & sin_sesgo

    ev = tx[observable].reset_index(drop=True)
    ev["evento_id"] = ev.index          # msno se repite: hace falta una llave por evento
    ev.to_csv(EVENTOS_LISTOS, index=False)

    print("fin de los datos        :", FIN_DATOS.date())
    print("eventos etiquetados     :", len(ev))
    print("descartados por censura :", int((~etiquetable).sum()))
    print("descartados por ventana :", int((etiquetable & ~sin_sesgo).sum()))
    print("tasa de abandono        :", f"{ev.abandona.mean():.2%}")


# =====================================================================================
# PASO 2 â€” las dos tablas y su relaciÃ³n
# =====================================================================================
eventos = Tabla(nombre="transacciones",
                ruta=str(EVENTOS_LISTOS),
                llave="evento_id",
                fechas=("transaction_date", "membership_expire_date"))
entidades = Tabla(nombre="miembros",
                  ruta=str(MIEMBROS),
                  llave="msno",
                  fechas=("registration_init_time",))
relacion = Relacion(eventos=eventos, entidades=entidades, llave="msno")

# =====================================================================================
# PASO 3 â€” el experimento
# =====================================================================================
#tasas base por particion: 3,63 % entrenamiento / 0,78 % validacion / 1,31 % prueba.
#Ese desplazamiento es la razon de criterio="auc": con log_loss, elegir_n se detiene
#en el primer arbol porque cualquier aprendizaje empeora el nivel sobre validacion.
exp = Experimento(
    relacion= relacion, 
    corte=CorteTemporal(
        "transaction_date",
        inicio_validacion="2016-06-01",
        inicio_prueba= "2016-09-01",

    ),
    objetivo= Objetivo("abandona"),
    numericas=["payment_plan_days", 
               "plan_list_price", 
               "actual_amount_paid",
               "is_cancel"],
    categoricas=["payment_method_id", 
                 "registered_via"],
    derivadas=[
        ParteDeFecha("dia_semana", 
                     fecha="transaction_date", 
                     parte="dayofweek"),
        
        ConteoPrevio("renovaciones_previas",
                     llave="msno",
                     fecha="transaction_date",),
        DiferenciaDias("antiguedad_dias",
                       desde="registration_init_time",
                       hasta="transaction_date",),
        SumaPrevia("abandonos_previos",
                llave="msno",
                fecha="transaction_date",
                columna="abandona",),
        Razon("descuento",
              numerador="actual_amount_paid",
              denominador="plan_list_price",),

    ],
    #con 1,6 millones de filas el motor exacto tarda horas y el de histograma
    #minuto y medio: agrupa cada variable en 256 cubetas en vez de evaluar
    #todos los cortes posibles
    modelo=ConfigModelo(learning_rate=0.05, max_depth=3, min_samples_leaf=150,
                        n_max=200, verbose=1,
                        motor="histograma", criterio="auc"),
)


# =====================================================================================
# PASO 4 â€” entrenar e interpretar
# =====================================================================================
#Este caso NO funciona, y el bloque esta escrito para documentarlo en vez de disimularlo:
#el AUC de validacion junto al de prueba, las importancias y los umbrales cuentan solos
#que el modelo no generaliza. El diagnostico completo esta en el README.
if __name__ == "__main__":
    preparar_eventos()

    res = entrenar(exp)
    print("mejor nÃºmero de Ã¡rboles:", res.mejor_n, " (Ã³ptimo exacto:", res.n_minimo, ")")
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in res.metricas.items()})

    #las dos cifras en la misma linea: es el resumen del problema
    auc_val = roc_auc_score(res.y_val, res.p_val)
    print(f"\nAUC validacion {auc_val:.4f}   AUC prueba {res.metricas['auc']:.4f}")
    print(f"tasa base validacion {res.y_val.mean():.3%}")

    #el motor de histograma no tiene feature_importances_, asi que `importancias`
    #cae en la rama de permutacion: tarda un par de minutos
    print("\ncaida de AUC al barajar cada variable (permutacion, sobre validacion):")
    print(importancias(res, exp).round(4).to_string())

    #ojo: con criterio="auc" esta curva SUBE, no baja como la de log-loss
    c = res.curva_val
    print("\ncurva de AUC en validacion:",
          {n: round(c[n - 1], 4) for n in (1, 5, 10, 25, 50, 100, 200)})

    #fracciones pequeÃ±as porque la tasa base es del 2,68 %. La cabeza de la tabla
    #muestra el sintoma clave: miles de filas empatadas con lift por debajo de 1
    print("\numbrales (sobre validacion):")
    print(res.umbrales(fracciones=(0.005, 0.01, 0.02, 0.05, 0.10)).round(4).to_string())

    res.guardar(str(RAIZ / "modelo_kkbox.joblib"))


