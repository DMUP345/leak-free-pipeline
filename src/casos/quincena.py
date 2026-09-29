"""El caso Quincena escrito como configuraciÃ³n. Nada mÃ¡s cambia."""
from pathlib import Path

from nucleo import (ConfigModelo, ConteoPrevio, CorteTemporal, Experimento, Objetivo,
                    Razon, Relacion, SumaPrevia, Tabla, entrenar)

#relativo a este archivo, no al directorio desde donde se ejecuta
RAIZ = Path(__file__).resolve().parent.parent.parent
DATOS = Path(__file__).resolve().parent.parent / "datasets" / "quincena"

prestamos = Tabla("prestamos", str(DATOS / "prestamos.csv"), llave="prestamo_id", fechas=("fecha_desembolso",))
clientes = Tabla("clientes", str(DATOS / "clientes.csv"), llave="cliente_id")

exp = Experimento(
    relacion=Relacion(eventos=prestamos, entidades=clientes, llave="cliente_id"),
    corte=CorteTemporal("fecha_desembolso", inicio_validacion="2025-08-01", inicio_prueba="2025-10-01"),
    objetivo=Objetivo("mora"),
    numericas=["edad", "ingreso_mensual", "meses_empleo", "monto", "plazo_dias"],
    categoricas=["tipo_contrato", "ciudad"],
    derivadas=[
        Razon("cuota_ingreso", numerador="monto", denominador="ingreso_mensual"),
        ConteoPrevio("prestamos_previos", llave="cliente_id", fecha="fecha_desembolso"),
        SumaPrevia("moras_previas", llave="cliente_id", fecha="fecha_desembolso", columna="mora"),
    ],
    modelo=ConfigModelo(learning_rate=0.05, max_depth=3, min_samples_leaf=30),
)

#lo de arriba son definiciones baratas, asi que `from config import exp` no entrena nada;
#lo de aqui abajo solo corre al ejecutar el archivo
if __name__ == "__main__":
    res = entrenar(exp)
    print("mejor nÃºmero de Ã¡rboles:", res.mejor_n)
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in res.metricas.items()})
    res.guardar(str(RAIZ / "modelo_quincena.joblib"))

# el mismo pipeline sobre otro dominio vive en ejemplo_otro_dominio.py