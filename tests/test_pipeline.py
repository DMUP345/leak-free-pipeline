"""Pruebas de las funciones puras del nucleo.

La mitad prueba calculos y la otra mitad prueba MENSAJES DE ERROR, a proposito:
buena parte del valor del modulo esta en fallar temprano y con contexto, y eso se
rompe con la misma facilidad que una cuenta.

Importan desde `nucleo` sin ningun parche de sys.path: el paquete esta instalado en
modo editable (`pip install -e .`), asi que resuelve igual desde cualquier directorio.
"""
import numpy as np
import pandas as pd
import pytest

from nucleo import (ConfigModelo, ConteoPrevio, CorteTemporal, DiferenciaDias,
                    Experimento, Objetivo, ParteDeFecha, Razon, Relacion, SumaPrevia,
                    Tabla, elegir_n, tabla_de_umbrales)


# ---------------------------------------------------------------------------
# variables de historial: que NO miren el futuro
# ---------------------------------------------------------------------------

def test_suma_previa_no_se_cuenta_a_si_misma():
    """La prueba que mas importa: si SumaPrevia incluyera la fila actual, el modelo
    veria su propia respuesta. Las tres filas de A tienen mora 1, 0, 1 y el
    resultado tiene que ser 0, 1, 1 -- nunca 1, 1, 2."""
    df = pd.DataFrame({
        "cliente": ["A", "A", "A", "B"],
        "fecha": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01", "2025-01-15"]),
        "mora": [1, 0, 1, 1],
    })
    v = SumaPrevia("moras_previas", llave="cliente", fecha="fecha", columna="mora")
    assert v.calcular(df).tolist() == [0, 1, 1, 0]


def test_conteo_previo_respeta_el_orden_aunque_la_tabla_venga_desordenada():
    """Las filas llegan desordenadas a proposito: el conteo tiene que salir del
    orden cronologico por entidad, no del orden del archivo."""
    df = pd.DataFrame({
        "cliente": ["A", "B", "A", "A"],
        "fecha": pd.to_datetime(["2025-03-01", "2025-01-01", "2025-01-01", "2025-02-01"]),
    })
    v = ConteoPrevio("previos", llave="cliente", fecha="fecha")
    assert v.calcular(df).tolist() == [2, 0, 0, 1]


# ---------------------------------------------------------------------------
# el resto de las variables derivadas
# ---------------------------------------------------------------------------

def test_razon_convierte_infinito_en_nulo():
    """Un denominador en cero da inf y sklearn lo rechaza: tiene que salir nulo
    para que lo rellene el imputador del ColumnTransformer."""
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 0.0]})
    r = Razon("r", numerador="a", denominador="b").calcular(df)
    assert r.iloc[0] == 0.5
    assert pd.isna(r.iloc[1])


def test_diferencia_dias_trunca_y_propaga_nulos():
    df = pd.DataFrame({
        "desde": pd.to_datetime(["2018-01-01 10:30:00", None]),
        "hasta": pd.to_datetime(["2018-01-16", "2018-02-01"]),
    })
    d = DiferenciaDias("dias", desde="desde", hasta="hasta").calcular(df)
    assert d.iloc[0] == 14        # son 14 dias y 13:30, se trunca a 14
    assert pd.isna(d.iloc[1])


def test_parte_de_fecha_extrae_el_componente_pedido():
    #2025-01-06 es lunes (0) y 2025-01-11 es sabado (5)
    df = pd.DataFrame({"f": pd.to_datetime(["2025-01-06", "2025-01-11"])})
    assert ParteDeFecha("d", fecha="f", parte="dayofweek").calcular(df).tolist() == [0, 5]


# ---------------------------------------------------------------------------
# eleccion del numero de arboles
# ---------------------------------------------------------------------------

def test_elegir_n_sin_tolerancia_da_el_optimo_exacto():
    curva = [0.5, 0.42, 0.40, 0.401, 0.45]
    assert elegir_n(curva, None, None, None, ConfigModelo(tolerancia_se=0)) == (3, 3)


def test_elegir_n_con_auc_busca_el_maximo_y_recorta():
    """Con criterio auc la curva sube: el optimo esta en 4, pero el 2 ya cae dentro
    de la holgura de 0.01, asi que se prefiere el modelo mas chico."""
    curva = [0.70, 0.80, 0.805, 0.806, 0.80]
    cfg = ConfigModelo(criterio="auc", tolerancia_auc=0.01)
    assert elegir_n(curva, None, None, None, cfg) == (2, 4)


def test_elegir_n_con_auc_sin_holgura_da_el_maximo():
    curva = [0.70, 0.80, 0.805, 0.806, 0.80]
    cfg = ConfigModelo(criterio="auc", tolerancia_auc=0)
    assert elegir_n(curva, None, None, None, cfg) == (4, 4)


# ---------------------------------------------------------------------------
# tabla de umbrales
# ---------------------------------------------------------------------------

def test_tabla_de_umbrales_cobertura_creciente_y_completa():
    y = np.array([0, 0, 0, 1, 0, 1, 0, 0, 1, 0])
    p = np.array([0.1, 0.2, 0.15, 0.9, 0.3, 0.8, 0.25, 0.05, 0.7, 0.35])
    t = tabla_de_umbrales(y, p, fracciones=(0.3, 0.5, 1.0))
    assert t.loc[0.3, "atrapados"] == 3        # los 3 puntajes mas altos son los positivos
    assert t.loc[1.0, "cobertura"] == 1.0      # revisar todo atrapa todo
    assert t.cobertura.is_monotonic_increasing


def test_tabla_de_umbrales_reporta_los_empates_en_vez_de_la_fraccion_pedida():
    """Con puntajes empatados salen mas marcados que los pedidos, y la tabla debe
    decir lo que de verdad pasa: fue el sintoma clave del diagnostico de KKBox."""
    y = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])
    p = np.full(10, 0.5)                       # todos empatados
    t = tabla_de_umbrales(y, p, fracciones=(0.1,))
    assert t.loc[0.1, "revisados"] == 10       # no 1


def test_tabla_de_umbrales_sin_positivos_falla():
    with pytest.raises(ValueError, match="positivos"):
        tabla_de_umbrales(np.zeros(10), np.linspace(0, 1, 10))


# ---------------------------------------------------------------------------
# validaciones que tienen que fallar temprano y con contexto
# ---------------------------------------------------------------------------

def test_corte_temporal_nombra_las_particiones_vacias():
    df = pd.DataFrame({"f": pd.to_datetime(["2019-01-01", "2019-06-01"])})
    with pytest.raises(ValueError, match="particiones vacias"):
        CorteTemporal("f", "2025-01-01", "2025-06-01").partir(df)


def test_corte_temporal_detecta_la_columna_que_llego_como_texto():
    df = pd.DataFrame({"f": ["2025-01-01", "2025-06-01"]})
    with pytest.raises(TypeError, match="no es de tipo fecha"):
        CorteTemporal("f", "2025-03-01", "2025-05-01").partir(df)


def test_corte_temporal_rechaza_cortes_invertidos():
    with pytest.raises(ValueError, match="anterior"):
        CorteTemporal("f", "2025-06-01", "2025-01-01")


def _relacion_sin_leer_archivos():
    """Tabla y Relacion no tocan el disco al construirse: solo .cargar() y .unir() lo
    hacen, asi que estas rutas inexistentes sirven para probar las validaciones."""
    return Relacion(
        eventos=Tabla("ev", "no_se_lee.csv", llave="id"),
        entidades=Tabla("en", "tampoco.csv", llave="k"),
        llave="k")


def test_experimento_rechaza_el_objetivo_como_variable_de_entrada():
    """La fuga mas burda posible: predecir mora usando mora."""
    with pytest.raises(ValueError, match="objetivo"):
        Experimento(
            relacion=_relacion_sin_leer_archivos(),
            corte=CorteTemporal("f", "2025-01-01", "2025-06-01"),
            objetivo=Objetivo("mora"),
            numericas=["mora"],
            categoricas=[],
        )


def test_experimento_rechaza_nombres_repetidos():
    with pytest.raises(ValueError, match="repetido"):
        Experimento(
            relacion=_relacion_sin_leer_archivos(),
            corte=CorteTemporal("f", "2025-01-01", "2025-06-01"),
            objetivo=Objetivo("mora"),
            numericas=["monto"],
            categoricas=["monto"],
        )


def test_experimento_rechaza_una_derivada_que_no_cumple_el_protocolo():
    class NoEsVariable:
        nombre = "x"
        # le falta calcular()

    with pytest.raises(TypeError, match="Variable"):
        Experimento(
            relacion=_relacion_sin_leer_archivos(),
            corte=CorteTemporal("f", "2025-01-01", "2025-06-01"),
            objetivo=Objetivo("mora"),
            numericas=[],
            categoricas=[],
            derivadas=[NoEsVariable()],
        )


@pytest.mark.parametrize("kwargs, mensaje", [
    ({"motor": "gpu"}, "motor"),
    ({"criterio": "f1"}, "criterio"),
    ({"tolerancia_se": -1}, "tolerancia_se"),
    ({"imputacion": "moda"}, "imputacion"),
    ({"learning_rate": 0}, "learning_rate"),
    ({"max_depth": 0}, "max_depth"),
])
def test_config_modelo_valida_sus_perillas(kwargs, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        ConfigModelo(**kwargs)
