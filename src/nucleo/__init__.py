"""El núcleo: lo que no cambia entre casos.

Un pipeline de modelado para problemas de **entidades + eventos**. Nada de lo que hay
aquí menciona una columna concreta: los casos viven en `src/casos/` y se limitan a
instanciar estas clases con los nombres de sus propios datos.

Lo que exporta este paquete es la superficie pública. Todo lo demás —`_perdida_por_fila`,
por ejemplo— es interno y puede cambiar sin aviso.

    from nucleo import Experimento, entrenar

Las piezas, en el orden en que se usan:

    Tabla, Relacion          qué archivos se cargan y cómo se unen
    Variable (Protocol)      el contrato de una variable derivada: nombre + calcular(df)
    Razon, ConteoPrevio,     las implementaciones. Añadir una no exige tocar nada más:
    SumaPrevia,              ConteoPrevio y SumaPrevia excluyen la fila actual por
    DiferenciaDias,          construcción, que es lo que evita la fuga temporal
    ParteDeFecha
    CorteTemporal            entrenamiento / validación / prueba por fecha
    Objetivo, ConfigModelo   qué se predice y con qué perillas
    Experimento              junta todo lo anterior y lo valida al construirse
    entrenar(exp)            devuelve un Resultado
    tabla_de_umbrales        qué se atrapa al revisar cada fracción de los casos
    importancias             por impureza o por permutación, según lo que el motor permita
"""
from .pipeline import (ConfigModelo, ConteoPrevio, CorteTemporal, DiferenciaDias,
                       Experimento, Objetivo, ParteDeFecha, Razon, Relacion, Resultado,
                       SumaPrevia, Tabla, Variable, construir_pipeline, construir_tabla,
                       elegir_n, entrenar, importancias, tabla_de_umbrales)

__all__ = [
    # configuración de un caso
    "Tabla", "Relacion", "CorteTemporal", "Objetivo", "ConfigModelo", "Experimento",
    # variables derivadas
    "Variable", "Razon", "ConteoPrevio", "SumaPrevia", "DiferenciaDias", "ParteDeFecha",
    # proceso y resultados
    "entrenar", "Resultado", "construir_tabla", "construir_pipeline", "elegir_n",
    "tabla_de_umbrales", "importancias",
]
