"""Genera las figuras del README en figuras/.

No es parte del paquete: es una herramienta de desarrollo. Los PNG van versionados,
así que solo hace falta ejecutarlo cuando cambien los resultados. Necesita los datos
de Olist y de KKBox.

    python src/figuras.py

Paleta y reglas de trazo: un color por serie asignado en orden fijo, nunca ciclado;
ejes y rejilla recesivos; el texto en tinta, nunca en el color de la serie; leyenda
siempre que haya dos series; sin ejes dobles.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from casos.kkbox import exp as EXP_KKBOX
from casos.olist import experimento as EXP_OLIST
from casos.quincena import exp as EXP_QUINCENA
from nucleo import construir_tabla, entrenar, tabla_de_umbrales

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "figuras"

# paleta validada (dataviz, modo claro sobre #fcfcfb): ΔE 24,7 en visión CVD
SUPERFICIE = "#fcfcfb"
SERIE_1 = "#2a78d6"     # azul
SERIE_2 = "#eb6834"     # naranja
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
MUTE = "#898781"
REJILLA = "#e1e0d9"
EJE = "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SUPERFICIE,      # opaco: GitHub lo muestra igual en tema claro y oscuro
    "axes.facecolor": SUPERFICIE,
    "savefig.facecolor": SUPERFICIE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": EJE,
    "axes.labelcolor": TINTA_2,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.titlecolor": TINTA,
    "xtick.color": MUTE,
    "ytick.color": MUTE,
    "xtick.labelcolor": TINTA_2,
    "ytick.labelcolor": TINTA_2,
    "grid.color": REJILLA,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "figure.dpi": 140,
})


def _limpiar(ax, ejes=("top", "right")):
    for lado in ejes:
        ax.spines[lado].set_visible(False)
    ax.tick_params(length=0)


def _guardar(fig, nombre):
    SALIDA.mkdir(exist_ok=True)
    ruta = SALIDA / nombre
    fig.savefig(ruta, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("  escrita", ruta.relative_to(RAIZ))


# ---------------------------------------------------------------------------

def figura_deriva_umbral(res_olist):
    """Un umbral fijo elegido en validación marca casi el doble en prueba.

    Barras agrupadas: lo que el umbral debía marcar contra lo que marcó de verdad.
    Dos series -> dos colores en orden fijo, con leyenda y etiquetas directas.
    """
    df = construir_tabla(EXP_OLIST)
    m = EXP_OLIST.corte.partir(df)
    X = df[EXP_OLIST.columnas_numericas + EXP_OLIST.categoricas]
    p_pru = res_olist.pipeline.predict_proba(X[m["pru"]])[:, 1]

    fracciones = (0.05, 0.10, 0.20)
    umbrales = tabla_de_umbrales(res_olist.y_val, res_olist.p_val, fracciones)["umbral"]
    marcado_pru = [float((p_pru >= u).mean()) for u in umbrales]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    y = np.arange(len(fracciones))
    alto = 0.36
    ax.barh(y + alto / 2, [f * 100 for f in fracciones], alto,
            color=SERIE_1, label="lo que debía marcar")
    ax.barh(y - alto / 2, [v * 100 for v in marcado_pru], alto,
            color=SERIE_2, label="lo que marcó en prueba")

    for i, (f, v) in enumerate(zip(fracciones, marcado_pru)):
        ax.text(f * 100 + 0.6, i + alto / 2, f"{f:.0%}", va="center", fontsize=8.5,
                color=TINTA_2)
        ax.text(v * 100 + 0.6, i - alto / 2, f"{v * 100:.1f} %".replace(".", ","),
                va="center", fontsize=8.5, color=TINTA, fontweight="semibold")

    ax.set_yticks(y, [f"umbral {u:.3f}".replace(".", ",") for u in umbrales])
    ax.set_xlabel("porcentaje de pedidos marcados")
    ax.set_xlim(0, max(marcado_pru) * 100 * 1.18)
    ax.set_title("Un umbral fijo no sobrevive al tiempo")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    _limpiar(ax, ("top", "right", "left"))
    ax.legend(loc="lower right", fontsize=8.5, labelcolor=TINTA_2)
    fig.text(0.5, -0.06, "Olist · la tasa real de retraso bajó de 6,8 % a 5,5 % entre los "
             "dos periodos, y aun así el mismo umbral marca casi el doble",
             ha="center", fontsize=8, color=MUTE, wrap=True)
    _guardar(fig, "01_deriva_umbral.png")


def figura_curva_validacion(res_q):
    """El mínimo exacto de la curva es ruido; la regla del error estándar recorta."""
    curva = np.asarray(res_q.curva_val)
    n = np.arange(1, len(curva) + 1)
    #los primeros 20 arboles caen en picado y aplastarian la zona plana, que es donde
    #esta la historia: el minimo y el punto elegido se distinguen apenas
    DESDE = 20

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(n[DESDE:], curva[DESDE:], color=SERIE_1, linewidth=2)

    for x, etiqueta, color, dy in [
        (res_q.n_minimo, f"mínimo exacto\nn = {res_q.n_minimo}", MUTE, 18),
        (res_q.mejor_n, f"regla del error estándar\nn = {res_q.mejor_n}", TINTA, 42),
    ]:
        ax.plot([x], [curva[x - 1]], "o", markersize=8, color=color,
                markeredgecolor=SUPERFICIE, markeredgewidth=2, zorder=5)
        ax.annotate(etiqueta, (x, curva[x - 1]), textcoords="offset points",
                    xytext=(10, dy), fontsize=8.5, color=color,
                    arrowprops=dict(arrowstyle="-", color=EJE, linewidth=1))

    ax.set_xlabel("árboles (desde 20: antes la curva cae en picado)")
    ax.set_ylabel("log-loss en validación")
    ax.set_title("El mínimo de la curva es ruido")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    _limpiar(ax)
    fig.text(0.5, -0.06, "Quincena · el óptimo en "
             f"{res_q.n_minimo} árboles le gana al de {res_q.n_minimo - 6} por 0,00017. "
             f"La regla emparejada elige {res_q.mejor_n}: {100 - res_q.mejor_n * 100 // res_q.n_minimo} "
             "% menos árboles, mismo desempeño en prueba",
             ha="center", fontsize=8, color=MUTE)
    _guardar(fig, "02_curva_validacion.png")


def figura_inestabilidad_kkbox():
    """Ninguna variable mantiene su relación con el objetivo a lo largo del tiempo.

    Pequeños múltiplos: una variable por panel, una sola serie en cada uno, así que
    el color no tiene que distinguir nada y basta un hue.
    """
    df = construir_tabla(EXP_KKBOX)
    y = df[EXP_KKBOX.objetivo.columna]
    trim = df.transaction_date.dt.to_period("Q")
    variables = ["plan_list_price", "is_cancel", "abandonos_previos", "dia_semana"]
    titulos = ["precio del plan", "es cancelación", "abandonos previos", "día de la semana"]

    filas = []
    for t in sorted(trim.unique()):
        sub = trim == t
        if y[sub].nunique() < 2 or sub.sum() < 5000:
            continue
        fila = {"t": str(t)}
        for c in variables:
            s = df[c][sub]
            fila[c] = roc_auc_score(y[sub], s.fillna(s.median()))
        filas.append(fila)
    t = pd.DataFrame(filas)

    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.9), sharey=True)
    x = np.arange(len(t))
    for ax, col, titulo in zip(axes, variables, titulos):
        ax.axhline(0.5, color=EJE, linewidth=1.2, zorder=1)
        ax.plot(x, t[col], color=SERIE_1, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SUPERFICIE, markeredgewidth=1.5, zorder=3)
        ax.set_title(titulo, fontsize=9.5)
        ax.set_xticks(x[::2], [s[2:] for s in t.t[::2]], fontsize=7.5)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        _limpiar(ax, ("top", "right"))
    axes[0].set_ylabel("AUC de la variable sola")
    axes[0].set_ylim(0.05, 0.85)
    #el rotulo va en el panel con hueco bajo la linea, no encima de los datos
    axes[2].text(len(t) - 0.6, 0.44, "sin relación", fontsize=7.5, color=MUTE, ha="right")
    fig.suptitle("Ninguna variable mantiene su relación con el objetivo",
                 fontsize=11, fontweight="semibold", color=TINTA, y=1.04)
    fig.text(0.5, -0.1, "KKBox · por debajo de 0,5 la relación va al revés. El precio del "
             "plan pasa de 0,11 a 0,68 y vuelve a 0,40; el día de la semana, que no puede "
             "predecir nada, llega a 0,73",
             ha="center", fontsize=8, color=MUTE)
    _guardar(fig, "03_inestabilidad_kkbox.png")


def figura_lift_kkbox(res_kkbox):
    """La cabeza del ranking es anti-predictiva y el medio funciona bien."""
    fracciones = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30)
    t = tabla_de_umbrales(res_kkbox.y_val, res_kkbox.p_val, fracciones)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(t))
    ax.axhline(1, color=EJE, linewidth=1.2, zorder=1)
    ax.plot(x, t.lift, color=SERIE_1, linewidth=2, marker="o", markersize=7,
            markeredgecolor=SUPERFICIE, markeredgewidth=2, zorder=3)

    peor = int(np.argmin(t.lift.values))
    mejor = int(np.argmax(t.lift.values))
    ax.annotate(f"lift {t.lift.iloc[peor]:.2f}".replace(".", ",")
                + f"\n{int(t.revisados.iloc[peor]):,} casos".replace(",", ".")
                + f", {int(t.atrapados.iloc[peor])} aciertos",
                (x[peor], t.lift.iloc[peor]), textcoords="offset points", xytext=(14, 26),
                fontsize=8.5, color=TINTA, fontweight="semibold",
                arrowprops=dict(arrowstyle="-", color=EJE, linewidth=1))
    ax.annotate(f"lift {t.lift.iloc[mejor]:.2f}".replace(".", ","),
                (x[mejor], t.lift.iloc[mejor]),
                textcoords="offset points", xytext=(8, -4), fontsize=8.5, color=TINTA)
    ax.text(len(t) - 1.05, 1.25, "azar", fontsize=8, color=MUTE, ha="right")

    ax.set_xticks(x, [f"{f:.1%}".replace(".", ",") if f < 0.01 else f"{f:.0%}"
                      for f in fracciones])
    ax.set_xlabel("fracción de casos revisados, de mayor a menor puntaje")
    ax.set_ylabel("lift sobre la tasa base")
    ax.set_title("El orden está roto en la cabeza y funciona en el medio")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    _limpiar(ax)
    fig.text(0.5, -0.06, "KKBox · los casos que el modelo señala con más confianza "
             "aciertan doce veces menos que elegir al azar",
             ha="center", fontsize=8, color=MUTE)
    _guardar(fig, "04_lift_kkbox.png")


if __name__ == "__main__":
    print("entrenando Quincena...")
    res_q = entrenar(EXP_QUINCENA)
    figura_curva_validacion(res_q)

    print("entrenando Olist...")
    res_o = entrenar(EXP_OLIST)
    figura_deriva_umbral(res_o)

    print("entrenando KKBox...")
    res_k = entrenar(EXP_KKBOX)
    figura_lift_kkbox(res_k)
    figura_inestabilidad_kkbox()
