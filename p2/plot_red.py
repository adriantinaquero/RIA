import matplotlib.pyplot as plt

def dibujar_red(genoma, config, nombres_entrada, nombres_salida, ruta):
    gc = config.genome_config
    entradas, salidas = list(gc.input_keys), list(gc.output_keys)
    activas = [c.key for c in genoma.connections.values() if c.enabled]
    # Profundidad de cada nodo: camino más largo desde las entradas
    prof = {n: 0 for n in entradas}
    for _ in range(len(genoma.nodes) + 1):
        for (a, b) in activas:
            nueva = prof[a] + 1 if a in prof and a != b else -1
            if prof.get(b, -1) < nueva <= len(genoma.nodes):
                prof[b] = nueva
    ocultas = [n for n in prof if n not in entradas and n not in salidas]
    capas = [entradas]
    for p in sorted({prof[n] for n in ocultas}):
        capas.append(sorted(n for n in ocultas if prof[n] == p))
    capas.append(salidas)
    pos = {n: (i, j - (len(c) - 1) / 2)
           for i, c in enumerate(capas) for j, n in enumerate(c)}
    fig, ax = plt.subplots(figsize=(6, 4))
    for (a, b) in activas:
        if a in pos and b in pos:
            w = genoma.connections[(a, b)].weight
            ax.annotate("", xy=pos[b], xytext=pos[a], arrowprops=dict(
                        arrowstyle="->", lw=0.5 + 1.5 * min(abs(w), 3) / 3,
                        color="tab:green" if w > 0 else "tab:red"))
    nombres = dict(zip(entradas, nombres_entrada))
    nombres.update(zip(salidas, nombres_salida))
    for n, (x, y) in pos.items():
        ax.scatter(x, y, s=300, color="lightgrey", edgecolor="black", zorder=3)
        ax.text(x, y + 0.25, nombres.get(n, str(n)), ha="center", fontsize=8)
    ax.axis("off")
    fig.savefig(ruta, dpi=200, bbox_inches="tight")