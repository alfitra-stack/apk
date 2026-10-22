"""Grafik bertema gelap untuk aplikasi (matplotlib)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BG, CARD, TXT, MUTED, GRID = "#0b1220", "#131c31", "#e6edf7", "#8fa1c0", "#24304d"
CL = ["#22d3ee", "#fbbf24", "#fb7185", "#a78bfa", "#34d399", "#f472b6"]


def _base(figsize):
    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(figsize=figsize)
    _style(fig, ax)
    return fig, ax


def _style(fig, *axes):
    fig.patch.set_facecolor(CARD)
    for ax in axes:
        ax.set_facecolor(CARD)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(TXT)
        ax.grid(color=GRID, lw=.6, alpha=.7)
        ax.set_axisbelow(True)


def _legend(ax, **kw):
    lg = ax.legend(frameon=False, **kw)
    for t in lg.get_texts():
        t.set_color(TXT)
    return lg


def ranking(pc):
    s = pc.sort_values("indeks_harga")
    fig, ax = _base((8, 9))
    ax.barh(s.provinsi, s.indeks_harga, color=[CL[c % len(CL)] for c in s.cluster], height=.72)
    ax.axvline(100, color="#ffffff", ls="--", lw=1, alpha=.8)
    ax.text(100.8, 0.2, "rata-rata nasional = 100", color=TXT, fontsize=8)
    ax.set_xlabel("Indeks Harga Pangan")
    ax.tick_params(axis="y", labelsize=7.5, colors=TXT)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def scatter_ump(prov):
    fig, ax = _base((6.5, 4.4))
    x, y = prov.ump / 1e6, prov.indeks_harga
    ax.scatter(x, y, s=55, color=CL[0], alpha=.85, edgecolor=BG, lw=.6)
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, a + b * xs, color=CL[1], lw=2)
    for _, r in prov.nlargest(2, "indeks_harga").iterrows():
        ax.annotate(r.provinsi, (r.ump / 1e6, r.indeks_harga), color=TXT, fontsize=7.5,
                    xytext=(5, 4), textcoords="offset points")
    r_ = np.corrcoef(x, y)[0, 1]
    ax.set_title(f"UMP vs Indeks Harga Pangan (r = {r_:.2f})", fontsize=10)
    ax.set_xlabel("UMP 2024 (juta Rp)"); ax.set_ylabel("Indeks harga")
    fig.tight_layout()
    return fig


def corr_heatmap(prov, komoditas):
    c = prov[komoditas].corr()
    fig, ax = _base((6.5, 5.6))
    im = ax.imshow(c.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c))); ax.set_yticks(range(len(c)))
    ax.set_xticklabels(c.columns, rotation=60, ha="right", fontsize=6.5, color=MUTED)
    ax.set_yticklabels(c.columns, fontsize=6.5, color=MUTED)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=.045)
    cb.ax.tick_params(colors=MUTED, labelsize=7); cb.outline.set_visible(False)
    ax.set_title("Korelasi Antarharga Komoditas", fontsize=10)
    fig.tight_layout()
    return fig


def pca_plot(pc, names, ev):
    fig, ax = _base((8, 6))
    for c in sorted(pc.cluster.unique()):
        d = pc[pc.cluster == c]
        ax.scatter(d.pc1, d.pc2, s=85, color=CL[c % len(CL)], label=names[c],
                   edgecolor=BG, lw=.8, alpha=.95, zorder=3)
        for _, r in d.iterrows():
            ax.annotate(r.provinsi, (r.pc1, r.pc2), fontsize=6, color=MUTED,
                        xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({ev[0]*100:.0f}%)"); ax.set_ylabel(f"PC2 ({ev[1]*100:.0f}%)")
    _legend(ax, fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def elbow_sil(sil_df, k_sel):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    _style(fig, *axes)
    axes[0].plot(sil_df.k, sil_df.inertia, "-o", color=CL[0], lw=2)
    axes[0].set_title("Elbow Method"); axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
    axes[1].plot(sil_df.k, sil_df.silhouette, "-o", color=CL[4], lw=2)
    axes[1].set_title("Silhouette Score"); axes[1].set_xlabel("k")
    for a in axes:
        a.axvline(k_sel, color=CL[1], ls="--", lw=1.2, alpha=.9)
    fig.tight_layout()
    return fig


def trend(h, row):
    fig, ax = _base((8, 3.8))
    lab = list(pd.to_datetime(h.bulan).dt.strftime("%b %y")) + ["Jan 25\n(prediksi)"]
    xs = np.arange(len(lab))
    ax.plot(xs[:-1], h.indeks_harga.values, "-o", color=CL[0], lw=2.2, label="Aktual", zorder=3)
    ax.plot(xs[-2:], [h.indeks_harga.iloc[-1], row.prediksi_bulan_depan], "--o",
            color=CL[2], lw=2.2, label="Prediksi", zorder=3)
    ax.axhline(100, color="#ffffff", ls=":", lw=1, alpha=.6)
    ax.set_xticks(xs); ax.set_xticklabels(lab)
    ax.set_ylabel("Indeks harga")
    _legend(ax, fontsize=8)
    fig.tight_layout()
    return fig


def compare_bars(prov_name, pred_val, cluster_mean, cluster_label):
    fig, ax = _base((6, 2.6))
    labels = [prov_name, f"Rata-rata {cluster_label}", "Nasional"]
    vals = [pred_val, cluster_mean, 100]
    cols = [CL[3], CL[1], CL[0]]
    ys = list(range(len(labels)))[::-1]
    for y, v, c in zip(ys, vals, cols):
        ax.hlines(y, 0, v, color=c, lw=2, alpha=.5)
        ax.scatter([v], [y], s=120, color=c, zorder=3, edgecolor=BG)
        ax.text(v + 0.012 * max(vals), y + .18, f"{v:.1f}", color=TXT, fontsize=9)
    ax.set_yticks(ys); ax.set_yticklabels(labels, color=TXT)
    lo = min(vals) * .9
    ax.set_xlim(lo, max(vals) * 1.05)
    ax.set_xlabel("Indeks harga (sumbu dipotong agar selisih terlihat)", fontsize=8)
    ax.set_title("Prediksi Jan 2025 dibanding pembanding", fontsize=10)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig
