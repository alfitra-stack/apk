import streamlit as st
import pandas as pd
import model as m
import charts as ch

st.set_page_config(page_title="Harga Pangan Antarprovinsi", page_icon="🌾", layout="wide")

CSS = """
<style>
.block-container {padding-top: 1.2rem; max-width: 1250px;}
.hero {
  background: linear-gradient(120deg, #0f766e 0%, #1d4ed8 55%, #6d28d9 100%);
  border-radius: 20px; padding: 28px 32px; margin-bottom: 18px;
  box-shadow: 0 10px 30px rgba(29,78,216,.35);
}
.hero h1 {margin: 0 0 6px 0; font-size: 1.9rem; line-height: 1.2; color: #fff;}
.hero p {margin: 0 0 14px 0; color: #dbeafe; font-size: .95rem;}
.chip {display: inline-block; background: rgba(255,255,255,.16); color: #fff;
  border: 1px solid rgba(255,255,255,.28); border-radius: 999px;
  padding: 3px 12px; margin: 0 6px 6px 0; font-size: .78rem;}
.kpi {background: #131c31; border: 1px solid #24304d; border-radius: 16px;
  padding: 16px 18px; height: 100%;}
.kpi .lab {color: #8fa1c0; font-size: .78rem; text-transform: uppercase; letter-spacing: .06em;}
.kpi .val {font-size: 1.9rem; font-weight: 700; color: #e6edf7; line-height: 1.25;}
.kpi .sub {color: #8fa1c0; font-size: .85rem;}
.kpi.a {border-top: 3px solid #22d3ee;} .kpi.b {border-top: 3px solid #fb7185;}
.kpi.c {border-top: 3px solid #34d399;} .kpi.d {border-top: 3px solid #fbbf24;}
.ccard {background: #131c31; border: 1px solid #24304d; border-radius: 16px;
  padding: 14px 16px; margin-bottom: 10px;}
.ccard .t {font-weight: 700; font-size: 1rem;}
.ccard .s {color: #8fa1c0; font-size: .82rem; margin-bottom: 6px;}
.pill {display:inline-block; background:#1c2745; border-radius:8px;
  padding: 2px 9px; margin: 2px 4px 2px 0; font-size: .76rem; color:#cbd5e1;}
.sect {font-size: 1.15rem; font-weight: 700; margin: 6px 0 2px 0;}
.note {background: #131c31; border-left: 4px solid #fbbf24; border-radius: 8px;
  padding: 10px 14px; font-size: .88rem; color: #cbd5e1;}
div[data-baseweb="tab-list"] {gap: 6px;}
button[data-baseweb="tab"] {background: #131c31; border-radius: 10px 10px 0 0; padding: 8px 18px;}
button[data-baseweb="tab"][aria-selected="true"] {background: #1c2745;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

CL = ch.CL


def kpi(col, cls, lab, val, sub=""):
    col.markdown(f'<div class="kpi {cls}"><div class="lab">{lab}</div>'
                 f'<div class="val">{val}</div><div class="sub">{sub}</div></div>',
                 unsafe_allow_html=True)


@st.cache_data
def get_data():
    return m.load_data()


@st.cache_data
def get_sil(_prov, komoditas):
    return m.silhouette_table(_prov, komoditas)


prov, panel, komoditas = get_data()

# ---------------- Sidebar ----------------
st.sidebar.markdown("### ⚙️ Pengaturan")
k = st.sidebar.slider("Jumlah cluster (k)", 2, 6, 3)
st.sidebar.caption("k = 3 dipilih pada analisis utama (lihat tab Evaluasi untuk Elbow dan Silhouette).")
st.sidebar.markdown("---")
st.sidebar.markdown("**Cara membaca indeks**  \nIndeks 100 = rata-rata harga nasional. "
                    "Di atas 100 berarti lebih mahal dari rata-rata.")

pc, sil, ev = m.run_kmeans(prov, komoditas, k)
names = m.cluster_names(pc, k)
pc["nama_cluster"] = pc.cluster.map(names)
res, pred, wfull = m.run_regression(panel, pc)

# ---------------- Hero ----------------
st.markdown(
    '<div class="hero"><h1>🌾 Analisis Pengelompokan dan Prediksi<br>Harga Bahan Pokok Antarprovinsi</h1>'
    '<p>Memetakan provinsi berdasarkan pola harga pangan dan memprediksi indeks harga bulan berikutnya.</p>'
    '<span class="chip">K-Means Clustering</span><span class="chip">Linear Regression</span>'
    '<span class="chip">38 Provinsi</span><span class="chip">Jul - Des 2024</span>'
    '<span class="chip">Data: Satu Data Indonesia</span></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "🧩 Clustering", "🔮 Prediksi", "🧪 Evaluasi"])

# ---------------- Tab 1: Data ----------------
with tab1:
    top = prov.loc[prov.indeks_harga.idxmax()]
    low = prov.loc[prov.indeks_harga.idxmin()]
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "a", "Jumlah provinsi", len(prov), f"{len(komoditas)} komoditas")
    kpi(c2, "b", "Indeks tertinggi", f"{top.indeks_harga:.1f}", top.provinsi)
    kpi(c3, "c", "Indeks terendah", f"{low.indeks_harga:.1f}", low.provinsi)
    kpi(c4, "d", "Selisih tertinggi-terendah", f"{top.indeks_harga - low.indeks_harga:.1f}", "poin indeks")
    st.write("")

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown('<div class="sect">Peringkat Indeks Harga per Provinsi</div>', unsafe_allow_html=True)
        st.caption("Warna batang menunjukkan cluster (sesuai nilai k di sidebar).")
        st.pyplot(ch.ranking(pc))
    with right:
        st.markdown('<div class="sect">UMP dan Indeks Harga</div>', unsafe_allow_html=True)
        st.pyplot(ch.scatter_ump(prov))
        st.markdown('<div class="sect">Korelasi Antarkomoditas</div>', unsafe_allow_html=True)
        st.pyplot(ch.corr_heatmap(prov, komoditas))

    st.markdown('<div class="sect">Data per provinsi (rata-rata Jul-Des 2024)</div>', unsafe_allow_html=True)
    st.dataframe(prov[["provinsi", "ump", "indeks_harga"] + komoditas].round(1),
                 use_container_width=True, hide_index=True)

# ---------------- Tab 2: Clustering ----------------
with tab2:
    sizes = pc.cluster.value_counts()
    c1, c2, c3 = st.columns(3)
    kpi(c1, "a", "Jumlah cluster", k, "K-Means, Z-score")
    kpi(c2, "d", "Silhouette score", f"{sil:.3f}", "makin dekat 1 makin tegas")
    kpi(c3, "c", "Cluster terbesar", int(sizes.max()), "provinsi")
    st.write("")

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown('<div class="sect">Peta Cluster (proyeksi PCA)</div>', unsafe_allow_html=True)
        st.pyplot(ch.pca_plot(pc, names, ev))
    with right:
        st.markdown('<div class="sect">Profil Cluster</div>', unsafe_allow_html=True)
        for c in sorted(pc.cluster.unique()):
            d = pc[pc.cluster == c]
            pills = "".join(f'<span class="pill">{p}</span>' for p in d.provinsi)
            st.markdown(
                f'<div class="ccard" style="border-left:5px solid {CL[c % len(CL)]}">'
                f'<div class="t">{names[c]}</div>'
                f'<div class="s">{len(d)} provinsi | indeks rata-rata {d.indeks_harga.mean():.1f} | '
                f'UMP rata-rata Rp{d.ump.mean():,.0f}'.replace(",", ".") + f'</div>{pills}</div>',
                unsafe_allow_html=True)
    if pc.cluster.value_counts().min() == 1:
        st.markdown('<div class="note">Ada cluster yang hanya berisi 1 provinsi. Itu terjadi karena Papua '
                    'Pegunungan memiliki harga yang sangat ekstrem (beberapa harganya konstan selama 6 bulan, '
                    'kemungkinan harga acuan).</div>', unsafe_allow_html=True)

# ---------------- Tab 3: Prediksi ----------------
with tab3:
    st.markdown('<div class="sect">Prediksi Indeks Harga Bulan Berikutnya (Januari 2025)</div>',
                unsafe_allow_html=True)
    st.caption("Model B3: Linear Regression dengan fitur indeks harga bulan ini, UMP, dan cluster provinsi.")
    pilih = st.selectbox("Pilih provinsi", sorted(prov.provinsi),
                         index=sorted(prov.provinsi).index("Kepulauan Riau")
                         if "Kepulauan Riau" in set(prov.provinsi) else 0)
    row = pred[pred.provinsi == pilih].iloc[0]
    cnum = int(row.cluster)
    cl_name = names[cnum]
    st.markdown(f'<span class="chip" style="background:{CL[cnum % len(CL)]}33;'
                f'border-color:{CL[cnum % len(CL)]}">{cl_name}</span>', unsafe_allow_html=True)

    a1, a2, a3, a4 = st.columns(4)
    kpi(a1, "a", "UMP 2024", f"Rp{row.ump/1e6:.2f} jt")
    kpi(a2, "d", "Indeks Des 2024", f"{row.indeks_harga:.1f}")
    kpi(a3, "b", "Prediksi Jan 2025", f"{row.prediksi_bulan_depan:.1f}", f"{row.perubahan:+.2f} dari Des 2024")
    kpi(a4, "c", "Selisih dari nasional", f"{row.prediksi_bulan_depan - 100:+.1f}", "poin indeks")
    st.write("")

    h = wfull[wfull.provinsi == pilih].sort_values("bulan")
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown('<div class="sect">Tren Indeks dan Prediksi</div>', unsafe_allow_html=True)
        st.pyplot(ch.trend(h, row))
    with right:
        cm = pc[pc.cluster == cnum].indeks_harga.mean()
        st.markdown('<div class="sect">Perbandingan</div>', unsafe_allow_html=True)
        st.pyplot(ch.compare_bars(pilih, row.prediksi_bulan_depan, cm, f"Cluster {cnum}"))

    st.markdown('<div class="sect">Harga komoditas provinsi ini (Jul-Des 2024)</div>', unsafe_allow_html=True)
    komod_pilih = st.multiselect("Komoditas", komoditas, default=["Beras", "Cabai Rawit", "Bawang Merah"])
    if komod_pilih:
        hh = h.set_index(pd.to_datetime(h.bulan).dt.strftime("%Y-%m"))[komod_pilih]
        st.line_chart(hh)

    with st.expander("Prediksi semua provinsi"):
        st.dataframe(pred.sort_values("prediksi_bulan_depan", ascending=False).round(2),
                     use_container_width=True, hide_index=True)

# ---------------- Tab 4: Evaluasi ----------------
with tab4:
    st.markdown('<div class="sect">Memilih jumlah cluster</div>', unsafe_allow_html=True)
    sil_df = get_sil(prov, komoditas)
    st.pyplot(ch.elbow_sil(sil_df, k))
    st.caption("Garis putus-putus menandai k yang sedang dipilih di sidebar.")
    st.dataframe(sil_df.round(3), use_container_width=True, hide_index=True)
    st.markdown('<div class="note">Silhouette tertinggi ada di k = 2 (0,70), tetapi itu hanya memisahkan '
                'Papua Pegunungan dari provinsi lain. k = 3 (0,39) dipilih agar cluster lebih informatif. '
                'Pemisahan antar cluster tidak tegas (silhouette &lt; 0,5).</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="sect">Evaluasi Linear Regression (split waktu)</div>', unsafe_allow_html=True)
    st.caption("Data latih: pasangan bulan Jul-Sep ke bulan berikutnya. Data uji: Okt-Nov ke bulan berikutnya.")
    st.dataframe(res.round(3), use_container_width=True)
    r3 = res.loc["B3: Indeks + UMP + cluster", "R2"]
    rb = res.loc["Baseline naif (bulan depan = bulan ini)", "R2"]
    teks = f"Dengan k = {k}, Model B3 memiliki R2 {r3:.3f}. "
    if rb > r3:
        teks += f"Model ini belum mengalahkan baseline naif (harga bulan depan = bulan ini, R2 {rb:.3f}). "
    teks += ("Harga antarbulan sangat stabil dan data hanya 6 bulan, sehingga prediksi ini bersifat "
             "indikatif, bukan peramalan andal. Hasil analisis utama menggunakan k = 3.")
    st.warning(teks)
