import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ===============================
# Load Model & Data
# ===============================
@st.cache_resource
def load_model_file():
    return joblib.load("model_histgradientboosting.pkl")

@st.cache_data
def load_data_file():
    return pd.read_csv("depreciated-clear.csv")

# ===============================
# Setup Halaman
# ===============================
st.set_page_config(page_title="Prediksi Harga Mobil Bekas", layout="wide")
st.title("🚘 Prediksi Harga Mobil Bekas")
st.caption("Prediksi harga hingga 20 tahun ke depan dimulai dari tahun 2026")

st.info("""
ℹ️ **Catatan Penting**:
- Semua prediksi dimulai dari tahun **2025** sebagai tahun awal mobil bekas.
- Artinya, **Harga Saat Ini**, **Usia Mobil**, dan **Kilometer** dihitung mulai dari tahun 2025.
- Prediksi dilakukan selama 20 tahun ke depan hingga tahun 2045.
""")

with st.spinner("🔄 Memuat model dan data..."):
    try:
        model = load_model_file()
        df = load_data_file()
    except Exception as e:
        st.error(f"❌ Gagal memuat model atau data: {e}")
        st.stop()

# ===============================
# Sidebar Input
# ===============================
st.sidebar.header("🛠️ Masukkan Parameter Mobil")

# Brand dan Model
brand_list = sorted(df['Brand'].dropna().unique())
selected_brand = st.sidebar.selectbox("🔍 Pilih Brand", brand_list)

model_list = df[df['Brand'] == selected_brand]['Model'].dropna().unique()
selected_model = st.sidebar.selectbox("🚗 Pilih Model", sorted(model_list))

# Ambil default dari data sesuai brand dan model
default_row = df[(df['Brand'] == selected_brand) & (df['Model'] == selected_model)].iloc[0]

default_year = int(default_row['Year'])
default_fuel = default_row['Fuel_simple']
default_trans = default_row['Transmission_simple']
default_price = int(default_row['Price'])

# Estimasi km awal tahun 2025 jika memungkinkan
if default_row["Simulation_Kilometer"] > 0 and default_row["Vehicle_Age"] > 0:
    km_per_year = default_row["Simulation_Kilometer"] / default_row["Vehicle_Age"]
    default_kilometer = int(km_per_year * (2025 - default_year))
else:
    default_kilometer = 0

# Pilihan lain
fuel_options = sorted(df['Fuel_simple'].dropna().unique())
trans_options = sorted(df['Transmission_simple'].dropna().unique())
year_options = sorted(df['Year'].dropna().unique())

# Input
year = st.sidebar.selectbox("📆 Tahun Produksi", options=year_options, index=year_options.index(default_year))

kilometer_str = st.sidebar.text_input("🛣️ Kilometer Saat Ini (tahun 2025)", value=f"{default_kilometer:,}".replace(",", "."))
kilometer_cleaned = kilometer_str.replace(".", "").replace(",", "").strip()
kilometer = int(kilometer_cleaned) if kilometer_cleaned.isdigit() else default_kilometer

fuel = st.sidebar.selectbox("⛽ Tipe Bahan Bakar", options=fuel_options, index=fuel_options.index(default_fuel))
transmission = st.sidebar.selectbox("⚙️ Transmisi", options=trans_options, index=trans_options.index(default_trans))

price_str = st.sidebar.text_input("💰 Harga Saat Ini (Tahun 2025)", value=f"{default_price:,}".replace(",", "."))
cleaned_price = price_str.replace(".", "").replace(",", "").strip()
price = int(cleaned_price) if cleaned_price.isdigit() else default_price

# Tombol Prediksi
prediksi_button = st.sidebar.button("🔮 Prediksi Harga")

# ===============================
# Hasil Prediksi
# ===============================
if prediksi_button:
    st.markdown("## 🔮 Hasil Prediksi Harga Mobil")

    simulation_years = np.arange(2026, 2026 + 20)
    predictions = []

    for year_sim in simulation_years:
        usia = year_sim - 2025  # Usia mobil dihitung dari tahun 2025
        km_sim = kilometer + (usia * 5777) if usia >= 0 else kilometer

        input_row = pd.DataFrame([{
            'Brand': selected_brand,
            'Year': year,
            'Fuel_simple': fuel,
            'Transmission_simple': transmission,
            'Price': price,
            'Simulation_Year': year_sim,
            'Vehicle_Age': year_sim - year,
            'Simulation_Kilometer': km_sim
        }])

        pred = model.predict(input_row)[0]
        predictions.append(pred)

    df_pred = pd.DataFrame({
        "Tahun": simulation_years,
        "Usia Mobil": simulation_years - year,
        "Harga Prediksi (Rp)": [f"Rp {int(p):,}".replace(",", ".") for p in predictions]
    })

    st.caption(f"Tahun Produksi: {year} — Harga Saat Ini (2025): Rp {price:,.0f}".replace(",", "."))
    st.dataframe(df_pred, use_container_width=True)

    fig, ax = plt.subplots()
    ax.plot(simulation_years, predictions, marker='o', color='blue')
    ax.set_title("Prediksi Harga Mobil Bekas 2026–2045")
    ax.set_xlabel("Tahun")
    ax.set_ylabel("Harga (Rp)")
    ax.ticklabel_format(style='plain', axis='y')
    ax.grid(True)
    st.pyplot(fig)
