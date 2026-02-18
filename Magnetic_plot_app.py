import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import io
import zipfile

# ==========================================
#           STREAMLIT PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Magnetic Field Analyzer", 
    page_icon="🧲", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FORCE LIGHT MODE CSS ---
st.markdown("""
    <style>
        :root { --primary-color: #ff4b4b; --background-color: #ffffff; --text-color: #000000; }
        .stApp { background-color: #ffffff; color: #000000; }
        [data-testid="stFileUploader"] section { background-color: #f0f2f6 !important; color: #000000 !important; }
        .stDownloadButton button { border: 2px solid #d6d6d6 !important; font-weight: bold !important; color: black !important; }
        h1, h2, h3, p, label { color: #000000 !important; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧲 Magnetic Field Analyzer")
st.caption("Developed by Dr. Anmol Mahendra")

# ==========================================
#           SIDEBAR: CONFIGURATION
# ==========================================
with st.sidebar:
    st.header("1. Data Input")
    uploaded_file = st.file_uploader("Upload CSV Scan", type=["csv"])
    
    st.header("2. Analysis Settings")
    cyl_radius = st.number_input("Reference Cylinder Radius (mm)", value=7.0, step=0.5)
    cyl_length = st.number_input("Reference Cylinder Length (mm)", value=46.0, step=1.0)
    
    st.header("3. Publication Settings")
    pub_format = st.selectbox("File Format", ["PNG", "PDF", "SVG"])
    pub_dpi = st.select_slider("DPI (for PNG)", options=[300, 600, 1200], value=300) if pub_format == "PNG" else 300
    
    file_ext = pub_format.lower()
    mime_type = "application/pdf" if file_ext == "pdf" else f"image/{file_ext}"

# ==========================================
#           DATA PROCESSING
# ==========================================
@st.cache_data
def load_and_process_data(file):
    df = pd.read_csv(file)
    if 'Field' in df.columns: df.rename(columns={'Field': 'Magnetic_Field_Reading'}, inplace=True)
    df['B_mT'] = df['Magnetic_Field_Reading'] * 1000
    cx, cy = (df['X_mm'].max() + df['X_mm'].min()) / 2, (df['Y_mm'].max() + df['Y_mm'].min()) / 2
    
    # Simple peak finding
    unique_zs = np.sort(df['Z_mm'].unique())
    z_vals = []
    for z in unique_zs:
        slice_df = df[df['Z_mm'] == z]
        if len(slice_df) < 4: continue
        b = griddata((slice_df['X_mm']-cx, slice_df['Y_mm']-cy), slice_df['B_mT'], (0, 0), method='nearest')
        z_vals.append(b)
    
    cz = unique_zs[np.argmax(z_vals)]
    df['X_rel'], df['Y_rel'], df['Z_rel'] = df['X_mm']-cx, df['Y_mm']-cy, df['Z_mm']-cz
    return df, {'peak_b': np.max(z_vals), 'z_rels': unique_zs-cz, 'vals': z_vals}

# ==========================================
#       STATIC EXPORT HELPERS
# ==========================================
def get_mpl_img(fig, fmt, dpi):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()

# ==========================================
#           MAIN APP LOGIC
# ==========================================
if uploaded_file:
    df, meta, prof = load_and_process_data(uploaded_file)
    
    # DOWNLOAD ALL - Optimized for Memory
    if st.sidebar.button("📦 Download All Plots (.zip)"):
        with st.spinner("Generating High-Res Files..."):
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as zf:
                # Plot 1: Profile
                f1, ax1 = plt.subplots(); ax1.plot(prof['z_rels'], prof['vals'], 'k-'); ax1.set_title("Z-Profile")
                zf.writestr(f"profile.{file_ext}", get_mpl_img(f1, file_ext, pub_dpi))
                # Add more plots as needed here following same pattern
            st.sidebar.download_button("Save ZIP", zip_buf.getvalue(), "Analysis.zip")

    t1, t2, t3 = st.tabs(["Overview", "3D View", "Slices"])

    with t1:
        st.metric("Peak Field", f"{meta['peak_b']:.2f} mT")
        fig_p = go.Figure(go.Scatter(x=prof['z_rels'], y=prof['vals'], line=dict(color='black', width=3)))
        fig_p.update_layout(template="simple_white", title="Z-Axis Profile", xaxis_title="Z (mm)", yaxis_title="mT")
        st.plotly_chart(fig_p, width='stretch') # FIXED: Using width='stretch' for 2026

    with t2:
        fig_3d = go.Figure(go.Scatter3d(x=df['X_rel'], y=df['Y_rel'], z=df['Z_rel'], mode='markers', 
                                       marker=dict(size=2, color=df['B_mT'], colorscale='Viridis', showscale=True)))
        # Arrow and arrowhead
        fig_3d.add_trace(go.Scatter3d(x=[0, 15], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='red', width=6)))
        fig_3d.add_trace(go.Cone(x=[15], y=[0], z=[0], u=[5], v=[0], w=[0], showscale=False, colorscale=[[0,'red'],[1,'red']]))
        fig_3d.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"), paper_bgcolor='white')
        st.plotly_chart(fig_3d, width='stretch')

    with t3:
        sel_z = st.select_slider("Z Position", options=np.sort(df['Z_rel'].unique()), value=0.0)
        s_df = df[df['Z_rel'] == sel_z]
        fig_h = go.Figure(go.Heatmap(x=s_df['X_rel'], y=s_df['Y_rel'], z=s_df['B_mT'], colorscale='Viridis'))
        fig_h.update_layout(title=f"Slice at Z={sel_z}mm", xaxis_title="X", yaxis_title="Y")
        st.plotly_chart(fig_h, width='stretch')

else:
    st.info("Please upload a CSV file to begin.")
