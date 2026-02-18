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

# --- SAFE LIGHT MODE CSS ---
# Only targets the main container and sidebar, NOT the plots themselves
st.markdown("""
    <style>
        /* Main App Background - White */
        [data-testid="stAppViewContainer"] {
            background-color: #ffffff;
        }
        /* Sidebar Background - Light Grey */
        [data-testid="stSidebar"] {
            background-color: #f0f2f6;
        }
        /* Text Color - Black (Targeted) */
        h1, h2, h3, h4, p, label, .stMarkdown {
            color: #000000 !important;
        }
        /* File Uploader - Grey Background */
        [data-testid="stFileUploader"] section {
            background-color: #f9f9f9;
        }
        /* Buttons - Visible Borders */
        .stButton button, .stDownloadButton button {
            color: #000000 !important;
            border: 1px solid #444 !important;
            background-color: #ffffff !important;
        }
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
    if 'Field' in df.columns: 
        df.rename(columns={'Field': 'Magnetic_Field_Reading'}, inplace=True)
    
    df['B_mT'] = df['Magnetic_Field_Reading'] * 1000
    cx, cy = (df['X_mm'].max() + df['X_mm'].min()) / 2, (df['Y_mm'].max() + df['Y_mm'].min()) / 2
    
    unique_zs = np.sort(df['Z_mm'].unique())
    z_vals = []
    for z in unique_zs:
        slice_df = df[df['Z_mm'] == z]
        # Use nearest interpolation to avoid NaNs in sparse data
        b = griddata((slice_df['X_mm']-cx, slice_df['Y_mm']-cy), slice_df['B_mT'], (0, 0), method='nearest')
        z_vals.append(float(b))
    
    peak_idx = np.argmax(z_vals)
    cz = unique_zs[peak_idx]
    peak_b = z_vals[peak_idx]
    
    df['X_rel'], df['Y_rel'], df['Z_rel'] = df['X_mm']-cx, df['Y_mm']-cy, df['Z_mm']-cz
    
    # Returns 3 items (DataFrame, Metadata, Profile Data)
    return df, {'peak_b': peak_b, 'cx': cx, 'cy': cy, 'cz': cz}, {'z_rels': unique_zs-cz, 'vals': np.array(z_vals)}

# ==========================================
#       STATIC EXPORT HELPERS
# ==========================================
def set_mpl_style():
    # Enforces bold, black lines for Matplotlib exports
    plt.rcParams.update({
        'font.size': 12, 'font.weight': 'bold', 'axes.labelweight': 'bold',
        'axes.linewidth': 2, 'xtick.major.width': 2, 'ytick.major.width': 2,
        'axes.edgecolor': 'black', 'figure.facecolor': 'white', 'text.color': 'black',
        'xtick.color': 'black', 'ytick.color': 'black'
    })

def get_mpl_img(fig, fmt, dpi):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return buf.getvalue()

# ==========================================
#           MAIN APP LOGIC
# ==========================================
if uploaded_file:
    df, meta, prof = load_and_process_data(uploaded_file)
    B0 = meta['peak_b']

    # Pre-calculate center slice for download/display
    z0_val = prof['z_rels'][np.argmin(np.abs(prof['z_rels']))] # Find Z closest to 0
    s_df = df[df['Z_rel'] == z0_val]
    # Create grid for heatmap
    gx, gy = np.mgrid[s_df['X_rel'].min():s_df['X_rel'].max():100j, s_df['Y_rel'].min():s_df['Y_rel'].max():100j]
    gb = griddata((s_df['X_rel'], s_df['Y_rel']), s_df['B_mT'], (gx, gy), method='linear')

    # --- SIDEBAR DOWNLOAD ALL ---
    if st.sidebar.button("📦 Download All Plots (.zip)"):
        with st.spinner("Generating publication files..."):
            set_mpl_style()
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as zf:
                # 1. Profile Plot
                f, ax = plt.subplots()
                ax.plot(prof['z_rels'], prof['vals'], 'k-o', linewidth=2)
                ax.set_title("Z-Profile (mT)"); ax.set_xlabel("Z (mm)"); ax.set_ylabel("Field (mT)")
                ax.grid(True, linestyle='--')
                zf.writestr(f"1_profile.{file_ext}", get_mpl_img(f, file_ext, pub_dpi))
                
                # 2. Heatmap Plot
                f, ax = plt.subplots()
                cp = ax.pcolormesh(gx, gy, gb, cmap='viridis')
                f.colorbar(cp, label='mT')
                ax.set_title(f"Field Distribution (Z={z0_val:.1f}mm)")
                zf.writestr(f"2_heatmap.{file_ext}", get_mpl_img(f, file_ext, pub_dpi))
                
                # 3. Homogeneity Plot
                f, ax = plt.subplots()
                ax.plot(prof['z_rels'], (prof['vals']-B0)/B0*100, 'b-', linewidth=2)
                ax.set_title("Homogeneity (%)"); ax.set_ylabel("Deviation (%)")
                ax.grid(True, linestyle='--')
                zf.writestr(f"3_homogeneity.{file_ext}", get_mpl_img(f, file_ext, pub_dpi))
                
            st.sidebar.download_button("Save ZIP", zip_buf.getvalue(), f"Magnetic_Analysis_{file_ext}.zip")

    # --- TABS ---
    t1, t2, t3, t4 = st.tabs(["Overview", "3D Cloud", "Slice Viewer", "Homogeneity"])

    with t1:
        st.metric("Peak Field (B0)", f"{B0:.2f} mT")
        
        # RESTORED: mode='lines+markers'
        fig_p = go.Figure(go.Scatter(
            x=prof['z_rels'], 
            y=prof['vals'], 
            mode='lines+markers',  # <-- MARKERS ADDED BACK
            line=dict(color='black', width=3),
            marker=dict(color='red', size=8)
        ))
        fig_p.update_layout(
            template="simple_white", 
            title="Magnetic Field Profile", 
            xaxis_title="Z (mm)", 
            yaxis_title="mT",
            font=dict(color="black", size=14)
        )
        st.plotly_chart(fig_p, width='stretch')

    with t2:
        # 3D Plot
        fig_3d = go.Figure()
        # Data Points
        fig_3d.add_trace(go.Scatter3d(
            x=df['X_rel'], y=df['Y_rel'], z=df['Z_rel'], 
            mode='markers', 
            marker=dict(size=3, color=df['B_mT'], colorscale='Viridis', showscale=True, opacity=0.8),
            name="Scan Data"
        ))
        # Center Marker
        fig_3d.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0], mode='markers', marker=dict(size=6, color='red'), name="Center"
        ))
        # Arrow
        fig_3d.add_trace(go.Scatter3d(
            x=[0, 15], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='red', width=6), name="Axis"
        ))
        fig_3d.add_trace(go.Cone(
            x=[15], y=[0], z=[0], u=[5], v=[0], w=[0], showscale=False, colorscale=[[0,'red'],[1,'red']], sizeref=2
        ))
        
        fig_3d.update_layout(
            scene=dict(
                xaxis=dict(title="X", backgroundcolor="white", color="black"),
                yaxis=dict(title="Y", backgroundcolor="white", color="black"),
                zaxis=dict(title="Z", backgroundcolor="white", color="black"),
            ),
            paper_bgcolor='white',
            font=dict(color="black")
        )
        st.plotly_chart(fig_3d, width='stretch')

    with t3:
        sel_z = st.select_slider("Select Z Slice (mm)", options=np.sort(df['Z_rel'].unique()), value=z0_val)
        
        # Re-grid for the SELECTED slice
        curr_slice = df[df['Z_rel'] == sel_z]
        gx_s, gy_s = np.mgrid[curr_slice['X_rel'].min():curr_slice['X_rel'].max():60j, curr_slice['Y_rel'].min():curr_slice['Y_rel'].max():60j]
        gb_s = griddata((curr_slice['X_rel'], curr_slice['Y_rel']), curr_slice['B_mT'], (gx_s, gy_s), method='linear')

        # 2D Heatmap
        fig_h = go.Figure(go.Heatmap(x=gx_s[:,0], y=gy_s[0,:], z=gb_s, colorscale='Viridis'))
        fig_h.update_layout(
            title=f"Field Distribution (Z={sel_z:.2f}mm)", 
            xaxis_title="X (mm)", 
            yaxis_title="Y (mm)",
            template="simple_white"
        )
        st.plotly_chart(fig_h, width='stretch')

        # 3D Topology Surface
        st.markdown("### Field Topology")
        fig_s = go.Figure(go.Surface(z=gb_s, x=gx_s, y=gy_s, colorscale='Viridis'))
        fig_s.update_layout(
            scene=dict(
                xaxis=dict(title="X"), yaxis=dict(title="Y"), zaxis=dict(title="mT"),
                bgcolor="white"
            ),
            paper_bgcolor='white',
            font=dict(color="black")
        )
        st.plotly_chart(fig_s, width='stretch')

    with t4:
        st.subheader("Field Homogeneity")
        fig_hom = go.Figure(go.Scatter(
            x=prof['z_rels'], 
            y=(prof['vals']-B0)/B0*100, 
            mode='lines+markers',
            line=dict(color='blue', width=2)
        ))
        fig_hom.update_layout(
            template="simple_white", 
            title="Deviation from Peak (%)", 
            xaxis_title="Z (mm)", 
            yaxis_title="Deviation %",
            font=dict(color="black")
        )
        st.plotly_chart(fig_hom, width='stretch')

else:
    st.info("Awaiting CSV file upload...")
