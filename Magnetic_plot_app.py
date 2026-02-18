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
    page_title="Halbach Magnetic Scanner", 
    page_icon="🧲", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FORCE LIGHT MODE CSS ---
st.markdown("""
    <style>
        /* Force Global Light Theme */
        :root {
            --primary-color: #ff4b4b;
            --background-color: #ffffff;
            --secondary-background-color: #f0f2f6;
            --text-color: #31333F;
            --font: sans-serif;
        }
        
        /* Main App Background */
        .stApp {
            background-color: #ffffff;
            color: #31333F;
        }
        
        /* File Uploader - Force Light Mode */
        [data-testid="stFileUploader"] {
            background-color: #ffffff;
            border-radius: 5px;
            padding: 10px;
        }
        [data-testid="stFileUploader"] section {
            background-color: #f0f2f6 !important; 
            color: #31333F !important;
        }
        [data-testid="stFileUploader"] span {
            color: #31333F !important;
        }
        [data-testid="stFileUploader"] button {
            color: #31333F !important;
        }

        /* Download Buttons - Force Light Mode */
        .stDownloadButton button {
            background-color: #ffffff !important;
            color: #31333F !important;
            border: 1px solid #d6d6d6 !important;
        }
        .stDownloadButton button:hover {
            border-color: #ff4b4b !important;
            color: #ff4b4b !important;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #f0f2f6;
        }
        
        /* General Text */
        h1, h2, h3, h4, h5, h6, p, div, span, label, li {
            color: #31333F !important;
        }
        
        /* Metrics & Inputs */
        [data-testid="stMetricValue"] { color: #31333F !important; }
        .stNumberInput input { color: #31333F !important; background-color: white !important; }
        .stSelectbox div[data-baseweb="select"] > div { background-color: white !important; color: #31333F !important; }
        
        /* DataFrame */
        [data-testid="stDataFrame"] { background-color: white !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧲 Halbach Magnetic Field Analyzer")
st.markdown("Interactive viewer with **Publication-Quality Export** (300 DPI, White Background).")

# ==========================================
#           SIDEBAR: CONFIGURATION
# ==========================================
with st.sidebar:
    st.header("1. Data Input")
    uploaded_file = st.file_uploader("Upload CSV Scan", type=["csv"])
    
    st.header("2. Analysis Settings")
    cyl_radius = st.number_input("Reference Cylinder Radius (mm)", value=7.0, step=0.5)
    cyl_length = st.number_input("Reference Cylinder Length (mm)", value=46.0, step=1.0)
    
    st.header("3. Performance")
    max_3d_points = st.slider("Max 3D Points (Interactive)", 10000, 200000, 50000, 10000)

    st.divider()
    st.header("4. Publication Settings")
    pub_format = st.selectbox("File Format", ["PNG (Raster)", "PDF (Vector)", "SVG (Vector)"])
    
    if "PNG" in pub_format:
        pub_dpi = st.select_slider("Resolution (DPI)", options=[300, 600, 1200], value=300)
    else:
        pub_dpi = 300 
        
    file_ext = pub_format.split()[0].lower()
    mime_type = f"image/{'svg+xml' if file_ext == 'svg' else file_ext}"
    if file_ext == 'pdf': mime_type = "application/pdf"

# ==========================================
#           DATA PROCESSING
# ==========================================
@st.cache_data
def load_and_process_data(file):
    df = pd.read_csv(file)
    if 'Field' in df.columns and 'Magnetic_Field_Reading' not in df.columns:
        df.rename(columns={'Field': 'Magnetic_Field_Reading'}, inplace=True)
    
    df['B_mT'] = df['Magnetic_Field_Reading'] * 1000
    
    center_x = (df['X_mm'].max() + df['X_mm'].min()) / 2
    center_y = (df['Y_mm'].max() + df['Y_mm'].min()) / 2
    
    df['X_temp'] = df['X_mm'] - center_x
    df['Y_temp'] = df['Y_mm'] - center_y
    unique_zs = np.sort(df['Z_mm'].unique())
    z_vals, z_abs = [], []
    for z in unique_zs:
        slice_df = df[df['Z_mm'] == z]
        if len(slice_df) < 4: continue
        b = griddata((slice_df['X_temp'], slice_df['Y_temp']), slice_df['B_mT'], (0, 0), method='linear')
        if np.isnan(b): b = griddata((slice_df['X_temp'], slice_df['Y_temp']), slice_df['B_mT'], (0, 0), method='nearest')
        z_vals.append(b); z_abs.append(z)
    
    peak_idx = np.argmax(z_vals)
    center_z = z_abs[peak_idx]
    peak_b = z_vals[peak_idx]
    
    df['X_rel'] = df['X_mm'] - center_x
    df['Y_rel'] = df['Y_mm'] - center_y
    df['Z_rel'] = df['Z_mm'] - center_z
    
    return df, {'x': center_x, 'y': center_y, 'z': center_z, 'peak_b': peak_b}, {'z_rels': np.array(z_abs) - center_z, 'vals': np.array(z_vals)}

def calculate_stats(ppm_grid, gx, gy):
    def get_metrics(v):
        v = v[~np.isnan(v)]
        return [np.max(v)-np.min(v), np.min(v), np.max(v)] if len(v)>0 else [np.nan]*3
    r2 = gx**2 + gy**2
    m1, m2, m3 = get_metrics(ppm_grid), get_metrics(ppm_grid[r2<=4]), get_metrics(ppm_grid[r2<=1])
    return pd.DataFrame({
        "Region": ["Whole Plane", "R < 2mm", "R < 1mm"], 
        "Pk-Pk (PPM)": [m1[0], m2[0], m3[0]], 
        "Min (PPM)": [m1[1], m2[1], m3[1]], 
        "Max (PPM)": [m1[2], m2[2], m3[2]]
    })

# ==========================================
#       STATIC EXPORT FUNCTIONS (MATPLOTLIB)
# ==========================================
def convert_fig_to_bytes(fig, fmt, dpi):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    return buf.getvalue()

def create_static_profile(z_rels, vals, peak_b):
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif', 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'text.color': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'axes.edgecolor': 'black'})
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(z_rels, vals, 'k-o', markersize=4, label='Field Profile')
    ax.plot(0, peak_b, 'r*', markersize=12, label='Max Field')
    ax.set_xlabel("Z Relative (mm)")
    ax.set_ylabel("Magnetic Field (mT)")
    ax.set_title("Z-Axis Field Profile")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5, color='gray')
    return fig

def create_static_3d(df, centers):
    plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif', 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'text.color': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'axes.edgecolor': 'black'})
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(df['X_rel'], df['Y_rel'], df['Z_rel'], c=df['B_mT'], cmap='viridis', s=0.5, alpha=0.6)
    cbar = fig.colorbar(p, label='Field (mT)', pad=0.1, shrink=0.7)
    
    ax.scatter(0, 0, 0, color='red', s=50)
    ax.quiver(0, 0, 0, 15, 0, 0, color='red', length=15)
    ax.text(15, 0, 0, f"  {centers['peak_b']:.2f} mT", color='black', fontweight='bold')
    
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.set_title("3D Magnetic Field Cloud")
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    return fig

def create_static_heatmap(gx, gy, gb, z_pos, center_val):
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif', 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'text.color': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'axes.edgecolor': 'black'})
    fig, ax = plt.subplots(figsize=(7, 6))
    c = ax.pcolormesh(gx, gy, gb, cmap='viridis', shading='auto')
    cbar = fig.colorbar(c, label=f'mT (Center: {center_val:.2f})')
    ax.plot(0, 0, 'k+', markersize=10, markeredgewidth=2)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    ax.set_title(f"Field Distribution at Z={z_pos:.2f}mm")
    ax.set_aspect('equal')
    return fig

def create_static_topology(gx, gy, gb, z_pos):
    """Creates a static 3D surface plot for the field topology."""
    plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif', 'figure.facecolor': 'white', 'axes.facecolor': 'white'})
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(gx, gy, gb, cmap='viridis', linewidth=0, antialiased=True)
    cbar = fig.colorbar(surf, label='Field (mT)', pad=0.1, shrink=0.7)
    ax.set_title(f"Field Topology at Z={z_pos:.2f}mm")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Field (mT)")
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    return fig

def create_static_homogeneity(x_l, y_l, z_rels, bx, by, z_vals, B0):
    plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif', 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'text.color': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'axes.edgecolor': 'black'})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(x_l, (bx-B0)/B0*100, 'r-', linewidth=1.5)
    axes[0].set_title("X-Axis (Y=0, Z=0)")
    axes[0].set_ylabel("Deviation (%)")
    
    axes[1].plot(y_l, (by-B0)/B0*100, 'g-', linewidth=1.5)
    axes[1].set_title("Y-Axis (X=0, Z=0)")
    axes[1].set_xlabel("Position (mm)")
    
    axes[2].plot(z_rels, (z_vals-B0)/B0*100, 'b-', linewidth=1.5)
    axes[2].set_title("Z-Axis (X=0, Y=0)")
    
    for ax in axes:
        ax.grid(True, linestyle=':', alpha=0.6, color='gray')
        
    plt.tight_layout()
    return fig

# ==========================================
#           MAIN APP LOGIC
# ==========================================
if uploaded_file is not None:
    df, centers, profile_data = load_and_process_data(uploaded_file)
    
    # --- GLOBAL DOWNLOAD ALL BUTTON ---
    st.sidebar.divider()
    if st.sidebar.button(f"📦 Download All Plots (.zip)"):
        with st.spinner(f"Generating {file_ext.upper()} report..."):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                # 1. Z-Profile
                fig1 = create_static_profile(profile_data['z_rels'], profile_data['vals'], centers['peak_b'])
                zip_file.writestr(f"1_Z_Profile.{file_ext}", convert_fig_to_bytes(fig1, file_ext, pub_dpi))
                
                # 2. 3D Plot
                fig2 = create_static_3d(df, centers)
                zip_file.writestr(f"2_3D_Cloud.{file_ext}", convert_fig_to_bytes(fig2, file_ext, pub_dpi))
                
                # Data for slices
                z0_slice = df.iloc[(df['Z_rel'] - 0).abs().argsort()[:1]]
                z0_val = z0_slice['Z_rel'].values[0]
                sd = df[df['Z_rel'] == z0_val]
                gx, gy = np.mgrid[sd['X_rel'].min():sd['X_rel'].max():100j, sd['Y_rel'].min():sd['Y_rel'].max():100j]
                gb = griddata((sd['X_rel'], sd['Y_rel']), sd['B_mT'], (gx, gy), method='linear')
                
                # 3. Heatmap
                fig3 = create_static_heatmap(gx, gy, gb, z0_val, centers['peak_b'])
                zip_file.writestr(f"3_Heatmap_Center.{file_ext}", convert_fig_to_bytes(fig3, file_ext, pub_dpi))
                
                # 4. Topology (NEW)
                fig_topo = create_static_topology(gx, gy, gb, z0_val)
                zip_file.writestr(f"4_Topology_Center.{file_ext}", convert_fig_to_bytes(fig_topo, file_ext, pub_dpi))
                
                # 5. Homogeneity
                z0_s = df[df['Z_rel'] == z0_val]
                x_l = np.linspace(z0_s['X_rel'].min(), z0_s['X_rel'].max(), 100)
                y_l = np.linspace(z0_s['Y_rel'].min(), z0_s['Y_rel'].max(), 100)
                bx = griddata((z0_s['X_rel'], z0_s['Y_rel']), z0_s['B_mT'], (x_l, np.zeros_like(x_l)), method='linear')
                by = griddata((z0_s['X_rel'], z0_s['Y_rel']), z0_s['B_mT'], (np.zeros_like(y_l), y_l), method='linear')
                fig4 = create_static_homogeneity(x_l, y_l, profile_data['z_rels'], bx, by, profile_data['vals'], centers['peak_b'])
                zip_file.writestr(f"5_Homogeneity.{file_ext}", convert_fig_to_bytes(fig4, file_ext, pub_dpi))

            st.sidebar.download_button(
                label=f"Click to Save ZIP ({file_ext.upper()})", 
                data=zip_buffer.getvalue(), 
                file_name=f"Magnetic_Analysis_{file_ext}.zip", 
                mime="application/zip"
            )

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "3D Visualization", "Slice Viewer", "Homogeneity"])
    
    # --- TAB 1: OVERVIEW ---
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Peak Field (B0)", f"{centers['peak_b']:.2f} mT")
        c2.metric("Center X", f"{centers['x']:.2f} mm")
        c3.metric("Center Y", f"{centers['y']:.2f} mm")
        c4.metric("Center Z", f"{centers['z']:.2f} mm")
        
        # Interactive
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=profile_data['z_rels'], y=profile_data['vals'], mode='lines+markers', name='Field', line=dict(color='black'), marker=dict(color='black')))
        fig_p.add_trace(go.Scatter(x=[0], y=[centers['peak_b']], mode='markers', marker=dict(color='red', size=12, symbol='star'), name='Max'))
        fig_p.update_layout(
            title="Z-Axis Field Profile", xaxis_title="Z (mm)", yaxis_title="B (mT)", 
            template="simple_white", height=400,
            paper_bgcolor='white', plot_bgcolor='white'
        )
        st.plotly_chart(fig_p, use_container_width=True)
        
        fig_static = create_static_profile(profile_data['z_rels'], profile_data['vals'], centers['peak_b'])
        st.download_button(f"💾 Download {file_ext.upper()}", convert_fig_to_bytes(fig_static, file_ext, pub_dpi), f"Z_Profile.{file_ext}", mime_type)

    # --- TAB 2: 3D VISUALIZATION ---
    with tab2:
        if len(df) > max_3d_points:
            step = len(df) // max_3d_points
            df_3d = df.iloc[::step]
            st.warning(f"Previewing {len(df_3d):,} points. Download will use FULL data.")
        else:
            df_3d = df

        fig_3d = go.Figure()
        
        # Scatter with Colorbar
        fig_3d.add_trace(go.Scatter3d(
            x=df_3d['X_rel'], y=df_3d['Y_rel'], z=df_3d['Z_rel'], 
            mode='markers', 
            marker=dict(
                size=2, 
                color=df_3d['B_mT'], 
                colorscale='Viridis', 
                opacity=0.6, 
                showscale=True, 
                colorbar=dict(title="Field (mT)")
            ), 
            name="Scan"
        ))
        
        # Center Marker
        fig_3d.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=6, color='red'), name="Center"))
        
        # Arrow Line
        fig_3d.add_trace(go.Scatter3d(x=[0, 15], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='red', width=5), name="Axis"))
        
        # Arrowhead
        fig_3d.add_trace(go.Cone(
            x=[15], y=[0], z=[0], 
            u=[5], v=[0], w=[0], 
            sizemode="absolute", sizeref=4, anchor="tail", 
            showscale=False, colorscale=[[0, 'red'], [1, 'red']], 
            name="Arrowhead"
        ))
        
        # Text Label
        fig_3d.add_trace(go.Scatter3d(x=[15], y=[0], z=[0], mode='text', text=[f"{centers['peak_b']:.2f} mT"], textposition="top right"))
        
        fig_3d.update_layout(
            scene=dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data',
                xaxis=dict(backgroundcolor="white", gridcolor="lightgrey", title=dict(font=dict(color='black')), tickfont=dict(color='black')),
                yaxis=dict(backgroundcolor="white", gridcolor="lightgrey", title=dict(font=dict(color='black')), tickfont=dict(color='black')),
                zaxis=dict(backgroundcolor="white", gridcolor="lightgrey", title=dict(font=dict(color='black')), tickfont=dict(color='black')),
            ),
            height=600, template="none", paper_bgcolor='white'
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        
        if st.button(f"💾 Generate {file_ext.upper()} Image"):
            with st.spinner("Rendering full point cloud..."):
                fig_static_3d = create_static_3d(df, centers)
                st.download_button(f"Click to Download 3D {file_ext.upper()}", convert_fig_to_bytes(fig_static_3d, file_ext, pub_dpi), f"3D_Scan_Full.{file_ext}", mime_type)

    # --- TAB 3: SLICE VIEWER ---
    with tab3:
        available_zs = np.sort(df['Z_rel'].unique())
        sel_z = st.select_slider("Select Z Position (mm)", options=available_zs, value=0.0)
        slice_d = df[df['Z_rel'] == sel_z]
        
        gx, gy = np.mgrid[slice_d['X_rel'].min():slice_d['X_rel'].max():100j, slice_d['Y_rel'].min():slice_d['Y_rel'].max():100j]
        gb = griddata((slice_d['X_rel'], slice_d['Y_rel']), slice_d['B_mT'], (gx, gy), method='linear')
        
        center_val = float(griddata((slice_d['X_rel'], slice_d['Y_rel']), slice_d['B_mT'], (0, 0), method='linear'))
        if np.isnan(center_val): center_val = 0.0

        c_hm, c_st = st.columns([2, 1])
        with c_hm:
            # 1. HEATMAP (2D)
            f_hm = go.Figure(go.Heatmap(x=np.unique(gx), y=np.unique(gy), z=gb, colorscale='Viridis', colorbar=dict(title=f"mT<br>Center: {center_val:.2f}")))
            f_hm.add_trace(go.Scatter(x=[0], y=[0], mode='markers+text', marker=dict(color='black', size=8, line=dict(color='white', width=1)), text=[f"{center_val:.2f}"], textposition="top center", textfont=dict(color='white')))
            f_hm.update_layout(title=f"Field Distribution at Z={sel_z:.2f}mm", xaxis_title="X", yaxis_title="Y", template="simple_white", paper_bgcolor='white', plot_bgcolor='white')
            st.plotly_chart(f_hm)
            
            fig_static_hm = create_static_heatmap(gx, gy, gb, sel_z, center_val)
            st.download_button(f"💾 Download Heatmap {file_ext.upper()}", convert_fig_to_bytes(fig_static_hm, file_ext, pub_dpi), f"Slice_Z_{sel_z:.1f}.{file_ext}", mime_type)

            # 2. TOPOLOGY (3D Surface) -- RESTORED
            st.markdown("### Field Topology (3D Surface)")
            f_topo = go.Figure(go.Surface(x=gx, y=gy, z=gb, colorscale='Viridis'))
            f_topo.update_layout(
                title=f"Field Topology at Z={sel_z:.2f}mm",
                scene=dict(
                    xaxis=dict(title='X (mm)', backgroundcolor="white", gridcolor="lightgrey"),
                    yaxis=dict(title='Y (mm)', backgroundcolor="white", gridcolor="lightgrey"),
                    zaxis=dict(title='Field (mT)', backgroundcolor="white", gridcolor="lightgrey"),
                ),
                height=600, template="none", paper_bgcolor='white'
            )
            st.plotly_chart(f_topo)
            
            fig_static_topo = create_static_topology(gx, gy, gb, sel_z)
            st.download_button(f"💾 Download Topology {file_ext.upper()}", convert_fig_to_bytes(fig_static_topo, file_ext, pub_dpi), f"Topology_Z_{sel_z:.1f}.{file_ext}", mime_type)

        with c_st:
            st.markdown("### Stats (PPM)")
            stats = calculate_stats((gb-centers['peak_b'])/centers['peak_b']*1e6, gx, gy)
            st.dataframe(stats.style.format("{:.2f}", subset=["Pk-Pk (PPM)", "Min (PPM)", "Max (PPM)"]), hide_index=True)

    # --- TAB 4: HOMOGENEITY ---
    with tab4:
        B0 = centers['peak_b']
        z0_s = df[df['Z_rel'] == df.iloc[(df['Z_rel'] - 0).abs().argsort()[:1]]['Z_rel'].values[0]]
        x_l = np.linspace(z0_s['X_rel'].min(), z0_s['X_rel'].max(), 100)
        y_l = np.linspace(z0_s['Y_rel'].min(), z0_s['Y_rel'].max(), 100)
        bx = griddata((z0_s['X_rel'], z0_s['Y_rel']), z0_s['B_mT'], (x_l, np.zeros_like(x_l)), method='linear')
        by = griddata((z0_s['X_rel'], z0_s['Y_rel']), z0_s['B_mT'], (np.zeros_like(y_l), y_l), method='linear')
        
        f_hom = make_subplots(rows=1, cols=3, subplot_titles=("X-Axis", "Y-Axis", "Z-Axis"))
        f_hom.add_trace(go.Scatter(x=x_l, y=(bx-B0)/B0*100, name="X", line=dict(color='red')), row=1, col=1)
        f_hom.add_trace(go.Scatter(x=y_l, y=(by-B0)/B0*100, name="Y", line=dict(color='green')), row=1, col=2)
        f_hom.add_trace(go.Scatter(x=profile_data['z_rels'], y=(profile_data['vals']-B0)/B0*100, name="Z", line=dict(color='blue')), row=1, col=3)
        f_hom.update_layout(template="simple_white", showlegend=False, paper_bgcolor='white', plot_bgcolor='white')
        f_hom.update_yaxes(title_text="Deviation (%)", row=1, col=1)
        st.plotly_chart(f_hom, use_container_width=True)
        
        fig_static_hom = create_static_homogeneity(x_l, y_l, profile_data['z_rels'], bx, by, profile_data['vals'], B0)
        st.download_button(f"💾 Download {file_ext.upper()}", convert_fig_to_bytes(fig_static_hom, file_ext, pub_dpi), f"Homogeneity.{file_ext}", mime_type)

else:
    st.info("Awaiting CSV file upload...")