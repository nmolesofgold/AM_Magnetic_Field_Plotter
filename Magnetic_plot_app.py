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

# --- LIGHT MODE CONFIG ---
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] { background-color: #ffffff; }
        [data-testid="stSidebar"] { background-color: #f0f2f6; }
        [data-testid="stHeader"] { background-color: #ffffff; }
        h1, h2, h3, h4, p, label, .stMarkdown, div, span, .stMetricValue { color: #000000 !important; }
        [data-testid="stFileUploader"] section { background-color: #f8f9fa !important; border: 1px dashed #444 !important; }
        .stButton button, .stDownloadButton button { background-color: #ffffff !important; color: #000000 !important; border: 1px solid #000000 !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧲 Magnetic Field Analyzer")
st.caption("Developed by [Dr. Anmol Mahendra](https://www.linkedin.com/in/nmolesofgold/)")

# ==========================================
#           SESSION STATE MANAGEMENT
# ==========================================
# This function clears the generated ZIP whenever settings change
def reset_report():
    st.session_state.zip_ready = False
    st.session_state.zip_data = None

if "zip_ready" not in st.session_state:
    st.session_state.zip_ready = False
    st.session_state.zip_data = None

# ==========================================
#           SIDEBAR: CONFIGURATION
# ==========================================
with st.sidebar:
    st.header("1. Data Input")
    uploaded_file = st.file_uploader("Upload CSV Scan", type=["csv"], on_change=reset_report)
    
    st.header("2. Analysis Settings")
    cyl_radius = st.number_input("Reference Cylinder Radius (mm)", value=7.0, step=0.5, on_change=reset_report)
    cyl_length = st.number_input("Reference Cylinder Length (mm)", value=46.0, step=1.0, on_change=reset_report)
    
    st.header("3. Performance")
    max_3d_points = st.slider("Max 3D Points (Interactive)", 5000, 100000, 20000, 5000)

    st.divider()
    st.header("4. Publication Settings")
    # Triggers reset_report when changed
    pub_format = st.selectbox("File Format", ["PNG", "PDF", "SVG"], on_change=reset_report)
    
    if pub_format == "PNG":
        pub_dpi = st.select_slider("DPI (for PNG)", options=[300, 600, 1200], value=300, on_change=reset_report)
    else:
        pub_dpi = 300 # DPI doesn't matter much for vector formats but needed for savefig argument
    
    file_ext = pub_format.lower()
    mime_type = "application/pdf" if file_ext == "pdf" else f"image/{file_ext}"
    if file_ext == "svg": mime_type = "image/svg+xml"

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
        b = griddata((slice_df['X_mm']-cx, slice_df['Y_mm']-cy), slice_df['B_mT'], (0, 0), method='nearest')
        z_vals.append(float(b))
    
    peak_idx = np.argmax(z_vals)
    cz = unique_zs[peak_idx]
    peak_b = z_vals[peak_idx]
    
    df['X_rel'], df['Y_rel'], df['Z_rel'] = df['X_mm']-cx, df['Y_mm']-cy, df['Z_mm']-cz
    
    return df, {'peak_b': peak_b, 'cx': cx, 'cy': cy, 'cz': cz}, {'z_rels': unique_zs-cz, 'vals': np.array(z_vals)}

def calculate_stats(ppm_grid, gx, gy):
    def get_metrics(v):
        v = v[~np.isnan(v)]
        return [np.max(v)-np.min(v), np.min(v), np.max(v)] if len(v)>0 else [np.nan]*3
    
    r2 = gx**2 + gy**2
    m1 = get_metrics(ppm_grid)
    m2 = get_metrics(ppm_grid[r2<=4])
    m3 = get_metrics(ppm_grid[r2<=1])
    
    return pd.DataFrame({
        "Region": ["Whole Plane", "R < 2mm", "R < 1mm"], 
        "Pk-Pk (PPM)": [m1[0], m2[0], m3[0]], 
        "Min (PPM)": [m1[1], m2[1], m3[1]], 
        "Max (PPM)": [m1[2], m2[2], m3[2]]
    })

def get_cylinder_mesh(radius, length, z_center=0):
    z = np.linspace(z_center - length/2, z_center + length/2, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid)
    y_grid = radius * np.sin(theta_grid)
    return x_grid, y_grid, z_grid

def apply_black_axes(fig):
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True, showgrid=True, gridcolor='lightgrey', title_font=dict(color='black', size=14, family="Arial Black"), tickfont=dict(color='black', size=12))
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True, showgrid=True, gridcolor='lightgrey', title_font=dict(color='black', size=14, family="Arial Black"), tickfont=dict(color='black', size=12))
    fig.update_layout(font=dict(color='black'), paper_bgcolor='white', plot_bgcolor='white', title_font=dict(color='black', size=18, family="Arial Black"), legend_font=dict(color='black'))
    return fig

# ==========================================
#       STATIC EXPORT FUNCTIONS
# ==========================================
def set_mpl_style():
    plt.rcParams.update({'font.size': 12, 'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.linewidth': 2, 'xtick.major.width': 2, 'ytick.major.width': 2, 'axes.edgecolor': 'black', 'figure.facecolor': 'white', 'text.color': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'axes.facecolor': 'white'})

def get_mpl_img(fig, fmt, dpi):
    buf = io.BytesIO()
    # Explicitly set facecolor to white for PDF/SVG transparency issues
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    return buf.getvalue()

def create_static_profile(z_rels, vals, B0):
    f, ax = plt.subplots(figsize=(8, 5))
    ax.plot(z_rels, vals, 'k-o', linewidth=2, label="Profile")
    ax.plot(0, B0, 'r*', markersize=12, label="Max Field") 
    ax.set_title("Z-Profile"); ax.set_xlabel("Z (mm)"); ax.set_ylabel("Field (mT)")
    ax.grid(True, linestyle='--'); ax.legend()
    return f

def create_static_heatmap(gx, gy, gb, z_val):
    f, ax = plt.subplots(figsize=(6, 5))
    cp = ax.pcolormesh(gx, gy, gb, cmap='viridis')
    f.colorbar(cp, label='mT')
    ax.set_title(f"Field Distribution (Z={z_val:.1f}mm)")
    return f

def create_static_topology(gx, gy, gb, z_val):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(gx, gy, gb, cmap='viridis', edgecolor='none')
    fig.colorbar(surf, label='mT', shrink=0.7, pad=0.1)
    ax.set_title(f"3D Topology (Z={z_val:.1f}mm)")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("mT")
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0)); ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0)); ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    return fig

def create_static_homogeneity(x_l, y_l, z_rels, bx, by, z_vals, B0):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(x_l, (bx-B0)/B0*100, 'r-', linewidth=2)
    axes[0].set_title("X-Axis"); axes[0].set_ylabel("Deviation (%)")
    axes[1].plot(y_l, (by-B0)/B0*100, 'g-', linewidth=2)
    axes[1].set_title("Y-Axis"); axes[1].set_xlabel("Position (mm)")
    axes[2].plot(z_rels, (z_vals-B0)/B0*100, 'b-', linewidth=2)
    axes[2].set_title("Z-Axis")
    for ax in axes:
        ax.grid(True, linestyle=':', alpha=0.6, color='gray')
        for spine in ax.spines.values(): spine.set_linewidth(2)
    plt.tight_layout()
    return fig

def create_static_3d_cloud(df, centers):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # NO DOWNSAMPLING: Use full dataset
    p = ax.scatter(df['X_rel'], df['Y_rel'], df['Z_rel'], c=df['B_mT'], cmap='viridis', s=0.5, alpha=0.6)
    fig.colorbar(p, label='Field (mT)', pad=0.1, shrink=0.7)
    
    # Markers (No Arrow/Text)
    ax.scatter(0, 0, 0, color='red', s=50, marker='o', label='Center')
    
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.set_title("3D Magnetic Field Cloud")
    
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    return fig

# ==========================================
#           MAIN APP LOGIC
# ==========================================
if uploaded_file:
    df, meta, prof = load_and_process_data(uploaded_file)
    B0 = meta['peak_b']

    # --- PRE-CALCULATIONS ---
    z0_val = prof['z_rels'][np.argmin(np.abs(prof['z_rels']))]
    s_df = df[df['Z_rel'] == z0_val]
    gx, gy = np.mgrid[s_df['X_rel'].min():s_df['X_rel'].max():100j, s_df['Y_rel'].min():s_df['Y_rel'].max():100j]
    gb = griddata((s_df['X_rel'], s_df['Y_rel']), s_df['B_mT'], (gx, gy), method='linear')
    
    x_l = np.linspace(s_df['X_rel'].min(), s_df['X_rel'].max(), 100)
    y_l = np.linspace(s_df['Y_rel'].min(), s_df['Y_rel'].max(), 100)
    bx = griddata((s_df['X_rel'], s_df['Y_rel']), s_df['B_mT'], (x_l, np.zeros_like(x_l)), method='linear')
    by = griddata((s_df['X_rel'], s_df['Y_rel']), s_df['B_mT'], (np.zeros_like(y_l), y_l), method='linear')

    stats = calculate_stats((gb-B0)/B0*1e6, gx, gy)
    ppm_val = stats.loc[stats['Region'] == 'R < 1mm', 'Pk-Pk (PPM)'].values[0]

    # --- SIDEBAR: DOWNLOAD MANAGER ---
    # Force reset if user changes formats in the middle of a session
    if st.sidebar.button(f"🚀 Generate {pub_format} Report"):
        with st.sidebar.status(f"Generating {pub_format} Plots...", expanded=True):
            set_mpl_style()
            zip_buf = io.BytesIO()
            
            # Re-generate figures freshly every time button is clicked
            # This ensures they match the CURRENT pub_format
            st.write("Processing Profile...")
            f1 = create_static_profile(prof['z_rels'], prof['vals'], B0)
            
            st.write("Processing Heatmaps...")
            f2 = create_static_heatmap(gx, gy, gb, z0_val)
            f3 = create_static_topology(gx, gy, gb, z0_val)
            
            st.write("Processing Homogeneity...")
            f4 = create_static_homogeneity(x_l, y_l, prof['z_rels'], bx, by, prof['vals'], B0)
            
            st.write("Processing 3D Cloud...")
            f5 = create_static_3d_cloud(df, meta)

            with zipfile.ZipFile(zip_buf, "w") as zf:
                zf.writestr(f"1_profile.{file_ext}", get_mpl_img(f1, file_ext, pub_dpi))
                zf.writestr(f"2_heatmap.{file_ext}", get_mpl_img(f2, file_ext, pub_dpi))
                zf.writestr(f"3_topology_3d.{file_ext}", get_mpl_img(f3, file_ext, pub_dpi))
                zf.writestr(f"4_homogeneity.{file_ext}", get_mpl_img(f4, file_ext, pub_dpi))
                zf.writestr(f"5_cloud_3d.{file_ext}", get_mpl_img(f5, file_ext, pub_dpi))
            
            st.session_state.zip_data = zip_buf.getvalue()
            st.session_state.zip_ready = True
            st.write("Done!")

    if st.session_state.zip_ready:
        st.sidebar.download_button(
            label=f"📦 Download ZIP ({file_ext.upper()})", 
            data=st.session_state.zip_data, 
            file_name=f"Magnetic_Analysis_{file_ext}.zip",
            mime="application/zip"
        )

    # --- TABS ---
    t1, t2, t3, t4 = st.tabs(["Overview", "3D Cloud", "Slice Viewer", "Homogeneity"])

    with t1:
        c1, c2 = st.columns(2)
        c1.metric("Peak Field (B0)", f"{B0:.2f} mT")
        c2.metric("Homogeneity (R<1mm)", f"{ppm_val:.1f} PPM")
        
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=prof['z_rels'], y=prof['vals'], mode='lines+markers', name='Field Profile', line=dict(color='black', width=3), marker=dict(color='black', size=6)))
        fig_p.add_trace(go.Scatter(x=[0], y=[B0], mode='markers', name='Max Field', marker=dict(color='red', size=14, symbol='star')))
        fig_p.update_layout(title="Magnetic Field Profile", xaxis_title="Z (mm)", yaxis_title="mT")
        fig_p = apply_black_axes(fig_p)
        st.plotly_chart(fig_p, width='stretch')
        
        # Live generation for single button
        f1_live = create_static_profile(prof['z_rels'], prof['vals'], B0)
        st.download_button(f"💾 Download Profile ({file_ext.upper()})", get_mpl_img(f1_live, file_ext, pub_dpi), f"Profile.{file_ext}", mime_type)

    with t2:
        if len(df) > max_3d_points:
            step = len(df) // max_3d_points
            df_3d = df.iloc[::step]
            st.warning(f"Displaying {len(df_3d):,} points (Downsampled). Download uses full data.")
        else:
            df_3d = df

        fig_3d = go.Figure()
        cx_mesh, cy_mesh, cz_mesh = get_cylinder_mesh(cyl_radius, cyl_length, 0)
        fig_3d.add_trace(go.Surface(x=cx_mesh, y=cy_mesh, z=cz_mesh, opacity=0.2, colorscale='Greys', showscale=False, name='Cylinder', showlegend=False))
        fig_3d.add_trace(go.Scatter3d(x=df_3d['X_rel'], y=df_3d['Y_rel'], z=df_3d['Z_rel'], mode='markers', marker=dict(size=2, color=df_3d['B_mT'], colorscale='Viridis', showscale=True, opacity=0.8, colorbar=dict(title="mT")), name="Scan"))
        fig_3d.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=5, color='red'), showlegend=False))
        fig_3d.add_trace(go.Scatter3d(x=[0, 15], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='red', width=6), showlegend=False))
        fig_3d.add_trace(go.Cone(x=[15], y=[0], z=[0], u=[5], v=[0], w=[0], showscale=False, colorscale=[[0,'red'],[1,'red']], sizemode="absolute", sizeref=2, showlegend=False))
        fig_3d.add_trace(go.Scatter3d(x=[15], y=[0], z=[0], mode='text', text=[f"{B0:.2f} mT"], textposition="top center", textfont=dict(color="black", size=12, family="Arial Black"), showlegend=False))
        fig_3d.update_layout(height=800, scene=dict(xaxis=dict(title="X", backgroundcolor="white", color="black"), yaxis=dict(title="Y", backgroundcolor="white", color="black"), zaxis=dict(title="Z", backgroundcolor="white", color="black"), aspectmode='data'), paper_bgcolor='white', font=dict(color="black"), showlegend=True)
        st.plotly_chart(fig_3d, width='stretch')
        
        if st.button("Generate Full 3D Plot for Download"):
             f5_live = create_static_3d_cloud(df, meta)
             st.download_button(f"💾 Download 3D Cloud ({file_ext.upper()})", get_mpl_img(f5_live, file_ext, pub_dpi), f"3D_Cloud.{file_ext}", mime_type)

    with t3:
        sel_z = st.select_slider("Select Z Slice (mm)", options=np.sort(df['Z_rel'].unique()), value=z0_val)
        curr_slice = df[df['Z_rel'] == sel_z]
        gx_s, gy_s = np.mgrid[curr_slice['X_rel'].min():curr_slice['X_rel'].max():60j, curr_slice['Y_rel'].min():curr_slice['Y_rel'].max():60j]
        gb_s = griddata((curr_slice['X_rel'], curr_slice['Y_rel']), curr_slice['B_mT'], (gx_s, gy_s), method='linear')

        center_val = griddata((curr_slice['X_rel'], curr_slice['Y_rel']), curr_slice['B_mT'], (0, 0), method='linear')
        if np.isnan(center_val): center_val = griddata((curr_slice['X_rel'], curr_slice['Y_rel']), curr_slice['B_mT'], (0, 0), method='nearest')
        
        f_h_static = create_static_heatmap(gx_s, gy_s, gb_s, sel_z)
        f_t_static = create_static_topology(gx_s, gy_s, gb_s, sel_z)

        c1, c2 = st.columns(2)
        with c1:
            fig_h = go.Figure(go.Heatmap(x=gx_s[:,0], y=gy_s[0,:], z=gb_s, colorscale='Viridis'))
            fig_h.add_trace(go.Scatter(x=[0], y=[0], mode='markers+text', marker=dict(color='black', size=15, symbol='cross-thin', line=dict(color='white', width=2)), text=[f"{float(center_val):.2f} mT"], textposition="top center", textfont=dict(color='black', size=14, family="Arial Black"), name="Center"))
            fig_h.update_layout(title=f"2D Heatmap (Z={sel_z:.1f})", xaxis_title="X", yaxis_title="Y", width=500, height=500, xaxis=dict(scaleanchor="y", scaleratio=1), yaxis=dict(constrain='domain'))
            fig_h = apply_black_axes(fig_h)
            st.plotly_chart(fig_h, width='stretch')
            st.download_button(f"💾 Download Heatmap ({file_ext.upper()})", get_mpl_img(f_h_static, file_ext, pub_dpi), f"Heatmap_Z{sel_z:.1f}.{file_ext}", mime_type)

        with c2:
            fig_s = go.Figure(go.Surface(z=gb_s, x=gx_s, y=gy_s, colorscale='Viridis'))
            fig_s.add_trace(go.Scatter3d(x=[0], y=[0], z=[float(center_val)], mode='markers+text', marker=dict(color='red', size=5, symbol='circle'), text=[f"{float(center_val):.2f} mT"], textposition="top center", textfont=dict(color='black', size=12, family="Arial Black"), name="Center"))
            fig_s.update_layout(title="3D Topology", scene=dict(xaxis=dict(title="X", backgroundcolor="white", color="black"), yaxis=dict(title="Y", backgroundcolor="white", color="black"), zaxis=dict(title="mT", backgroundcolor="white", color="black"), aspectratio=dict(x=1, y=1, z=0.7)), width=500, height=500, paper_bgcolor='white', font=dict(color="black"))
            st.plotly_chart(fig_s, width='stretch')
            st.download_button(f"💾 Download Topology ({file_ext.upper()})", get_mpl_img(f_t_static, file_ext, pub_dpi), f"Topology_Z{sel_z:.1f}.{file_ext}", mime_type)

    with t4:
        st.subheader("Field Homogeneity (All Axes)")
        fig_hom = make_subplots(rows=1, cols=3, subplot_titles=("X-Axis", "Y-Axis", "Z-Axis"))
        fig_hom.add_trace(go.Scatter(x=x_l, y=(bx-B0)/B0*100, name="X", line=dict(color='red', width=2)), row=1, col=1)
        fig_hom.add_trace(go.Scatter(x=y_l, y=(by-B0)/B0*100, name="Y", line=dict(color='green', width=2)), row=1, col=2)
        fig_hom.add_trace(go.Scatter(x=prof['z_rels'], y=(prof['vals']-B0)/B0*100, name="Z", line=dict(color='blue', width=2)), row=1, col=3)
        fig_hom.update_layout(title="Deviation from Peak (%)", showlegend=False)
        fig_hom = apply_black_axes(fig_hom)
        fig_hom.update_xaxes(title_text="Position (mm)", color="black")
        fig_hom.update_yaxes(title_text="Deviation %", color="black", row=1, col=1)
        st.plotly_chart(fig_hom, width='stretch')
        
        f4_live = create_static_homogeneity(x_l, y_l, prof['z_rels'], bx, by, prof['vals'], B0)
        st.download_button(f"💾 Download Homogeneity ({file_ext.upper()})", get_mpl_img(f4_live, file_ext, pub_dpi), f"Homogeneity.{file_ext}", mime_type)
        
        st.subheader("Homogeneity Statistics (PPM)")
        st.dataframe(stats.style.format("{:.2f}", subset=["Pk-Pk (PPM)", "Min (PPM)", "Max (PPM)"]), hide_index=True)

else:
    st.info("Awaiting CSV file upload...")

