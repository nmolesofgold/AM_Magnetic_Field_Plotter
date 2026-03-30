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
#            STREAMLIT PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Magnetic Field Analyzer",
    page_icon="🧲",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.linkedin.com/in/nmolesofgold/',
        'Report a bug': "https://www.linkedin.com/in/nmolesofgold/",
        'About': "### Magnetic Field Analyzer\n**Developed by Dr. Anmol Mahendra**.\n\nA versatile tool for visualising magnetic field scans."
    }
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
#            SESSION STATE MANAGEMENT
# ==========================================
def reset_report():
    st.session_state.zip_ready = False
    st.session_state.zip_data = None

if "zip_ready" not in st.session_state:
    st.session_state.zip_ready = False
    st.session_state.zip_data = None

# ==========================================
#            SIDEBAR: CONFIGURATION
# ==========================================
with st.sidebar:
    st.header("1. Data Input")
    
    # Unit Selection
    c1, c2 = st.columns(2)
    with c1:
        dist_unit = st.selectbox("Distance Unit", ["mm", "cm", "m", "inches"], index=0, on_change=reset_report)
    with c2:
        field_unit = st.selectbox("Field Unit", ["milliTesla (mT)", "Tesla (T)", "Gauss (G)", "microTesla (µT)"], index=0, on_change=reset_report)

    with st.expander("ℹ️ File Format Instructions"):
        st.markdown("""
        **Required Header Row:**
        `X_mm,Y_mm,Z_mm,Field`
        
        *(Note: Keep the headers as X_mm, Y_mm, Z_mm even if your data is in cm or inches. Select your actual units in the dropdown above).*
        """)
        
        dummy_data = pd.DataFrame({
            'X_mm': [125.7, 125.7, 125.69], 
            'Y_mm': [-26.59, -26.58, -26.57], 
            'Z_mm': [-4, -4, -4], 
            'Field': [208.3, 208.3, 208.3]
        })
        csv_template = dummy_data.to_csv(index=False).encode('utf-8')
        st.download_button("📄 Download Template", csv_template, "magnetic_template.csv", "text/csv")

    # MULTI-FILE UPLOAD
    uploaded_files = st.file_uploader("Upload CSV Scan(s)", type=["csv"], accept_multiple_files=True, on_change=reset_report)
    
    st.header("2. Analysis Volume (Homogeneity)")
    vol_shape = st.selectbox("Target Volume Shape", ["Cylinder", "Sphere"], on_change=reset_report)
    vol_radius = st.number_input("Radius (mm)", value=7.0, step=0.5, on_change=reset_report)
    vol_length = 46.0  # Default hidden value to prevent errors
    
    if vol_shape == "Cylinder":
        vol_length = st.number_input("Length (mm)", value=46.0, step=1.0, on_change=reset_report)
    
    st.header("3. Performance")
    max_3d_points = st.slider("Max 3D Points (Interactive)", 5000, 100000, 20000, 5000)

    st.divider()
    st.header("4. Publication Settings")
    pub_format = st.selectbox("File Format", ["PNG", "PDF", "SVG"], on_change=reset_report)
    pub_dpi = st.select_slider("DPI (for PNG)", options=[300, 600, 1200], value=300) if pub_format == "PNG" else 300
    
    file_ext = pub_format.lower()
    mime_type = "application/pdf" if file_ext == "pdf" else f"image/{file_ext}"

# ==========================================
#            DATA PROCESSING
# ==========================================
@st.cache_data
def load_and_process_data(file_bytes, filename, dist_unit, field_unit):
    df = pd.read_csv(io.BytesIO(file_bytes))
    if 'Field' in df.columns: 
        df.rename(columns={'Field': 'Magnetic_Field_Reading'}, inplace=True)
    
    # Unit Conversions -> Background standard is mm and mT
    if field_unit == "Tesla (T)": df['B_mT'] = df['Magnetic_Field_Reading'] * 1000
    elif field_unit == "milliTesla (mT)": df['B_mT'] = df['Magnetic_Field_Reading']
    elif field_unit == "Gauss (G)": df['B_mT'] = df['Magnetic_Field_Reading'] * 0.1
    elif field_unit == "microTesla (µT)": df['B_mT'] = df['Magnetic_Field_Reading'] * 0.001

    dist_mult = 1.0
    if dist_unit == "cm": dist_mult = 10.0
    elif dist_unit == "m": dist_mult = 1000.0
    elif dist_unit == "inches": dist_mult = 25.4
    
    for col in ['X_mm', 'Y_mm', 'Z_mm']:
        if col in df.columns: df[col] = df[col] * dist_mult

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

def get_volume_mesh(shape, radius, length, z_center=0):
    if shape == "Cylinder":
        z = np.linspace(z_center - length/2, z_center + length/2, 50)
        theta = np.linspace(0, 2*np.pi, 50)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = radius * np.cos(theta_grid)
        y_grid = radius * np.sin(theta_grid)
        return x_grid, y_grid, z_grid
    else: # Sphere
        u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
        x_grid = radius * np.cos(u) * np.sin(v)
        y_grid = radius * np.sin(u) * np.sin(v)
        z_grid = z_center + radius * np.cos(v)
        return x_grid, y_grid, z_grid

def calc_vol_homogeneity(df, shape, r, l, b0):
    """Filters points inside the 3D volume and calculates Peak-to-Peak PPM."""
    if shape == "Cylinder":
        mask = (df['X_rel']**2 + df['Y_rel']**2 <= r**2) & (df['Z_rel'].abs() <= l/2)
    else:
        mask = (df['X_rel']**2 + df['Y_rel']**2 + df['Z_rel']**2 <= r**2)
    
    pts = df[mask]['B_mT']
    if len(pts) == 0: return np.nan, np.nan, np.nan
        
    pk_pk = pts.max() - pts.min()
    ppm = (pk_pk / b0) * 1e6
    return ppm, pts.min(), pts.max()

def apply_black_axes(fig):
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True, showgrid=True, gridcolor='lightgrey', title_font=dict(color='black', size=14, family="Arial Black"), tickfont=dict(color='black', size=12))
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True, showgrid=True, gridcolor='lightgrey', title_font=dict(color='black', size=14, family="Arial Black"), tickfont=dict(color='black', size=12))
    fig.update_layout(font=dict(color='black'), paper_bgcolor='white', plot_bgcolor='white', title_font=dict(color='black', size=18, family="Arial Black"), legend_font=dict(color='black'))
    return fig

# Distinct color palettes for multi-file rendering
COLORSCALES = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Blues', 'Reds', 'Greens']
LINE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# ==========================================
#        STATIC EXPORT FUNCTIONS 
# ==========================================
def set_mpl_style():
    plt.rcParams.update({'font.size': 12, 'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.linewidth': 2, 'xtick.major.width': 2, 'ytick.major.width': 2, 'axes.edgecolor': 'black', 'figure.facecolor': 'white', 'text.color': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'axes.facecolor': 'white'})

def get_mpl_img(fig, fmt, dpi):
    buf = io.BytesIO()
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
    p = ax.scatter(df['X_rel'], df['Y_rel'], df['Z_rel'], c=df['B_mT'], cmap='viridis', s=0.5, alpha=0.6)
    fig.colorbar(p, label='Field (mT)', pad=0.1, shrink=0.7)
    ax.scatter(0, 0, 0, color='red', s=50, marker='o', label='Center')
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.set_title("3D Magnetic Field Cloud")
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0)); ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0)); ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    return fig

# ==========================================
#            MAIN APP LOGIC
# ==========================================
if uploaded_files:
    # Process all uploaded files into a dictionary
    datasets = {}
    for file in uploaded_files:
        bytes_data = file.getvalue()
        df, meta, prof = load_and_process_data(bytes_data, file.name, dist_unit, field_unit)
        datasets[file.name] = {
            'df': df, 
            'meta': meta, 
            'prof': prof, 
            'B0': meta['peak_b']
        }
    
    is_comp_mode = len(datasets) > 1

    # --- SIDEBAR: DOWNLOAD MANAGER ---
    if is_comp_mode:
        st.sidebar.info("📦 Static Multi-file ZIP Export is disabled. View comparisons interactively in the tabs.")
    else:
        if st.sidebar.button(f"🚀 Generate {pub_format} Report"):
            with st.sidebar.status(f"Generating {pub_format} Plots...", expanded=True):
                set_mpl_style()
                zip_buf = io.BytesIO()
                
                single_data = list(datasets.values())[0]
                df_single = single_data['df']
                prof_s = single_data['prof']
                meta_s = single_data['meta']
                B0_s = single_data['B0']

                z0_val = prof_s['z_rels'][np.argmin(np.abs(prof_s['z_rels']))]
                s_df = df_single[df_single['Z_rel'] == z0_val]
                gx, gy = np.mgrid[s_df['X_rel'].min():s_df['X_rel'].max():100j, s_df['Y_rel'].min():s_df['Y_rel'].max():100j]
                gb = griddata((s_df['X_rel'], s_df['Y_rel']), s_df['B_mT'], (gx, gy), method='linear')
                x_l = np.linspace(s_df['X_rel'].min(), s_df['X_rel'].max(), 100)
                y_l = np.linspace(s_df['Y_rel'].min(), s_df['Y_rel'].max(), 100)
                bx = griddata((s_df['X_rel'], s_df['Y_rel']), s_df['B_mT'], (x_l, np.zeros_like(x_l)), method='linear')
                by = griddata((s_df['X_rel'], s_df['Y_rel']), s_df['B_mT'], (np.zeros_like(y_l), y_l), method='linear')

                st.write("Processing Static Images...")
                f1 = create_static_profile(prof_s['z_rels'], prof_s['vals'], B0_s)
                f2 = create_static_heatmap(gx, gy, gb, z0_val)
                f3 = create_static_topology(gx, gy, gb, z0_val)
                f4 = create_static_homogeneity(x_l, y_l, prof_s['z_rels'], bx, by, prof_s['vals'], B0_s)
                f5 = create_static_3d_cloud(df_single, meta_s)

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
        if not is_comp_mode:
            single_b0 = list(datasets.values())[0]['B0']
            st.metric("Peak Field (B0)", f"{single_b0:.2f} mT")
        
        fig_p = go.Figure()
        
        for idx, (name, data) in enumerate(datasets.items()):
            color = LINE_COLORS[idx % len(LINE_COLORS)]
            prof = data['prof']
            B0 = data['B0']
            
            fig_p.add_trace(go.Scatter(
                x=prof['z_rels'], y=prof['vals'], 
                mode='lines+markers', name=name, 
                line=dict(color=color, width=3), 
                marker=dict(color=color, size=6)
            ))
            
            if not is_comp_mode:
                fig_p.add_trace(go.Scatter(x=[0], y=[B0], mode='markers', name='Max Field', marker=dict(color='red', size=14, symbol='star')))

        fig_p.update_layout(title="Magnetic Field Profile", xaxis_title="Z (mm)", yaxis_title="mT")
        fig_p = apply_black_axes(fig_p)
        st.plotly_chart(fig_p, width='stretch')

    with t2:
        fig_3d = go.Figure()
        cx_mesh, cy_mesh, cz_mesh = get_volume_mesh(vol_shape, vol_radius, vol_length, 0)
        fig_3d.add_trace(go.Surface(x=cx_mesh, y=cy_mesh, z=cz_mesh, opacity=0.2, colorscale='Greys', showscale=False, name=vol_shape, showlegend=False))
        fig_3d.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=5, color='red'), showlegend=False))
        fig_3d.add_trace(go.Scatter3d(x=[0, 15], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='red', width=6), showlegend=False))
        
        for idx, (name, data) in enumerate(datasets.items()):
            df = data['df']
            c_scale = COLORSCALES[idx % len(COLORSCALES)]
            
            if len(df) > max_3d_points:
                step = len(df) // max_3d_points
                df_3d = df.iloc[::step]
                if idx == 0: st.warning(f"Displaying {len(df_3d):,} points per dataset (Downsampled for browser performance).")
            else:
                df_3d = df

            fig_3d.add_trace(go.Scatter3d(
                x=df_3d['X_rel'], y=df_3d['Y_rel'], z=df_3d['Z_rel'], 
                mode='markers', name=name,
                marker=dict(size=2, color=df_3d['B_mT'], colorscale=c_scale, showscale=not is_comp_mode, opacity=0.5)
            ))
            
        fig_3d.update_layout(height=800, scene=dict(xaxis=dict(title="X", backgroundcolor="white", color="black"), yaxis=dict(title="Y", backgroundcolor="white", color="black"), zaxis=dict(title="Z", backgroundcolor="white", color="black"), aspectmode='data'), paper_bgcolor='white', font=dict(color="black"), showlegend=True)
        st.plotly_chart(fig_3d, width='stretch')

    with t3:
        # Create a master Z slider based on all unique Z values across all files
        all_zs = []
        for d in datasets.values(): all_zs.extend(d['df']['Z_rel'].unique())
        all_zs = np.sort(np.unique(all_zs))
        
        z0_default = all_zs[np.argmin(np.abs(all_zs))]
        sel_z = st.select_slider("Select Master Z Slice (mm)", options=all_zs, value=z0_default)

        c1, c2 = st.columns(2)
        
        # --- 2D HEATMAP (Dropdown Selected) ---
        with c1:
            file_2d = st.selectbox("📂 Select file for 2D Heatmap:", list(datasets.keys())) if is_comp_mode else list(datasets.keys())[0]
            df_2d = datasets[file_2d]['df']
            
            # Find closest Z slice for this specific file
            closest_z_2d = df_2d.iloc[(df_2d['Z_rel'] - sel_z).abs().argsort()[:1]]['Z_rel'].values[0]
            curr_slice_2d = df_2d[df_2d['Z_rel'] == closest_z_2d]
            
            gx_s, gy_s = np.mgrid[curr_slice_2d['X_rel'].min():curr_slice_2d['X_rel'].max():60j, curr_slice_2d['Y_rel'].min():curr_slice_2d['Y_rel'].max():60j]
            gb_s = griddata((curr_slice_2d['X_rel'], curr_slice_2d['Y_rel']), curr_slice_2d['B_mT'], (gx_s, gy_s), method='linear')
            center_val = griddata((curr_slice_2d['X_rel'], curr_slice_2d['Y_rel']), curr_slice_2d['B_mT'], (0, 0), method='nearest')

            # Bug fix applied here: z=gb_s.T
            fig_h = go.Figure(go.Heatmap(x=gx_s[:,0], y=gy_s[0,:], z=gb_s.T, colorscale='Viridis'))
            fig_h.add_trace(go.Scatter(x=[0], y=[0], mode='markers+text', marker=dict(color='black', size=15, symbol='cross-thin', line=dict(color='white', width=2)), text=[f"{float(center_val):.2f} mT"], textposition="top center", textfont=dict(color='black', size=14, family="Arial Black"), name="Center"))
            fig_h.update_layout(title=f"2D Heatmap (Z={closest_z_2d:.1f})", xaxis_title="X", yaxis_title="Y", width=500, height=500, xaxis=dict(scaleanchor="y", scaleratio=1), yaxis=dict(constrain='domain'))
            fig_h = apply_black_axes(fig_h)
            st.plotly_chart(fig_h, width='stretch')

        # --- 3D TOPOLOGY (Overlaid) ---
        with c2:
            fig_s = go.Figure()
            
            for idx, (name, data) in enumerate(datasets.items()):
                df_3d = data['df']
                c_scale = COLORSCALES[idx % len(COLORSCALES)]
                
                # Find closest Z slice for this specific file
                closest_z = df_3d.iloc[(df_3d['Z_rel'] - sel_z).abs().argsort()[:1]]['Z_rel'].values[0]
                curr_slice = df_3d[df_3d['Z_rel'] == closest_z]
                
                gxs, gys = np.mgrid[curr_slice['X_rel'].min():curr_slice['X_rel'].max():60j, curr_slice['Y_rel'].min():curr_slice['Y_rel'].max():60j]
                gbs = griddata((curr_slice['X_rel'], curr_slice['Y_rel']), curr_slice['B_mT'], (gxs, gys), method='linear')
                
                fig_s.add_trace(go.Surface(
                    z=gbs, x=gxs, y=gys, 
                    colorscale=c_scale, opacity=0.8, 
                    name=name, showscale=not is_comp_mode
                ))

            fig_s.update_layout(title="3D Topology Comparison", scene=dict(xaxis=dict(title="X", backgroundcolor="white", color="black"), yaxis=dict(title="Y", backgroundcolor="white", color="black"), zaxis=dict(title="mT", backgroundcolor="white", color="black"), aspectratio=dict(x=1, y=1, z=0.7)), width=500, height=500, paper_bgcolor='white', font=dict(color="black"))
            st.plotly_chart(fig_s, width='stretch')

    with t4:
        st.subheader("Field Homogeneity Deviation (%)")
        fig_hom = make_subplots(rows=1, cols=3, subplot_titles=("X-Axis", "Y-Axis", "Z-Axis"))
        
        for idx, (name, data) in enumerate(datasets.items()):
            color = LINE_COLORS[idx % len(LINE_COLORS)]
            df = data['df']
            B0 = data['B0']
            prof = data['prof']

            z0_val = prof['z_rels'][np.argmin(np.abs(prof['z_rels']))]
            s_df = df[df['Z_rel'] == z0_val]
            
            x_l = np.linspace(s_df['X_rel'].min(), s_df['X_rel'].max(), 100)
            y_l = np.linspace(s_df['Y_rel'].min(), s_df['Y_rel'].max(), 100)
            bx = griddata((s_df['X_rel'], s_df['Y_rel']), s_df['B_mT'], (x_l, np.zeros_like(x_l)), method='linear')
            by = griddata((s_df['X_rel'], s_df['Y_rel']), s_df['B_mT'], (np.zeros_like(y_l), y_l), method='linear')

            fig_hom.add_trace(go.Scatter(x=x_l, y=(bx-B0)/B0*100, name=name, line=dict(color=color, width=2), showlegend=True), row=1, col=1)
            fig_hom.add_trace(go.Scatter(x=y_l, y=(by-B0)/B0*100, name=name, line=dict(color=color, width=2), showlegend=False), row=1, col=2)
            fig_hom.add_trace(go.Scatter(x=prof['z_rels'], y=(prof['vals']-B0)/B0*100, name=name, line=dict(color=color, width=2), showlegend=False), row=1, col=3)

        fig_hom.update_layout(title="Deviation from Peak (%)", showlegend=True)
        fig_hom = apply_black_axes(fig_hom)
        fig_hom.update_xaxes(title_text="Position (mm)", color="black")
        fig_hom.update_yaxes(title_text="Deviation %", color="black", row=1, col=1)
        st.plotly_chart(fig_hom, width='stretch')

        # --- VOLUME HOMOGENEITY MATH ---
        st.divider()
        st.subheader(f"Volume Homogeneity Analysis ({vol_shape})")
        st.write(f"**Target Volume boundaries:** Radius = {vol_radius} mm" + (f", Length = {vol_length} mm" if vol_shape=="Cylinder" else ""))
        
        hom_data = []
        for name, data in datasets.items():
            ppm, min_b, max_b = calc_vol_homogeneity(data['df'], vol_shape, vol_radius, vol_length, data['B0'])
            hom_data.append({
                "Dataset": name,
                "Pk-Pk (PPM)": ppm,
                "Min Field (mT)": min_b,
                "Max Field (mT)": max_b
            })
            
        st.dataframe(pd.DataFrame(hom_data).style.format({"Pk-Pk (PPM)": "{:.1f}", "Min Field (mT)": "{:.4f}", "Max Field (mT)": "{:.4f}"}), hide_index=True, use_container_width=True)

else:
    st.info("Awaiting CSV file upload...")

# --- HIDDEN SEO METADATA ---
st.markdown("""
<div style="display:none;">
    <h1>Magnetic Field Analysis Software for Python</h1>
    <p>A comprehensive tool for plotting and analyzing 3D magnetic field data.</p>
    <p>Perfect for researchers working with:</p>
    <ul>
        <li>Halbach Arrays (Permanent Magnets)</li>
        <li>Solenoids and Electromagnets</li>
        <li>Dipole and Quadrupole Magnets</li>
        <li>Helmholtz Coils</li>
    </ul>
    <p>Features: Homogeneity calculation (PPM), 3D Scatter Plots, Field Topology, and Vector Field visualization.</p>
    <p>Developed by Dr. Anmol Mahendra.</p>
</div>
""", unsafe_allow_html=True)
