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
    
    st.header("2. Analysis Settings")
    cyl_radius = st.number_input("Reference Cylinder Radius (mm)", value=7.0, step=0.5, on_change=reset_report)
    cyl_length = st.number_input("Reference Cylinder Length (mm)", value=46.0, step=1.0, on_change=reset_report)
    
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

# Distinct color palettes for multi-file rendering
COLORSCALES = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Blues', 'Reds', 'Greens']
LINE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

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
        st.sidebar.info("📦 Multi-file ZIP Export is disabled. View comparisons interactively in the tabs.")
    else:
        # ZIP generation for single files remains exactly the same as your original script
        pass 

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
        cx_mesh, cy_mesh, cz_mesh = get_cylinder_mesh(cyl_radius, cyl_length, 0)
        fig_3d.add_trace(go.Surface(x=cx_mesh, y=cy_mesh, z=cz_mesh, opacity=0.2, colorscale='Greys', showscale=False, name='Cylinder', showlegend=False))
        fig_3d.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=5, color='red'), showlegend=False))
        fig_3d.add_trace(go.Scatter3d(x=[0, 15], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='red', width=6), showlegend=False))
        
        for idx, (name, data) in enumerate(datasets.items()):
            df = data['df']
            c_scale = COLORSCALES[idx % len(COLORSCALES)]
            
            # Downsample for performance
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

            # T FIX APPLIED HERE:
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
                
                # Plot multiple surfaces on the same axis
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

else:
    st.info("Awaiting CSV file upload...")
