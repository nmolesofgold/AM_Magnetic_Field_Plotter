import io
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from plotly.subplots import make_subplots
from scipy.interpolate import griddata


# ==========================================
#            STREAMLIT PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Magnetic Field Analyzer",
    page_icon="🧲",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.linkedin.com/in/nmolesofgold/",
        "Report a bug": "https://www.linkedin.com/in/nmolesofgold/",
        "About": (
            "### Magnetic Field Analyzer\n"
            "**Developed by Dr. Anmol Mahendra**.\n\n"
            "A versatile tool for visualising magnetic field scans."
        ),
    },
)

st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"] { background-color: #ffffff; }
        [data-testid="stSidebar"] { background-color: #f0f2f6; }
        [data-testid="stHeader"] { background-color: #ffffff; }

        h1, h2, h3, h4, p, label, .stMarkdown, .stMetricValue {
            color: #000000 !important;
        }

        [data-testid="stFileUploader"] section {
            background-color: #f8f9fa !important;
            border: 1px dashed #444 !important;
        }

        .stButton button, .stDownloadButton button {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #000000 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

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
#            CONSTANTS
# ==========================================
DIST_MULTIPLIERS = {
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "inches": 25.4,
}

FIELD_MULTIPLIERS_TO_MT = {
    "milliTesla (mT)": 1.0,
    "Tesla (T)": 1000.0,
    "Gauss (G)": 0.1,
    "microTesla (µT)": 0.001,
}

COLORSCALES = ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Blues", "Reds", "Greens"]
LINE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


# ==========================================
#            SIDEBAR
# ==========================================
with st.sidebar:
    st.header("1. Data Input")

    c1, c2 = st.columns(2)
    with c1:
        dist_unit = st.selectbox(
            "Distance Unit",
            ["mm", "cm", "m", "inches"],
            index=0,
            on_change=reset_report,
        )
    with c2:
        field_unit = st.selectbox(
            "Field Unit",
            ["milliTesla (mT)", "Tesla (T)", "Gauss (G)", "microTesla (µT)"],
            index=0,
            on_change=reset_report,
        )

    with st.expander("ℹ️ File Format Instructions"):
        st.markdown(
            """
            **Required Header Row:**
            `X_mm,Y_mm,Z_mm,Field`

            _(Keep the headers as X_mm, Y_mm, Z_mm even if your raw data is in cm or inches.
            Select your real units from the dropdown above.)_
            """
        )
        dummy_data = pd.DataFrame(
            {
                "X_mm": [125.7, 125.7, 125.69],
                "Y_mm": [-26.59, -26.58, -26.57],
                "Z_mm": [-4, -4, -4],
                "Field": [208.3, 208.3, 208.3],
            }
        )
        csv_template = dummy_data.to_csv(index=False).encode("utf-8")
        st.download_button("📄 Download Template", csv_template, "magnetic_template.csv", "text/csv")

    uploaded_files = st.file_uploader(
        "Upload CSV Scan(s)",
        type=["csv"],
        accept_multiple_files=True,
        on_change=reset_report,
    )

    st.header("2. Center & Interpolation")
    center_mode = st.radio(
        "Center Definition",
        ["Auto (Halbach center logic)", "Manual Override"],
        index=0,
        on_change=reset_report,
    )

    manual_cx = 0.0
    manual_cy = 0.0
    manual_cz = 0.0
    if center_mode == "Manual Override":
        manual_cx = st.number_input("Manual Center X (mm)", value=0.0, step=0.1, on_change=reset_report)
        manual_cy = st.number_input("Manual Center Y (mm)", value=0.0, step=0.1, on_change=reset_report)
        manual_cz = st.number_input("Manual Center Z (mm)", value=0.0, step=0.1, on_change=reset_report)

    interp_method = st.selectbox(
        "Interpolation Method",
        ["linear", "nearest", "cubic"],
        index=0,
        on_change=reset_report,
    )

    z_match_tol = st.number_input(
        "Slice Match Tolerance (mm)",
        min_value=0.000001,
        value=0.001,
        step=0.001,
        format="%.6f",
        on_change=reset_report,
    )

    st.header("3. Analysis Volume (Homogeneity / Overlay)")
    vol_shape = st.selectbox(
        "Target Volume Shape",
        ["Cylinder", "Sphere", "Cube", "Cuboid"],
        on_change=reset_report,
    )

    vol_radius = 7.0
    vol_length = 46.0
    cube_side = 14.0
    cuboid_x = 14.0
    cuboid_y = 14.0
    cuboid_z = 46.0

    if vol_shape in ["Cylinder", "Sphere"]:
        vol_radius = st.number_input("Radius (mm)", value=7.0, step=0.5, on_change=reset_report)
        if vol_shape == "Cylinder":
            vol_length = st.number_input("Length (mm)", value=46.0, step=1.0, on_change=reset_report)

    elif vol_shape == "Cube":
        cube_side = st.number_input("Cube Side Length (mm)", value=14.0, step=0.5, on_change=reset_report)

    elif vol_shape == "Cuboid":
        cuboid_x = st.number_input("Cuboid X Length (mm)", value=14.0, step=0.5, on_change=reset_report)
        cuboid_y = st.number_input("Cuboid Y Length (mm)", value=14.0, step=0.5, on_change=reset_report)
        cuboid_z = st.number_input("Cuboid Z Length (mm)", value=46.0, step=0.5, on_change=reset_report)

    st.header("4. Performance")
    max_3d_points = st.slider("Max 3D Points (Interactive)", 5000, 100000, 20000, 5000)
    grid_resolution = st.slider("Slice Grid Resolution", 30, 200, 60, 10, on_change=reset_report)

    st.divider()
    st.header("5. Publication Settings")
    pub_format = st.selectbox("File Format", ["PNG", "PDF", "SVG"], on_change=reset_report)
    pub_dpi = st.select_slider("DPI (for PNG)", options=[300, 600, 1200], value=300) if pub_format == "PNG" else 300
    file_ext = pub_format.lower()


# ==========================================
#            GENERIC HELPERS
# ==========================================
def apply_black_axes(fig):
    fig.update_xaxes(
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        showgrid=True,
        gridcolor="lightgrey",
        title_font=dict(color="black", size=14, family="Arial Black"),
        tickfont=dict(color="black", size=12),
    )
    fig.update_yaxes(
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        showgrid=True,
        gridcolor="lightgrey",
        title_font=dict(color="black", size=14, family="Arial Black"),
        tickfont=dict(color="black", size=12),
    )
    fig.update_layout(
        font=dict(color="black"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        title_font=dict(color="black", size=18, family="Arial Black"),
        legend_font=dict(color="black"),
    )
    return fig


def set_mpl_style():
    plt.rcParams.update(
        {
            "font.size": 12,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.linewidth": 2,
            "xtick.major.width": 2,
            "ytick.major.width": 2,
            "axes.edgecolor": "black",
            "figure.facecolor": "white",
            "text.color": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.facecolor": "white",
        }
    )


def get_mpl_img(fig, fmt, dpi):
    buf = io.BytesIO()
    kwargs = dict(format=fmt, bbox_inches="tight", facecolor="white", edgecolor="none")
    if fmt == "png":
        kwargs["dpi"] = dpi
    fig.savefig(buf, **kwargs)
    plt.close(fig)
    return buf.getvalue()


def validate_dataframe(df):
    if "Field" in df.columns and "Magnetic_Field_Reading" not in df.columns:
        df = df.rename(columns={"Field": "Magnetic_Field_Reading"})

    required = {"X_mm", "Y_mm", "Z_mm", "Magnetic_Field_Reading"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    df = df[list(required)].copy()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(required)).copy()

    if df.empty:
        raise ValueError("No valid numeric rows found after cleaning the uploaded CSV.")

    return df


def safe_griddata(points_xy, values, xi, method="linear"):
    points_xy = np.asarray(points_xy)
    values = np.asarray(values)

    if len(values) == 0:
        return None

    method_to_use = method
    if method == "cubic" and len(values) < 4:
        method_to_use = "linear"
    if method_to_use in ("linear", "cubic") and len(values) < 3:
        method_to_use = "nearest"

    try:
        out = griddata(points_xy, values, xi, method=method_to_use)
    except Exception:
        out = None

    if out is None:
        try:
            return griddata(points_xy, values, xi, method="nearest")
        except Exception:
            return None

    out_arr = np.asarray(out)
    if np.all(np.isnan(out_arr)):
        try:
            return griddata(points_xy, values, xi, method="nearest")
        except Exception:
            return out

    if np.isnan(out_arr).any():
        try:
            nearest = griddata(points_xy, values, xi, method="nearest")
            out = np.where(np.isnan(out_arr), nearest, out_arr)
        except Exception:
            out = out_arr

    return out


def safe_percent_deviation(values, b0):
    if values is None:
        return None
    values = np.asarray(values, dtype=float)
    if np.isclose(b0, 0):
        return np.full(values.shape, np.nan)
    return (values - b0) / b0 * 100.0


# ==========================================
#            CENTER / DATA PROCESSING
# ==========================================
def compute_auto_center_halbach(df, z_interp_method="nearest"):
    cx = (df["X_mm"].max() + df["X_mm"].min()) / 2
    cy = (df["Y_mm"].max() + df["Y_mm"].min()) / 2

    unique_zs = np.sort(df["Z_mm"].unique())
    if len(unique_zs) == 0:
        raise ValueError("No Z slices found in the uploaded data.")

    z_vals = []
    for z in unique_zs:
        slice_df = df[np.isclose(df["Z_mm"], z)].copy()
        points = np.column_stack(((slice_df["X_mm"] - cx).values, (slice_df["Y_mm"] - cy).values))
        values = slice_df["B_mT"].values
        b = safe_griddata(points, values, (0, 0), method=z_interp_method)
        z_vals.append(float(b) if b is not None else np.nan)

    z_vals = np.asarray(z_vals, dtype=float)
    valid_mask = ~np.isnan(z_vals)
    if not np.any(valid_mask):
        raise ValueError("Unable to estimate centerline field for any Z slice.")

    valid_zs = unique_zs[valid_mask]
    valid_vals = z_vals[valid_mask]
    peak_idx = np.argmax(valid_vals)
    cz = valid_zs[peak_idx]
    peak_b = valid_vals[peak_idx]
    return cx, cy, cz, peak_b, valid_zs, valid_vals


@st.cache_data
def load_and_process_data(file_bytes, dist_unit, field_unit, center_mode, manual_cx, manual_cy, manual_cz, interp_method):
    df = pd.read_csv(io.BytesIO(file_bytes))
    df = validate_dataframe(df)

    df["B_mT"] = df["Magnetic_Field_Reading"] * FIELD_MULTIPLIERS_TO_MT[field_unit]
    dist_mult = DIST_MULTIPLIERS[dist_unit]
    for col in ["X_mm", "Y_mm", "Z_mm"]:
        df[col] = df[col] * dist_mult

    df["Z_mm"] = df["Z_mm"].round(6)

    if center_mode == "Manual Override":
        cx, cy, cz = float(manual_cx), float(manual_cy), float(manual_cz)

        unique_zs = np.sort(df["Z_mm"].unique())
        z_vals = []
        for z in unique_zs:
            slice_df = df[np.isclose(df["Z_mm"], z)].copy()
            points = np.column_stack(((slice_df["X_mm"] - cx).values, (slice_df["Y_mm"] - cy).values))
            values = slice_df["B_mT"].values
            b = safe_griddata(points, values, (0, 0), method=interp_method)
            z_vals.append(float(b) if b is not None else np.nan)

        z_vals = np.asarray(z_vals, dtype=float)
        valid_mask = ~np.isnan(z_vals)
        valid_zs = unique_zs[valid_mask]
        valid_vals = z_vals[valid_mask]

        if len(valid_vals) == 0:
            raise ValueError("Unable to estimate field values at the manual center.")

        manual_slice = df[np.isclose(df["Z_mm"], cz)].copy()
        if manual_slice.empty:
            peak_b = float(np.nanmax(valid_vals))
        else:
            pts = np.column_stack(((manual_slice["X_mm"] - cx).values, (manual_slice["Y_mm"] - cy).values))
            vals = manual_slice["B_mT"].values
            b0 = safe_griddata(pts, vals, (0, 0), method=interp_method)
            peak_b = float(b0) if b0 is not None else float(np.nanmax(valid_vals))
    else:
        cx, cy, cz, peak_b, valid_zs, valid_vals = compute_auto_center_halbach(df, interp_method)

    df["X_rel"] = df["X_mm"] - cx
    df["Y_rel"] = df["Y_mm"] - cy
    df["Z_rel"] = (df["Z_mm"] - cz).round(6)

    meta = {"peak_b": peak_b, "cx": cx, "cy": cy, "cz": cz}
    prof = {"z_rels": valid_zs - cz, "vals": valid_vals}
    return df, meta, prof


# ==========================================
#            SLICE HELPERS
# ==========================================
def get_available_zs(df):
    return np.sort(df["Z_rel"].unique())


def find_closest_z_value(df, target_z):
    zs = get_available_zs(df)
    if len(zs) == 0:
        return None
    return float(zs[np.argmin(np.abs(zs - target_z))])


def get_slice_df(df, target_z, z_tol):
    closest_z = find_closest_z_value(df, target_z)
    if closest_z is None:
        return None, df.iloc[0:0].copy()

    slice_df = df[np.abs(df["Z_rel"] - closest_z) <= max(z_tol, 1e-12)].copy()
    if slice_df.empty:
        slice_df = df[np.isclose(df["Z_rel"], closest_z)].copy()
    return closest_z, slice_df


def interpolate_slice_grid(slice_df, n_grid, interp_method):
    if slice_df.empty or len(slice_df) < 3:
        return None, None, None

    x_min, x_max = slice_df["X_rel"].min(), slice_df["X_rel"].max()
    y_min, y_max = slice_df["Y_rel"].min(), slice_df["Y_rel"].max()

    if np.isclose(x_min, x_max) or np.isclose(y_min, y_max):
        return None, None, None

    gx, gy = np.mgrid[x_min:x_max:complex(n_grid), y_min:y_max:complex(n_grid)]
    points = np.column_stack((slice_df["X_rel"].values, slice_df["Y_rel"].values))
    values = slice_df["B_mT"].values
    gb = safe_griddata(points, values, (gx, gy), method=interp_method)
    return gx, gy, gb


def get_center_value(slice_df, interp_method):
    if slice_df.empty:
        return np.nan
    points = np.column_stack((slice_df["X_rel"].values, slice_df["Y_rel"].values))
    vals = slice_df["B_mT"].values
    cval = safe_griddata(points, vals, (0, 0), method=interp_method)
    return float(cval) if cval is not None else np.nan


def centerline_profile_from_slice(slice_df, axis, n_points, interp_method):
    if slice_df.empty:
        return None, None

    points = np.column_stack((slice_df["X_rel"].values, slice_df["Y_rel"].values))
    values = slice_df["B_mT"].values

    if axis == "x":
        coords = np.linspace(slice_df["X_rel"].min(), slice_df["X_rel"].max(), n_points)
        vals = safe_griddata(points, values, (coords, np.zeros_like(coords)), method=interp_method)
    else:
        coords = np.linspace(slice_df["Y_rel"].min(), slice_df["Y_rel"].max(), n_points)
        vals = safe_griddata(points, values, (np.zeros_like(coords), coords), method=interp_method)

    return coords, vals


def prepare_slice_artifacts(df, target_z, z_tol, n_grid, interp_method, n_profile=100):
    closest_z, slice_df = get_slice_df(df, target_z, z_tol)

    if closest_z is None or slice_df.empty:
        return {
            "closest_z": None,
            "slice_df": slice_df,
            "gx": None,
            "gy": None,
            "gb": None,
            "center_val": np.nan,
            "x_l": None,
            "bx": None,
            "y_l": None,
            "by": None,
        }

    gx, gy, gb = interpolate_slice_grid(slice_df, n_grid, interp_method)
    center_val = get_center_value(slice_df, interp_method)
    x_l, bx = centerline_profile_from_slice(slice_df, "x", n_profile, interp_method)
    y_l, by = centerline_profile_from_slice(slice_df, "y", n_profile, interp_method)

    return {
        "closest_z": closest_z,
        "slice_df": slice_df,
        "gx": gx,
        "gy": gy,
        "gb": gb,
        "center_val": center_val,
        "x_l": x_l,
        "bx": bx,
        "y_l": y_l,
        "by": by,
    }


# ==========================================
#            VOLUME MESH + HOMOGENEITY
# ==========================================
def get_volume_mesh(shape, radius=None, length=None, cube_side=None, cuboid_dims=None, z_center=0):
    if shape == "Cylinder":
        z = np.linspace(z_center - length / 2, z_center + length / 2, 50)
        theta = np.linspace(0, 2 * np.pi, 50)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = radius * np.cos(theta_grid)
        y_grid = radius * np.sin(theta_grid)
        return x_grid, y_grid, z_grid

    if shape == "Sphere":
        u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:50j]
        x_grid = radius * np.cos(u) * np.sin(v)
        y_grid = radius * np.sin(u) * np.sin(v)
        z_grid = z_center + radius * np.cos(v)
        return x_grid, y_grid, z_grid

    return None, None, None


def get_box_edges(x_len, y_len, z_len):
    hx, hy, hz = x_len / 2, y_len / 2, z_len / 2
    verts = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]
    ])
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]
    return verts, edges


def add_volume_overlay_plotly(fig, shape, radius, length, cube_side, cuboid_dims):
    if shape in ["Cylinder", "Sphere"]:
        xg, yg, zg = get_volume_mesh(
            shape=shape,
            radius=radius,
            length=length,
            cube_side=cube_side,
            cuboid_dims=cuboid_dims,
            z_center=0,
        )
        fig.add_trace(
            go.Surface(
                x=xg,
                y=yg,
                z=zg,
                opacity=0.2,
                colorscale="Greys",
                showscale=False,
                name=shape,
                showlegend=False,
            )
        )
    elif shape == "Cube":
        verts, edges = get_box_edges(cube_side, cube_side, cube_side)
        for e in edges:
            p1, p2 = verts[e[0]], verts[e[1]]
            fig.add_trace(
                go.Scatter3d(
                    x=[p1[0], p2[0]],
                    y=[p1[1], p2[1]],
                    z=[p1[2], p2[2]],
                    mode="lines",
                    line=dict(color="gray", width=4),
                    showlegend=False,
                )
            )
    elif shape == "Cuboid":
        x_len, y_len, z_len = cuboid_dims
        verts, edges = get_box_edges(x_len, y_len, z_len)
        for e in edges:
            p1, p2 = verts[e[0]], verts[e[1]]
            fig.add_trace(
                go.Scatter3d(
                    x=[p1[0], p2[0]],
                    y=[p1[1], p2[1]],
                    z=[p1[2], p2[2]],
                    mode="lines",
                    line=dict(color="gray", width=4),
                    showlegend=False,
                )
            )
    return fig


def calc_vol_homogeneity(df, shape, radius, length, cube_side, cuboid_dims, b0):
    if shape == "Cylinder":
        mask = (df["X_rel"]**2 + df["Y_rel"]**2 <= radius**2) & (df["Z_rel"].abs() <= length / 2)
    elif shape == "Sphere":
        mask = (df["X_rel"]**2 + df["Y_rel"]**2 + df["Z_rel"]**2 <= radius**2)
    elif shape == "Cube":
        h = cube_side / 2
        mask = (df["X_rel"].abs() <= h) & (df["Y_rel"].abs() <= h) & (df["Z_rel"].abs() <= h)
    else:  # Cuboid
        hx, hy, hz = cuboid_dims[0] / 2, cuboid_dims[1] / 2, cuboid_dims[2] / 2
        mask = (df["X_rel"].abs() <= hx) & (df["Y_rel"].abs() <= hy) & (df["Z_rel"].abs() <= hz)

    pts = df.loc[mask, "B_mT"]
    if len(pts) == 0:
        return np.nan, np.nan, np.nan, 0

    pk_pk = pts.max() - pts.min()
    ppm = np.nan if np.isclose(b0, 0) else (pk_pk / b0) * 1e6
    return ppm, pts.min(), pts.max(), int(len(pts))


def make_homogeneity_dataframe(datasets, shape, radius, length, cube_side, cuboid_dims):
    rows = []
    for name, data in datasets.items():
        ppm, min_b, max_b, n_pts = calc_vol_homogeneity(
            data["df"], shape, radius, length, cube_side, cuboid_dims, data["B0"]
        )
        rows.append({
            "Dataset": name,
            "Pk-Pk (PPM)": ppm,
            "Min Field (mT)": min_b,
            "Max Field (mT)": max_b,
            "Points in Volume": n_pts,
            "B0 (mT)": data["B0"],
            "Center X (mm)": data["meta"]["cx"],
            "Center Y (mm)": data["meta"]["cy"],
            "Center Z (mm)": data["meta"]["cz"],
        })
    return pd.DataFrame(rows)


# ==========================================
#            STATIC EXPORT FIGURES
# ==========================================
def create_static_profile_multi(datasets):
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (name, data) in enumerate(datasets.items()):
        color = LINE_COLORS[idx % len(LINE_COLORS)]
        ax.plot(data["prof"]["z_rels"], data["prof"]["vals"], marker="o", linewidth=2, color=color, label=name)
    ax.set_title("Magnetic Field Profile")
    ax.set_xlabel("Z (mm)")
    ax.set_ylabel("Field (mT)")
    ax.grid(True, linestyle="--")
    ax.legend()
    return fig


def create_static_3d_cloud_multi(datasets, shape, radius, length, cube_side, cuboid_dims, max_3d_points):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for idx, (name, data) in enumerate(datasets.items()):
        color = LINE_COLORS[idx % len(LINE_COLORS)]
        df = data["df"]
        if len(df) > max_3d_points:
            step = max(1, len(df) // max_3d_points)
            df = df.iloc[::step].copy()

        ax.scatter(df["X_rel"], df["Y_rel"], df["Z_rel"], s=1, alpha=0.35, color=color, label=name)

    # Overlay shapes
    if shape == "Cylinder":
        xg, yg, zg = get_volume_mesh(shape="Cylinder", radius=radius, length=length)
        ax.plot_wireframe(xg, yg, zg, color="gray", alpha=0.25, linewidth=0.5)
    elif shape == "Sphere":
        xg, yg, zg = get_volume_mesh(shape="Sphere", radius=radius)
        ax.plot_wireframe(xg, yg, zg, color="gray", alpha=0.25, linewidth=0.5)
    elif shape == "Cube":
        verts, edges = get_box_edges(cube_side, cube_side, cube_side)
        for e in edges:
            p1, p2 = verts[e[0]], verts[e[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="gray", linewidth=1.5)
    elif shape == "Cuboid":
        verts, edges = get_box_edges(cuboid_dims[0], cuboid_dims[1], cuboid_dims[2])
        for e in edges:
            p1, p2 = verts[e[0]], verts[e[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="gray", linewidth=1.5)

    ax.scatter(0, 0, 0, color="red", s=60, label="Center")
    ax.set_title("3D Magnetic Field Cloud Comparison")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.legend()
    return fig


def create_static_slice_heatmap_panel(datasets, sel_z, z_tol, grid_resolution, interp_method):
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, datasets.items()):
        arts = prepare_slice_artifacts(data["df"], sel_z, z_tol, grid_resolution, interp_method)
        if arts["gx"] is None or arts["gb"] is None:
            ax.set_title(f"{name}\nNo valid slice")
            ax.axis("off")
            continue
        pc = ax.pcolormesh(arts["gx"], arts["gy"], arts["gb"], cmap="viridis", shading="auto")
        ax.scatter([0], [0], c="black", s=50, marker="+")
        ax.set_title(f"{name}\nZ={arts['closest_z']:.3f} mm")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_aspect("equal")
        fig.colorbar(pc, ax=ax, shrink=0.8, label="mT")

    fig.suptitle("2D Heatmap Comparison")
    plt.tight_layout()
    return fig


def create_static_topology_panel(datasets, sel_z, z_tol, grid_resolution, interp_method):
    n = len(datasets)
    fig = plt.figure(figsize=(6 * n, 5))

    for i, (name, data) in enumerate(datasets.items(), start=1):
        ax = fig.add_subplot(1, n, i, projection="3d")
        arts = prepare_slice_artifacts(data["df"], sel_z, z_tol, grid_resolution, interp_method)
        if arts["gx"] is None or arts["gb"] is None:
            ax.set_title(f"{name}\nNo valid slice")
            continue
        surf = ax.plot_surface(arts["gx"], arts["gy"], arts["gb"], cmap="viridis", edgecolor="none")
        ax.set_title(f"{name}\nZ={arts['closest_z']:.3f} mm")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Field (mT)")
        fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08)

    fig.suptitle("3D Topology Comparison")
    plt.tight_layout()
    return fig


def create_static_homogeneity_multi(datasets, interp_method, z_match_tol, grid_resolution):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, (name, data) in enumerate(datasets.items()):
        color = LINE_COLORS[idx % len(LINE_COLORS)]
        B0 = data["B0"]
        prof = data["prof"]

        arts = prepare_slice_artifacts(data["df"], 0.0, z_match_tol, grid_resolution, interp_method)

        if arts["x_l"] is None or arts["y_l"] is None:
            continue

        axes[0].plot(arts["x_l"], safe_percent_deviation(arts["bx"], B0), color=color, linewidth=2, label=name)
        axes[1].plot(arts["y_l"], safe_percent_deviation(arts["by"], B0), color=color, linewidth=2)
        axes[2].plot(prof["z_rels"], safe_percent_deviation(prof["vals"], B0), color=color, linewidth=2)

    axes[0].set_title("X-Axis")
    axes[1].set_title("Y-Axis")
    axes[2].set_title("Z-Axis")
    axes[0].set_ylabel("Deviation (%)")
    for ax in axes:
        ax.set_xlabel("Position (mm)")
        ax.grid(True, linestyle=":")
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    axes[0].legend()
    fig.suptitle("Deviation from Peak (%)")
    plt.tight_layout()
    return fig


def export_figures_zip(figures, file_ext, pub_dpi):
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        for filename, fig in figures:
            zf.writestr(filename, get_mpl_img(fig, file_ext, pub_dpi))
    return zip_buf.getvalue()


# ==========================================
#            PLOTLY BUILDERS
# ==========================================
def build_overview_plot(datasets, is_comp_mode):
    fig = go.Figure()
    for idx, (name, data) in enumerate(datasets.items()):
        color = LINE_COLORS[idx % len(LINE_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=data["prof"]["z_rels"],
                y=data["prof"]["vals"],
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=3),
                marker=dict(color=color, size=6),
            )
        )
        if not is_comp_mode:
            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[data["B0"]],
                    mode="markers",
                    name="Max Field",
                    marker=dict(color="red", size=14, symbol="star"),
                )
            )

    fig.update_layout(title="Magnetic Field Profile", xaxis_title="Z (mm)", yaxis_title="Field (mT)")
    return apply_black_axes(fig)


def build_3d_cloud_plot(datasets, shape, radius, length, cube_side, cuboid_dims, max_3d_points, global_bmin, global_bmax):
    fig = go.Figure()
    fig = add_volume_overlay_plotly(fig, shape, radius, length, cube_side, cuboid_dims)

    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers",
                               marker=dict(size=5, color="red"), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0, 15], y=[0, 0], z=[0, 0], mode="lines",
                               line=dict(color="red", width=6), showlegend=False))

    any_downsampled = False

    for idx, (name, data) in enumerate(datasets.items()):
        df = data["df"]
        c_scale = COLORSCALES[idx % len(COLORSCALES)]
        if len(df) > max_3d_points:
            step = max(1, len(df) // max_3d_points)
            df = df.iloc[::step].copy()
            any_downsampled = True

        fig.add_trace(
            go.Scatter3d(
                x=df["X_rel"], y=df["Y_rel"], z=df["Z_rel"],
                mode="markers", name=name,
                marker=dict(
                    size=2,
                    color=df["B_mT"],
                    colorscale=c_scale,
                    cmin=global_bmin,
                    cmax=global_bmax,
                    showscale=False,
                    opacity=0.5
                )
            )
        )

    fig.update_layout(
        height=800,
        scene=dict(
            xaxis=dict(title="X (mm)", backgroundcolor="white", color="black"),
            yaxis=dict(title="Y (mm)", backgroundcolor="white", color="black"),
            zaxis=dict(title="Z (mm)", backgroundcolor="white", color="black"),
            aspectmode="data",
        ),
        paper_bgcolor="white",
        font=dict(color="black"),
        showlegend=True,
    )
    return fig, any_downsampled


def build_heatmap_plot(arts, title, global_bmin, global_bmax):
    fig = go.Figure(
        go.Heatmap(
            x=arts["gx"][:, 0],
            y=arts["gy"][0, :],
            z=arts["gb"].T,
            colorscale="Viridis",
            zmin=global_bmin,
            zmax=global_bmax,
            colorbar=dict(title="Field (mT)"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0], y=[0], mode="markers+text",
            marker=dict(color="black", size=15, symbol="cross-thin", line=dict(color="white", width=2)),
            text=[f"{arts['center_val']:.2f} mT"],
            textposition="top center",
            textfont=dict(color="black", size=14, family="Arial Black"),
            name="Center",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="X (mm)",
        yaxis_title="Y (mm)",
        height=500,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(constrain="domain"),
    )
    return apply_black_axes(fig)


def build_topology_comparison_plot(topology_inputs, global_bmin, global_bmax, is_comp_mode):
    fig = go.Figure()
    valid = False
    for idx, item in enumerate(topology_inputs):
        arts = item["artifacts"]
        if arts["gx"] is None or arts["gb"] is None:
            continue
        valid = True
        fig.add_trace(
            go.Surface(
                z=arts["gb"],
                x=arts["gx"],
                y=arts["gy"],
                colorscale=COLORSCALES[idx % len(COLORSCALES)],
                cmin=global_bmin,
                cmax=global_bmax,
                opacity=0.8,
                name=item["name"],
                showscale=not is_comp_mode,
            )
        )
    if not valid:
        return None

    fig.update_layout(
        title="3D Topology Comparison",
        scene=dict(
            xaxis=dict(title="X (mm)", backgroundcolor="white", color="black"),
            yaxis=dict(title="Y (mm)", backgroundcolor="white", color="black"),
            zaxis=dict(title="Field (mT)", backgroundcolor="white", color="black"),
            aspectratio=dict(x=1, y=1, z=0.7),
        ),
        height=500,
        paper_bgcolor="white",
        font=dict(color="black"),
    )
    return fig


def build_homogeneity_plot(datasets, grid_resolution, interp_method, z_match_tol):
    fig = make_subplots(rows=1, cols=3, subplot_titles=("X-Axis", "Y-Axis", "Z-Axis"))
    plotted_any = False

    for idx, (name, data) in enumerate(datasets.items()):
        color = LINE_COLORS[idx % len(LINE_COLORS)]
        B0 = data["B0"]
        prof = data["prof"]
        arts = prepare_slice_artifacts(data["df"], 0.0, z_match_tol, grid_resolution, interp_method)

        if arts["x_l"] is None or arts["y_l"] is None:
            continue

        fig.add_trace(go.Scatter(x=arts["x_l"], y=safe_percent_deviation(arts["bx"], B0),
                                 name=name, line=dict(color=color, width=2), showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=arts["y_l"], y=safe_percent_deviation(arts["by"], B0),
                                 name=name, line=dict(color=color, width=2), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=prof["z_rels"], y=safe_percent_deviation(prof["vals"], B0),
                                 name=name, line=dict(color=color, width=2), showlegend=False), row=1, col=3)
        plotted_any = True

    if not plotted_any:
        return None

    fig.update_layout(title="Deviation from Peak (%)", showlegend=True)
    fig = apply_black_axes(fig)
    fig.update_xaxes(title_text="Position (mm)", color="black")
    fig.update_yaxes(title_text="Deviation %", color="black", row=1, col=1)
    return fig


# ==========================================
#            MAIN APP LOGIC
# ==========================================
if uploaded_files:
    datasets = {}
    load_errors = []

    for file in uploaded_files:
        try:
            df, meta, prof = load_and_process_data(
                file.getvalue(),
                dist_unit,
                field_unit,
                center_mode,
                manual_cx,
                manual_cy,
                manual_cz,
                interp_method,
            )
            datasets[file.name] = {"df": df, "meta": meta, "prof": prof, "B0": meta["peak_b"]}
        except Exception as e:
            load_errors.append(f"{file.name}: {e}")

    for err in load_errors:
        st.error(err)

    if not datasets:
        st.warning("No valid datasets could be loaded from the uploaded files.")
        st.stop()

    is_comp_mode = len(datasets) > 1
    global_bmin = min(d["df"]["B_mT"].min() for d in datasets.values())
    global_bmax = max(d["df"]["B_mT"].max() for d in datasets.values())

    cuboid_dims = (cuboid_x, cuboid_y, cuboid_z)

    # ==========================================
    #            DOWNLOAD MANAGER
    # ==========================================
    if st.sidebar.button(f"🚀 Generate {pub_format} Report"):
        with st.sidebar.status(f"Generating {pub_format} plots...", expanded=True):
            set_mpl_style()
            figures = []

            # Overview
            figures.append((f"1_overview_profile.{file_ext}", create_static_profile_multi(datasets)))

            # 3D cloud comparison / single
            figures.append((
                f"2_cloud_3d.{file_ext}",
                create_static_3d_cloud_multi(
                    datasets, vol_shape, vol_radius, vol_length, cube_side, cuboid_dims, max_3d_points
                )
            ))

            # Choose default export slice near z=0
            all_zs = []
            for d in datasets.values():
                all_zs.extend(d["df"]["Z_rel"].unique())
            all_zs = np.sort(np.unique(np.asarray(all_zs, dtype=float)))
            export_z = float(all_zs[np.argmin(np.abs(all_zs))]) if len(all_zs) else 0.0

            figures.append((
                f"3_heatmap_comparison.{file_ext}",
                create_static_slice_heatmap_panel(datasets, export_z, z_match_tol, grid_resolution, interp_method)
            ))
            figures.append((
                f"4_topology_comparison.{file_ext}",
                create_static_topology_panel(datasets, export_z, z_match_tol, grid_resolution, interp_method)
            ))
            figures.append((
                f"5_homogeneity.{file_ext}",
                create_static_homogeneity_multi(datasets, interp_method, z_match_tol, grid_resolution)
            ))

            zip_data = export_figures_zip(figures, file_ext, pub_dpi)
            st.session_state.zip_data = zip_data
            st.session_state.zip_ready = True
            st.write("Done!")

    if st.session_state.zip_ready:
        st.sidebar.download_button(
            label=f"📦 Download ZIP ({file_ext.upper()})",
            data=st.session_state.zip_data,
            file_name=f"Magnetic_Analysis_{file_ext}.zip",
            mime="application/zip",
        )

    # ==========================================
    #            TABS
    # ==========================================
    t1, t2, t3, t4 = st.tabs(["Overview", "3D Cloud", "Slice Viewer", "Homogeneity"])

    with t1:
        if not is_comp_mode:
            st.metric("Peak Field (B0)", f"{list(datasets.values())[0]['B0']:.2f} mT")

        if center_mode == "Manual Override":
            st.caption(f"Using manual center: X={manual_cx:.3f} mm, Y={manual_cy:.3f} mm, Z={manual_cz:.3f} mm")
        else:
            st.caption("Using automatic Halbach center logic.")

        fig_p = build_overview_plot(datasets, is_comp_mode)
        st.plotly_chart(fig_p, use_container_width=True)

    with t2:
        fig_3d, any_downsampled = build_3d_cloud_plot(
            datasets,
            vol_shape,
            vol_radius,
            vol_length,
            cube_side,
            cuboid_dims,
            max_3d_points,
            global_bmin,
            global_bmax,
        )
        if any_downsampled:
            st.warning("One or more datasets were downsampled for browser performance.")
        st.plotly_chart(fig_3d, use_container_width=True)

    with t3:
        all_zs = []
        for d in datasets.values():
            all_zs.extend(d["df"]["Z_rel"].unique())
        all_zs = np.sort(np.unique(np.asarray(all_zs, dtype=float)))

        if len(all_zs) == 0:
            st.warning("No Z slices available for slice viewing.")
        else:
            z0_default = float(all_zs[np.argmin(np.abs(all_zs))])
            sel_z = st.select_slider("Select Master Z Slice (mm)", options=all_zs.tolist(), value=z0_default)

            c1, c2 = st.columns(2)

            with c1:
                file_2d = st.selectbox("📂 Select file for 2D Heatmap:", list(datasets.keys())) if is_comp_mode else list(datasets.keys())[0]
                arts_2d = prepare_slice_artifacts(datasets[file_2d]["df"], sel_z, z_match_tol, grid_resolution, interp_method)

                if arts_2d["gx"] is None or arts_2d["gb"] is None:
                    st.warning("Not enough points in this slice to generate a 2D interpolated heatmap.")
                else:
                    fig_h = build_heatmap_plot(
                        arts_2d,
                        f"2D Heatmap ({file_2d}, Z={arts_2d['closest_z']:.3f} mm)",
                        global_bmin,
                        global_bmax,
                    )
                    st.plotly_chart(fig_h, use_container_width=True)

            with c2:
                topology_inputs = []
                for name, data in datasets.items():
                    topology_inputs.append({
                        "name": name,
                        "artifacts": prepare_slice_artifacts(data["df"], sel_z, z_match_tol, grid_resolution, interp_method)
                    })

                fig_s = build_topology_comparison_plot(topology_inputs, global_bmin, global_bmax, is_comp_mode)
                if fig_s is None:
                    st.warning("No valid interpolated slices available for 3D topology comparison.")
                else:
                    st.plotly_chart(fig_s, use_container_width=True)

    with t4:
        st.subheader("Field Homogeneity Deviation (%)")
        fig_hom = build_homogeneity_plot(datasets, grid_resolution, interp_method, z_match_tol)

        if fig_hom is None:
            st.warning("Unable to generate homogeneity deviation plots from the current data.")
        else:
            st.plotly_chart(fig_hom, use_container_width=True)

        st.divider()
        st.subheader(f"Volume Homogeneity Analysis ({vol_shape})")

        if vol_shape == "Cylinder":
            st.write(f"**Target Volume boundaries:** Radius = {vol_radius} mm, Length = {vol_length} mm")
        elif vol_shape == "Sphere":
            st.write(f"**Target Volume boundaries:** Radius = {vol_radius} mm")
        elif vol_shape == "Cube":
            st.write(f"**Target Volume boundaries:** Side Length = {cube_side} mm")
        else:
            st.write(f"**Target Volume boundaries:** X = {cuboid_x} mm, Y = {cuboid_y} mm, Z = {cuboid_z} mm")

        hom_df = make_homogeneity_dataframe(datasets, vol_shape, vol_radius, vol_length, cube_side, cuboid_dims)

        st.dataframe(
            hom_df.style.format(
                {
                    "Pk-Pk (PPM)": "{:.1f}",
                    "Min Field (mT)": "{:.4f}",
                    "Max Field (mT)": "{:.4f}",
                    "B0 (mT)": "{:.4f}",
                    "Center X (mm)": "{:.4f}",
                    "Center Y (mm)": "{:.4f}",
                    "Center Z (mm)": "{:.4f}",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

        hom_csv = hom_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Homogeneity Results (CSV)",
            hom_csv,
            "homogeneity_results.csv",
            "text/csv",
        )

else:
    st.info("Awaiting CSV file upload...")
