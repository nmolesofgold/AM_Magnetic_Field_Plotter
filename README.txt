# 🧲 Halbach Magnetic Field Analyzer

A high-performance, interactive Python application built with **Streamlit** and **Plotly** for analyzing magnetic field scan data. 

## 🚀 Features
* **Interactive 3D Visualization:** Zoom, rotate, and pan through your point cloud.
* **Automatic Center Detection:** Locates Geometric X/Y and Magnetic Z centers.
* **Data Cleaning:** Toggleable option to remove calibration/air scans (the first Z-slice).

---

## 📊 Supported Data Format

The application expects a **CSV (.csv)** file using a comma (`,`) as the delimiter. For the most accurate results, ensure your data follows this structure:

### Required Columns
| Column Name | Units | Description |
| :--- | :--- | :--- |
| `X_mm` | mm | The physical X-coordinate of the sensor. |
| `Y_mm` | mm | The physical Y-coordinate of the sensor. |
| `Z_mm` | mm | The physical Z-coordinate (depth) of the sensor. |
| `Field` | Tesla (T) | The magnetic field reading (magnitude). |

*Note: The app also accepts `Magnetic_Field_Reading` as a substitute for the `Field` column name.*

### CSV Example Snippet
```csv
X_mm,Y_mm,Z_mm,Field
125.5,-26.6,-4.0,0.000182
126.0,-26.6,-4.0,0.000472
125.99,-26.51,-4.0,0.001102

## 🚀 Features

* **Automatic Center Detection:** * Finds the **Geometric X/Y Center** based on scan boundaries.
    * Identifies the **Magnetic Z-Center** by locating the peak field along the central axis.
* **Interactive 3D Visualization:** Fully rotatable and zoomable 3D point cloud with a reference cylinder and center axis marker.
* **Dynamic Slice Viewer:** Select any Z-position via a slider to view:
    * 2D Heatmaps of the field distribution.
    * 3D Field Topology (height map of field strength).
    * Real-time homogeneity statistics in PPM (Parts Per Million).
* **Homogeneity Profiling:** Automated deviation plots for X, Y, and Z axes relative to the peak center field ($B_0$).
* **Data Cleaning:** Automatically filters out the first Z-slice encountered in the file to remove calibration or "air" scans.

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have [Python 3.8+](https://www.python.org/downloads/) installed on your system.

### 2. Automatic Setup (Recommended)
Double-click the **`setup_environment.bat`** file. This will:
* Upgrade your Python package manager (pip).
* Install all required libraries (`pandas`, `scipy`, `streamlit`, `plotly`, etc.).

### 3. Manual Setup (Optional)
If you prefer the command line:
```bash
pip install -r requirements.txt