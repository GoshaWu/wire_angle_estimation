import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import re
import io
import zipfile
from datetime import datetime

# =========================================================================
# 1. 核心算法部分 (保持不变，严格遵循 V2 要求)
# =========================================================================

def bwareaopen(img_bin, min_size):
    """移除小于 min_size 的连通区域"""
    img_bin = img_bin.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bin, connectivity=8)
    output = np.zeros_like(img_bin)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            output[labels == i] = 255
    return output > 0

def process_wire_image(file_name, img_input):
    """钢丝图像预处理"""
    if isinstance(img_input, str):
        original_img = cv2.imread(img_input)
    else:
        original_img = img_input

    if len(original_img.shape) == 3:
        gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = original_img

    _, bw_raw = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw_raw_bool = (bw_raw > 0)

    rows_sum = np.sum(bw_raw_bool, axis=1)
    cols_sum = np.sum(bw_raw_bool, axis=0)
    threshold_count = 100
    
    valid_rows = np.where(rows_sum >= threshold_count)[0]
    valid_cols = np.where(cols_sum >= threshold_count)[0]

    if len(valid_rows) == 0 or len(valid_cols) == 0:
        img_stage1 = gray_img
    else:
        img_stage1 = gray_img[valid_rows.min():valid_rows.max()+1, valid_cols.min():valid_cols.max()+1]

    s1_h, s1_w = img_stage1.shape
    x_start = int(round(s1_w * 0.1))
    y_start = int(round(s1_h * 0.1))
    width = int(round(s1_w * 0.8))
    height = int(round(s1_h * 0.8))
    
    img_stage2 = img_stage1[y_start:y_start+height, x_start:x_start+width]

    _, bw_stage2_raw = cv2.threshold(img_stage2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw_stage2 = cv2.bitwise_not(bw_stage2_raw) 
    bw_stage2_clean = bwareaopen(bw_stage2, 200)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw_stage2_clean.astype(np.uint8)*255, connectivity=8)
    
    if num_labels <= 1:
        img_cropped = img_stage2
    else:
        max_area = -1
        max_idx = -1
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > max_area:
                max_area = area
                max_idx = i
        x, y, w, h = stats[max_idx, :4]
        padding = 10
        x_new = max(0, x - padding)
        y_new = max(0, y - padding)
        w_new = min(img_stage2.shape[1] - x_new, w + 2*padding)
        h_new = min(img_stage2.shape[0] - y_new, h + 2*padding)
        img_cropped = img_stage2[y_new:y_new+h_new, x_new:x_new+w_new]

    H, W = img_cropped.shape
    angle_value = 90
    match = re.search(r'\d+(?=\.)', file_name)
    if match:
        try:
            angle_value = float(match.group(0))
        except:
            pass

    img_part1 = img_cropped.copy()
    img_part2 = img_cropped.copy()
    X_grid, Y_grid = np.meshgrid(np.arange(1, W + 1), np.arange(1, H + 1)) 

    if angle_value < 30:
        col_start = int(round(3 * W / 8))
        col_end = int(round(5 * W / 8))
        img_masked = img_cropped.copy()
        img_masked[:, col_start-1:col_end] = 255 
        mid_col = int(round(W / 2))
        img_part1 = img_masked[:, :mid_col]
        img_part2 = img_masked[:, mid_col:]
    else:
        _, bw_final_raw = cv2.threshold(img_cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw_final = bwareaopen(cv2.bitwise_not(bw_final_raw), 200)
        y_coords, x_coords = np.where(bw_final) 
        y_coords = y_coords + 1
        x_coords = x_coords + 1
        
        if len(y_coords) == 0:
            Ax, Ay = 1, H
        else:
            y_max = np.max(y_coords)
            tolerance = 5
            bottom_mask = y_coords >= (y_max - tolerance)
            x_bottom = x_coords[bottom_mask]
            y_bottom = y_coords[bottom_mask]
            min_x_idx = np.argmin(x_bottom)
            Ax = x_bottom[min_x_idx]
            Ay = y_bottom[min_x_idx]
            
        Bx = W
        By = H / 2.0
        val = (Bx - Ax) * (Y_grid - Ay) - (By - Ay) * (X_grid - Ax)
        mask2 = val >= 0
        mask1 = val < 0
        img_part1[~mask1] = 255
        img_part2[~mask2] = 255

    img_part1 = cv2.flip(img_part1, 1)
    img_part2 = cv2.flip(img_part2, 1)
    
    _, img_part1_bin = cv2.threshold(img_part1, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, img_part2_bin = cv2.threshold(img_part2, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return img_part1_bin, img_part2_bin

def get_fitting_line_left(img):
    rows, cols = img.shape
    zero_pixel_coords_top = []
    zero_pixel_coords_bottom = []
    
    for col in range(cols):
        column_data = img[:, col]
        non_zero_indices = np.where(column_data == 0)[0]
        if len(non_zero_indices) > 0:
            top_zero_index = non_zero_indices[0]
            bottom_zero_index = non_zero_indices[-1]
            zero_pixel_coords_top.append([top_zero_index, col])
            zero_pixel_coords_bottom.append([bottom_zero_index, col])
            
    if not zero_pixel_coords_top:
        return [0, 0], [0, 0], [0, 0]
        
    zero_pixel_coords_top = np.array(zero_pixel_coords_top)
    zero_pixel_coords_bottom = np.array(zero_pixel_coords_bottom)
    
    p_top = np.polyfit(zero_pixel_coords_top[:, 1], zero_pixel_coords_top[:, 0], 1)
    p_buttom = np.polyfit(zero_pixel_coords_bottom[:, 1], zero_pixel_coords_bottom[:, 0], 1)
    
    point_a = [0, 0]
    found_a = False
    for row in range(rows):
        row_data = img[row, :]
        zero_indices = np.where(row_data == 0)[0]
        if len(zero_indices) > 0:
            zero_index = zero_indices[-1]
            point_a = [zero_index, row]
            found_a = True
            break
            
    if not found_a:
        point_a = [0, 0]
        
    img_temp = img.copy()
    img_temp[:, point_a[0]:] = 1 
    
    Y, X = np.where(img_temp == 0)
    if len(X) == 0:
        p_mid = [0, 0]
    else:
        p_mid = np.polyfit(X, Y, 1)
        
    return p_top, p_buttom, p_mid

def calculate_midline(p1, p2):
    m_mid = (p1[0] + p2[0]) / 2.0
    b_mid = (p1[1] + p2[1]) / 2.0
    return [m_mid, b_mid]

def get_fitting_line_right(img):
    Y, X = np.where(img == 0)
    if len(X) == 0:
        return [0, 0], [0, 0], [0, 0], [], []
        
    x_len = np.max(X) - np.min(X)
    y_len = np.max(Y) - np.min(Y)
    rows, cols = img.shape
    
    zero_pixel_coords_left = []
    zero_pixel_coords_right = []
    
    if x_len <= y_len:
        for row in range(rows - 1, -1, -1):
            column_data = img[row, :]
            zero_indices = np.where(column_data == 0)[0]
            if len(zero_indices) > 0:
                zero_pixel_coords_left.append([zero_indices[0], row])
                zero_pixel_coords_right.append([zero_indices[-1], row])
    else:
        for col in range(cols):
            column_data = img[:, col]
            zero_indices = np.where(column_data == 0)[0]
            if len(zero_indices) > 0:
                zero_pixel_coords_left.append([col, zero_indices[0]])
                zero_pixel_coords_right.append([col, zero_indices[-1]])
                
    zero_pixel_coords_left = np.array(zero_pixel_coords_left)
    zero_pixel_coords_right = np.array(zero_pixel_coords_right)
    
    def crop_coords(coords):
        length = len(coords)
        if length > 5:
            start_idx = int(math.floor(length * 0.2))
            if start_idx < 1: start_idx = 0
            end_idx = int(math.floor(length * 0.9))
            return coords[start_idx : end_idx + 1]
        return coords

    zero_pixel_coords_left = crop_coords(zero_pixel_coords_left)
    zero_pixel_coords_right = crop_coords(zero_pixel_coords_right)
    
    if len(zero_pixel_coords_left) == 0:
        p_left = [0, 0]
    else:
        p_left = np.polyfit(zero_pixel_coords_left[:, 0], zero_pixel_coords_left[:, 1], 1)
        
    if len(zero_pixel_coords_right) == 0:
        p_right = [0, 0]
    else:
        p_right = np.polyfit(zero_pixel_coords_right[:, 0], zero_pixel_coords_right[:, 1], 1)
        
    p_mid = calculate_midline(p_left, p_right)
    return p_left, p_right, p_mid, X, Y

def calculate_angle_between_lines(m1, m2):
    if math.isinf(m1) or math.isinf(m2):
        if math.isinf(m1) and math.isinf(m2):
            angle = 0.0
        else:
            angle = math.pi / 2.0
    else:
        angle = math.atan2(abs(m1 - m2), 1 + m1 * m2)
        angle = math.degrees(angle)
        if m2 > 0:
            angle = 180 - angle
    return angle

def compute_fitting_curve_in_memory(file_names, real_angles, poly_order=6, outlier_threshold=1.0):
    """内存中处理拟合曲线并生成结果"""
    given_angles = np.zeros(len(file_names))
    for i, fname in enumerate(file_names):
        match = re.search(r'\d+(?=\.)', fname)
        if match:
            given_angles[i] = float(match.group(0))
        else:
            dot_idx = fname.find('.')
            if dot_idx != -1:
                try:
                    given_angles[i] = float(fname[:dot_idx])
                except:
                    given_angles[i] = np.nan
            else:
                given_angles[i] = np.nan

    # 异常值清洗
    unique_motor_angles = np.unique(given_angles)
    valid_indices = np.ones(len(given_angles), dtype=bool)
    outlier_count = 0
    nan_mask = np.isnan(given_angles) | np.isnan(real_angles)
    valid_indices[nan_mask] = False
    
    for u_angle in unique_motor_angles:
        if np.isnan(u_angle): continue
        idx = np.where(given_angles == u_angle)[0]
        idx = [x for x in idx if valid_indices[x]]
        if not idx: continue
        group_vals = real_angles[idx]
        ref_val = np.median(group_vals)
        is_outlier_group = np.abs(group_vals - ref_val) > outlier_threshold
        if np.any(is_outlier_group):
            bad_indices = np.array(idx)[is_outlier_group]
            valid_indices[bad_indices] = False
            outlier_count += len(bad_indices)

    given_angles_clean = given_angles[valid_indices]
    real_angles_clean = real_angles[valid_indices]
    
    data_clean = np.column_stack((given_angles_clean, real_angles_clean))
    data_clean = data_clean[data_clean[:, 0].argsort()]
    
    unique_x, unique_indices = np.unique(data_clean[:, 0], return_inverse=True)
    unique_y_mean = np.zeros_like(unique_x)
    for i in range(len(unique_x)):
        unique_y_mean[i] = np.mean(data_clean[unique_indices == i, 1])
        
    fitting_data_ave = np.column_stack((unique_x, unique_y_mean))
    
    x = fitting_data_ave[:, 0] # Given
    y = fitting_data_ave[:, 1] # Real
    
    if len(x) == 0:
        return None, "No valid data for fitting", None

    p_f = np.polyfit(x, y, poly_order) 
    p_g = np.polyfit(y, x, poly_order) 
    
    f_x = np.polyval(p_f, x)
    g_y = np.polyval(p_g, y)
    
    error_f_x = np.mean((y - f_x)**2)
    error_g_y = np.mean((x - g_y)**2)
    
    # 绘图 (Result Area 2/3)
    fig = plt.figure(figsize=(12, 5))
    
    # Plot 1: f(Given)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(given_angles, real_angles, 'r.', markersize=6, alpha=0.3, label='Raw Data')
    if outlier_count > 0:
        ax1.plot(given_angles[~valid_indices], real_angles[~valid_indices], 'kx', markersize=4, label='Outliers')
    ax1.plot(x, y, 'bo', markersize=4, label='Mean')
    ax1.plot(x, f_x, 'r-', linewidth=1.5, label='Fit')
    ax1.set_title(f'Real = f(Given) | MSE: {error_f_x:.4f}')
    ax1.set_xlabel('Given (Motor)')
    ax1.set_ylabel('Real (Measured)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Plot 2: g(Real)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(y, x, 'bo', markersize=4)
    ax2.plot(y, g_y, 'r-', linewidth=1.5)
    ax2.set_title(f'Given = g(Real) | MSE: {error_g_y:.4f}')
    ax2.set_xlabel('Real (Measured)')
    ax2.set_ylabel('Given (Motor)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()

    # 结果文本生成
    export_coeffs_g = list(p_g)
    if len(fitting_data_ave) > 0:
        export_coeffs_g.append(fitting_data_ave[0, 0] / fitting_data_ave[0, 1])
        export_coeffs_g.append(fitting_data_ave[0, 1])
    
    result_text = f"""---------------- 拟合结果 ----------------
时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
配置: Order={poly_order}, Threshold={outlier_threshold:.2f}
f(x) MSE: {error_f_x:.6f}
g(y) MSE: {error_g_y:.6f}

--- g(y) 系数 (Given = g(Real)) ---
{'  '.join([f'{c:.10g}' for c in p_g])}

--- 导出代码 (Coeffs + Ratio + FirstReal) ---
{','.join([f'{c:.10g}' for c in export_coeffs_g])}
"""
    return fig, result_text, export_coeffs_g

# =========================================================================
# 2. 界面与交互 (Streamlit)
# =========================================================================

st.set_page_config(page_title="钢丝角度自动拟合工具 v2.0", layout="wide", page_icon="📏")

# CSS 仅用于 Metric 卡片样式
st.markdown("""
<style>
    .block-container {padding-top: 2rem;}
    .metric-card {background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #007bff;}
</style>
""", unsafe_allow_html=True)

# --- A. 左侧边栏 (Information Sidebar) ---
with st.sidebar:
    st.header("📘 操作指南")
    st.markdown("""
    1. **上传文件** 将包含 `.bmp` 图像的文件批量拖入右侧区域。
       
    2. **开始处理** 确认文件加载后，点击 **🚀 开始处理** 按钮。
       
    3. **下载结果** 处理完成后，下载包含图表和数据报告的 ZIP 包。
    """)
    
    st.markdown("---")
    st.caption("钢丝角度自动拟合工具 v2.0")
    st.caption("© 2025 Technical Support")
    st.caption("📧 junhwu@stu.edu.cn")

# --- B. 主工作区 (Main Workspace) ---

# 标题区
st.title("📏 钢丝角度自动拟合工具")
st.markdown("零配置、全自动钢丝折弯角度测量与拟合系统")

# 检查处理状态
if 'processed' not in st.session_state:
    st.session_state.processed = False

# ==========================================
# 状态 1: 未处理 (显示上传控件)
# ==========================================
if not st.session_state.processed:
    uploaded_files = st.file_uploader(
        "拖拽或点击上传 .bmp 图像文件", 
        type=['bmp'], 
        accept_multiple_files=True,
        help="支持批量上传"
    )

    if uploaded_files:
        st.write(f"📂 已加载 {len(uploaded_files)} 个文件。")
        
        # 按钮逻辑
        if st.button("🚀 开始处理"):
            
            # 反馈区：进度条与日志
            progress_bar = st.progress(0)
            status_text = st.empty()
            log_area = st.empty()
            
            file_names = []
            angles_calculated = []
            
            # 预先排序文件
            def get_file_number(f):
                match = re.search(r'\d+(?=\.)', f.name)
                return float(match.group(0)) if match else 0.0
            
            sorted_files = sorted(uploaded_files, key=get_file_number)
            total_files = len(sorted_files)
            
            # --- 核心处理循环 ---
            valid_count = 0
            logs = []
            
            for i, uploaded_file in enumerate(sorted_files):
                # 更新进度
                progress = (i + 1) / total_files
                progress_bar.progress(progress)
                status_text.text(f"正在处理 ({i+1}/{total_files}): {uploaded_file.name}")
                
                # 实时日志 (显示最近3条)
                logs.append(f"Processing: {uploaded_file.name}")
                if len(logs) > 3: logs.pop(0)
                log_area.code("\n".join(logs))

                try:
                    # 读取图像 (Streamlit Bytes -> OpenCV)
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    if img is None: raise Exception("Decode failed")

                    # 1. 处理与裁剪
                    below_image, above_image = process_wire_image(uploaded_file.name, img)
                    
                    # 2. 拟合边缘
                    _, p_buttom, _ = get_fitting_line_left(above_image)
                    _, p_right, _, _, _ = get_fitting_line_right(below_image)
                    
                    # 3. 计算夹角
                    angle_est = calculate_angle_between_lines(p_buttom[0], p_right[0])
                    
                    file_names.append(uploaded_file.name)
                    angles_calculated.append(angle_est)
                    valid_count += 1
                    
                except Exception as e:
                    file_names.append(uploaded_file.name)
                    angles_calculated.append(np.nan)
                    logs.append(f"Error on {uploaded_file.name}: {str(e)}")
            
            status_text.text("图像处理完成，正在进行曲线拟合...")
            
            # --- 结果生成与保存 ---
            if valid_count > 0:
                fig, result_txt, _ = compute_fitting_curve_in_memory(
                    np.array(file_names), 
                    np.array(angles_calculated)
                )
                
                # 保存到 Session State
                st.session_state.fig = fig
                st.session_state.result_txt = result_txt
                st.session_state.file_names = file_names
                st.session_state.angles = angles_calculated
                st.session_state.processed = True
                
                # 强制刷新，触发 UI 状态变更
                st.rerun()
            else:
                st.error("没有成功处理任何图片。")

# ==========================================
# 状态 2: 已处理 (显示结果 + 清空按钮)
# ==========================================
else: # st.session_state.processed is True
    
    # 清空按钮 (点击后重置状态并 Rerun) 
    st.info("✅ 数据处理已完成。点击下方按钮可清空数据并重新上传。", icon="🔒")
    
    if st.button("🗑️ 清空数据并重新上传", type="secondary"):
        st.session_state.clear() # 清空所有缓存
        st.rerun()               # 刷新页面回到初始状态

    # 结果展示区 (从 Session State 读取)
    if 'fig' in st.session_state:
        st.markdown("---")
        
        # 布局: 左侧(2/3) 图表, 右侧(1/3) 数据
        col_chart, col_data = st.columns([2, 1])
        
        with col_chart:
            st.subheader("📊 拟合曲线")
            st.pyplot(st.session_state.fig)
            
        with col_data:
            st.subheader("📝 关键指标")
            # 解析 MSE 用于显示
            try:
                mse_match = re.search(r'g\(y\) MSE: ([\d\.]+)', st.session_state.result_txt)
                mse_val = mse_match.group(1) if mse_match else "N/A"
            except:
                mse_val = "N/A"
                
            st.markdown(f"""
            <div class="metric-card">
                <h4>MSE (Given = g(Real))</h4>
                <h2 style="color: #007bff;">{mse_val}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.text_area("详细拟合报告", st.session_state.result_txt, height=300)

        # 下载区
        st.markdown("### 📥 结果交付")
        
        # 创建 ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            # 1. 保存报告
            zf.writestr("fitting_report.txt", st.session_state.result_txt)
            
            # 2. 保存 CSV
            df = pd.DataFrame({'FileName': st.session_state.file_names, 'MeasuredAngle': st.session_state.angles})
            zf.writestr("raw_data.csv", df.to_csv(index=False))
            
            # 3. 保存图片
            img_buffer = io.BytesIO()
            st.session_state.fig.savefig(img_buffer, format='png')
            zf.writestr("fitting_curve.png", img_buffer.getvalue())
            
        st.download_button(
            label="下载完整结果 (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"wire_fitting_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip"
        )