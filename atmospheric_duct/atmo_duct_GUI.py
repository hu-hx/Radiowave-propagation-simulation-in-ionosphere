import sys
import os
import json
import matplotlib
import time
import numpy as np

matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QScrollArea, QGroupBox, QFormLayout, QLineEdit, QCheckBox,
                             QPushButton, QTabWidget, QProgressBar, QStatusBar,
                             QSlider, QLabel, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QIcon

# Ensure Chinese characters are displayed correctly in Matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False



def _dst1_ortho(x):
    """DST-I ortho, along axis=0"""
    N = x.shape[0]
    z = np.zeros((1,) + x.shape[1:], dtype=x.dtype)
    v = np.concatenate([z, x, z, -x[::-1]], axis=0)  # 反对称扩展，长2(N+1)

    if np.iscomplexobj(x):
        V = np.fft.fft(v, axis=0)
        result = -V[1:N+1].imag + 1j * V[1:N+1].real
    else:
        V = np.fft.rfft(v, axis=0)
        result = -V[1:N+1].imag

    return result / np.sqrt(2 * (N + 1))

def dst(x, type=1, axis=0, norm='ortho'):
    x = np.moveaxis(x, axis, 0)
    return np.moveaxis(_dst1_ortho(x), 0, axis)

# DST-I ortho 矩阵是正交的，所以逆变换 == 正变换
idst = dst

def Mmodel(z, M0, C0, width, zb, z_thick, C1, C2, Md, evap, impend):
    # 标准大气
    M1 = M0 + C0 * z

    # 蒸发波导
    z0 = 0.15           # m，粗糙长度/经验常数
    M2 = M0 + C0 * (z - width * np.log((z + z0) / z0))

    # 悬空波导
    M3 = np.zeros_like(z)

    idx1 = z <= zb
    idx2 = (z > zb) & (z <= zb + z_thick)
    idx3 = z > zb + z_thick

    # 波导层以下：正常大气 基础层
    M3[idx1] = M0 + C1 * z[idx1]

    # 波导层内：M下降，形成陷获层
    M3[idx2] = M0 + C1 * zb - Md * (z[idx2] - zb) / z_thick

    # 波导层以上：恢复正常正梯度
    M3[idx3] = M0 + C1 * zb - Md + C2 * (z[idx3] - zb - z_thick)

    if evap and impend:
        M = M3 + M2 - M1
    elif impend:
        M = M3
    elif evap:
        M = M2
    else:
        M = M1

    # 将M转换为等效折射率
    n = 1 + M * 1e-6

    return n


class CalcThread(QThread):
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            self.progress_signal.emit(0)
            
            p = self.params
            f = p['f_mhz'] * 1e6
            theta0 = np.deg2rad(p['theta0'])
            z0 = p['z0']
            hr = p['hr']
            beta = np.deg2rad(p['beta'])
            x_max = p['x_max']
            dx = p['dx']
            z_max = p['z_max']
            dz_lam = p['dz_lam']
            step = p['step']

            M0 = p['M0']
            C0 = p['C0']
            width = p['width']
            zb = p['zb']
            z_thick = p['z_thick']
            C1 = p['C1']
            C2 = p['C2']
            Md = p['Md']
            evap = p['evap']
            impend = p['impend']

            c = 3e8
            f_Mhz = f / 1e6
            lam = c / f
            k0 = 2 * np.pi / lam
            dz = lam / dz_lam
            d = 0.75

            Nx = int(np.floor(x_max / dx)) + 1
            x = np.arange(Nx) * dx
            x_km = x / 1e3

            z_max_wid = z_max / d
            Nz = int(np.floor(z_max_wid / dz))
            z = (np.arange(1, Nz + 1) * dz).reshape(-1, 1)

            z_down = np.arange(0, Nz, step)
            z_down = z_down[((z_down + 1) * dz) <= z_max + dz * step]
            Nz_down = len(z_down)

            zn = np.arange(1, Nz + 1).reshape(-1, 1)
            pz = zn * np.pi / (Nz + 1) / dz

            u = np.zeros((Nz_down, Nx), dtype=complex)

            p0 = k0 * np.sin(theta0)
            w0 = np.sqrt(2 * np.log(2)) / (k0 * np.sin(beta / 2))
            u_curr = np.exp(-((z - z0) ** 2) / w0 ** 2) * np.exp(1j * p0 * (z - z0)) \
                - np.exp(-((z + z0) ** 2) / w0 ** 2) * np.exp(-1j * p0 * (z + z0))
            u_curr = u_curr / (w0 * np.sqrt(np.pi))

            d_rev = 1 / (1 - d)
            w = 0.5 + 0.5 * np.cos((d_rev * np.pi * (z - d * z_max_wid)) / z_max_wid)
            w[z < d * z_max_wid] = 1

            n = Mmodel(z, M0, C0, width, zb, z_thick, C1, C2, Md, evap, impend)
            M_profile = (n - 1) * 1e6

            k1 = np.exp(1j * k0 * dx * (n - 2))
            k2 = np.exp(1j * dx * np.sqrt(k0 ** 2 - pz ** 2 + 0j))

            for j in range(Nx - 1):
                dst_u = dst(u_curr, type=1, axis=0, norm='ortho')
                u[:, j] = u_curr[z_down, 0]
                u_curr = k1 * idst(k2 * dst_u, type=1, axis=0, norm='ortho') * w
                
                if j % 20 == 0 or j == Nx - 2:
                    self.progress_signal.emit(int(j / (Nx - 1) * 90))

            u[:, -1] = u_curr[z_down, 0]

            z = z[z_down]
            z_km = z / 1e3

            self.progress_signal.emit(95)
            u = np.abs(u)
            r_km = np.sqrt(x_km.reshape(1, -1) ** 2 + (z_km - z0 / 1e3) ** 2)

            L0 = 32.45 + 20 * np.log10(f_Mhz) + 20 * np.log10(r_km)
            F = 20 * np.log10(np.sqrt(r_km) * np.abs(u) + 1e-5)
            PL = L0 - F

            self.progress_signal.emit(100)
            self.result_signal.emit({
                'x_km': x_km,
                'z': z,
                'F': F,
                'PL': PL,
                'M': M_profile[z_down, 0]
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_signal.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('大气波导模拟GUI')
        # 窗口左上角图标
        if hasattr(sys, '_MEIPASS'):
            icon_path = os.path.join(sys._MEIPASS, '波导软件图标.ico')
        else:
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '波导软件图标.ico')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.setMinimumSize(1200, 700)

        # Store calculated results
        self.calc_results = None

        self.setStyleSheet("""
            QGroupBox#antenna_group {
                background-color: #e3f2fd;
                border: 1px solid #90caf9;
                border-radius: 6px;
                margin-top: 2ex;
                padding-top: 1ex;
            }
            QGroupBox#grid_group {
                background-color: #f1f8e9;
                border: 1px solid #aed581;
                border-radius: 6px;
                margin-top: 2ex;
                padding-top: 1ex;
            }
            QGroupBox#duct_group {
                background-color: #fff3e0;
                border: 1px solid #ffb74d;
                border-radius: 6px;
                margin-top: 2ex;
                padding-top: 1ex;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                left: 10px;
                color: #333;
                font-weight: bold;
            }
            QPushButton#calc_button {
                background-color: #1976d2;
                color: white;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton#calc_button:hover {
                background-color: #1565c0;
            }
            QPushButton#calc_button:disabled {
                background-color: #b0bec5;
            }
        """)

        self.init_ui()
        self.init_menu()
        self.load_defaults()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ====== Left Panel (Parameters) ======
        left_panel = QScrollArea()
        left_panel.setFixedWidth(350)
        left_panel.setWidgetResizable(True)
        param_widget = QWidget()
        self.param_layout = QVBoxLayout(param_widget)

        # 1. Antenna Params
        gb_antenna = QGroupBox("天线参数")
        gb_antenna.setObjectName("antenna_group")
        form_antenna = QFormLayout()
        self.in_f_mhz = QLineEdit()
        self.in_theta0 = QLineEdit()
        self.in_z0 = QLineEdit()
        self.in_hr = QLineEdit()
        self.in_beta = QLineEdit()
        form_antenna.addRow("频率 (MHz):", self.in_f_mhz)
        form_antenna.addRow("仰角 (°):", self.in_theta0)
        form_antenna.addRow("天线高度 (m):", self.in_z0)
        form_antenna.addRow("接收天线高度 (m):", self.in_hr)
        form_antenna.addRow("波束宽度 (°):", self.in_beta)
        gb_antenna.setLayout(form_antenna)

        # 2. Grid Params
        gb_grid = QGroupBox("网格参数")
        gb_grid.setObjectName("grid_group")
        form_grid = QFormLayout()
        self.in_x_max = QLineEdit()
        self.in_dx = QLineEdit()
        self.in_z_max = QLineEdit()
        self.in_dz_lam = QLineEdit()
        self.in_step = QLineEdit()
        form_grid.addRow("最大距离 (km):", self.in_x_max)
        form_grid.addRow("距离步长 (km):", self.in_dx)
        form_grid.addRow("最大高度 (m):", self.in_z_max)
        form_grid.addRow("lambda/dz:", self.in_dz_lam)
        form_grid.addRow("降采样步长:", self.in_step)
        gb_grid.setLayout(form_grid)

        # 3. Duct Params
        gb_duct = QGroupBox("波导参数")
        gb_duct.setObjectName("duct_group")
        form_duct = QFormLayout()
        self.in_M0 = QLineEdit()
        self.in_C0 = QLineEdit()
        self.cb_evap = QCheckBox("蒸发波导")
        self.cb_evap.setChecked(True)
        self.cb_evap.toggled.connect(self.on_evap_toggled)
        self.in_width = QLineEdit()
        self.cb_impend = QCheckBox("悬空波导")
        self.cb_impend.setChecked(True)
        self.cb_impend.toggled.connect(self.on_impend_toggled)
        self.in_zb = QLineEdit()
        self.in_z_thick = QLineEdit()
        self.in_C1 = QLineEdit()
        self.in_C2 = QLineEdit()
        self.in_Md = QLineEdit()

        form_duct.addRow("底修正折射率M0 (M-unit):", self.in_M0)
        form_duct.addRow("大气梯度C0 (M-unit/m):", self.in_C0)
        form_duct.addRow(self.cb_evap)
        form_duct.addRow("蒸发波导高度 (m):", self.in_width)
        form_duct.addRow(self.cb_impend)
        form_duct.addRow("悬空波导底高 (m):", self.in_zb)
        form_duct.addRow("悬空波导厚度 (m):", self.in_z_thick)
        form_duct.addRow("下层M梯度C1 (M-unit/m):", self.in_C1)
        form_duct.addRow("上层M梯度C2 (M-unit/m):", self.in_C2)
        form_duct.addRow("M亏损量Md (M-unit):", self.in_Md)
        gb_duct.setLayout(form_duct)

        # 4. Action Button
        self.btn_calc = QPushButton("开始计算")
        self.btn_calc.setObjectName("calc_button")
        self.btn_calc.setFixedHeight(40)
        self.btn_calc.clicked.connect(self.start_calculation)

        self.param_layout.addWidget(gb_antenna)
        self.param_layout.addWidget(gb_grid)
        self.param_layout.addWidget(gb_duct)
        self.param_layout.addStretch()
        self.param_layout.addWidget(self.btn_calc)
        
        left_panel.setWidget(param_widget)

        # ====== Right Panel (Plots) ======
        self.tabs = QTabWidget()
        
        # Tab 1: Propagation Factor
        self.tab_factor = QWidget()
        tab_factor_layout = QVBoxLayout(self.tab_factor)
        self.fig_factor = Figure()
        self.canvas_factor = FigureCanvas(self.fig_factor)
        self.toolbar_factor = NavigationToolbar(self.canvas_factor, self)
        tab_factor_layout.addWidget(self.toolbar_factor)
        tab_factor_layout.addWidget(self.canvas_factor)
        self.ax_factor = self.fig_factor.add_subplot(111)

        # Tab 2: Propagation Loss
        self.tab_loss = QWidget()
        tab_loss_layout = QVBoxLayout(self.tab_loss)
        
        plot_loss_area = QWidget()
        plot_loss_layout = QHBoxLayout(plot_loss_area)
        
        self.fig_loss = Figure()
        self.canvas_loss = FigureCanvas(self.fig_loss)
        self.ax_loss = self.fig_loss.add_subplot(111)
        
        # Slider for Receiver Height
        slider_layout = QVBoxLayout()
        self.lbl_hr_val = QLabel("接收高度: -- m")
        self.slider_hr = QSlider(Qt.Orientation.Vertical)
        self.slider_hr.setMinimum(0)
        self.slider_hr.setMaximum(400)
        self.slider_hr.setValue(10)
        self.slider_hr.valueChanged.connect(self.on_slider_changed)
        slider_layout.addWidget(self.lbl_hr_val)
        slider_layout.addWidget(self.slider_hr, alignment=Qt.AlignmentFlag.AlignHCenter)
        
        plot_loss_layout.addWidget(self.canvas_loss, stretch=1)
        plot_loss_layout.addLayout(slider_layout)
        
        self.toolbar_loss = NavigationToolbar(self.canvas_loss, self)
        tab_loss_layout.addWidget(self.toolbar_loss)
        tab_loss_layout.addWidget(plot_loss_area)

        # Tab 3: Modified Refractivity M
        self.tab_M = QWidget()
        tab_M_layout = QVBoxLayout(self.tab_M)
        self.fig_M = Figure()
        self.canvas_M = FigureCanvas(self.fig_M)
        self.toolbar_M = NavigationToolbar(self.canvas_M, self)
        tab_M_layout.addWidget(self.toolbar_M)
        tab_M_layout.addWidget(self.canvas_M)
        self.ax_M = self.fig_M.add_subplot(111)

        self.tabs.addTab(self.tab_factor, "传播因子图")
        self.tabs.addTab(self.tab_loss, "传播损耗图")
        self.tabs.addTab(self.tab_M, "修正折射率M图")

        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.tabs, stretch=1)

        # ====== Status Bar ======
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedWidth(200)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def init_menu(self):
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu("文件")
        
        action_save_preset = file_menu.addAction("保存参数预设")
        action_save_preset.setShortcut("Ctrl+S")
        action_save_preset.triggered.connect(self.save_preset)
        
        action_load_preset = file_menu.addAction("加载参数预设")
        action_load_preset.setShortcut("Ctrl+O")
        action_load_preset.triggered.connect(self.load_preset)
        
        file_menu.addSeparator()
        
        action_save_factor_img = file_menu.addAction("保存传播因子图")
        action_save_factor_img.triggered.connect(lambda: self.save_fig(self.fig_factor, "传播因子图.png"))
        
        action_save_loss_img = file_menu.addAction("保存传播损耗图")
        action_save_loss_img.triggered.connect(lambda: self.save_fig(self.fig_loss, "传播损耗图.png"))
        
        action_save_M_img = file_menu.addAction("保存修正折射率M图")
        action_save_M_img.triggered.connect(lambda: self.save_fig(self.fig_M, "修正折射率M图.png"))
        
        file_menu.addSeparator()
        
        action_export_loss = file_menu.addAction("导出传播损耗数据")
        action_export_loss.triggered.connect(self.export_loss_data)
        
        action_export_factor = file_menu.addAction("导出传播因子数据")
        action_export_factor.triggered.connect(self.export_factor_data)
        
        action_export_M = file_menu.addAction("导出修正折射率M数据")
        action_export_M.triggered.connect(self.export_M_data)
        
        file_menu.addSeparator()
        
        action_exit = file_menu.addAction("退出")
        action_exit.setShortcut("Ctrl+Q")
        action_exit.triggered.connect(self.close)
        
        # Help Menu
        help_menu = menubar.addMenu("帮助")
        action_about = help_menu.addAction("关于")
        action_about.triggered.connect(self.show_about)

    def load_defaults(self):
        self.in_f_mhz.setText("2e3")
        self.in_theta0.setText("0")
        self.in_z0.setText("15")
        self.in_hr.setText("10")
        self.in_beta.setText("3")
        
        self.in_x_max.setText("250")
        self.in_dx.setText("0.1")
        self.in_z_max.setText("400")
        self.in_dz_lam.setText("2")
        self.in_step.setText("10")
        
        self.in_M0.setText("338.5")
        self.in_C0.setText("0.117")
        self.in_width.setText("30")
        self.in_zb.setText("200")
        self.in_z_thick.setText("100")
        self.in_C1.setText("0.117")
        self.in_C2.setText("0.117")
        self.in_Md.setText("30")

    def on_evap_toggled(self, checked):
        self.in_width.setEnabled(checked)

    def on_impend_toggled(self, checked):
        self.in_zb.setEnabled(checked)
        self.in_z_thick.setEnabled(checked)
        self.in_C1.setEnabled(checked)
        self.in_C2.setEnabled(checked)
        self.in_Md.setEnabled(checked)

    def get_params(self):
        try:
            return {
                'f_mhz': float(self.in_f_mhz.text()),
                'theta0': float(self.in_theta0.text()),
                'z0': float(self.in_z0.text()),
                'hr': float(self.in_hr.text()),
                'beta': float(self.in_beta.text()),
                
                'x_max': float(self.in_x_max.text()) * 1e3,
                'dx': float(self.in_dx.text()) * 1e3,
                'z_max': float(self.in_z_max.text()),
                'dz_lam': float(self.in_dz_lam.text()),
                'step': int(self.in_step.text()),
                
                'M0': float(self.in_M0.text()),
                'C0': float(self.in_C0.text()),
                'evap': self.cb_evap.isChecked(),
                'width': float(self.in_width.text()),
                'impend': self.cb_impend.isChecked(),
                'zb': float(self.in_zb.text()),
                'z_thick': float(self.in_z_thick.text()),
                'C1': float(self.in_C1.text()),
                'C2': float(self.in_C2.text()),
                'Md': float(self.in_Md.text())
            }
        except ValueError as e:
            QMessageBox.critical(self, "输入错误", "请检查参数格式是否正确！")
            return None

    def start_calculation(self):
        params = self.get_params()
        if not params:
            return
            
        self.btn_calc.setEnabled(False)
        self.status_bar.showMessage("正在计算...")
        self.progress_bar.setValue(0)
        
        self.calc_thread = CalcThread(params)
        self.calc_thread.progress_signal.connect(self.progress_bar.setValue)
        self.calc_thread.result_signal.connect(self.on_calc_finished)
        self.calc_thread.error_signal.connect(self.on_calc_error)
        self.calc_thread.start()

    def on_calc_finished(self, results):
        self.calc_results = results
        params = self.get_params()
        
        self.progress_bar.setValue(100)
        self.status_bar.showMessage("计算完成")
        self.btn_calc.setEnabled(True)
        
        # Reset the view bounds
        self.ax_loss_xlim = False
        
        # Update Slider
        self.slider_hr.setMaximum(int(params['z_max']))
        self.slider_hr.setValue(int(params['hr']))
        self.lbl_hr_val.setText(f"接收高度: {int(params['hr'])} m")
        
        self.update_plots()

    def on_calc_error(self, err_msg):
        self.btn_calc.setEnabled(True)
        self.status_bar.showMessage("计算出错")
        QMessageBox.critical(self, "计算错误", f"计算过程中发生错误:\n{err_msg}")

    def update_plots(self):
        if not self.calc_results:
            return
            
        params = self.get_params()
        f_ghz = params['f_mhz'] / 1e3
        z0 = params['z0']
        
        x_km = self.calc_results['x_km']
        z = self.calc_results['z']
        F = self.calc_results['F']
        
        # Plot Factor
        self.fig_factor.clear()
        self.ax_factor = self.fig_factor.add_subplot(111)
        pcm = self.ax_factor.pcolormesh(x_km, z[:, 0], F, shading='auto', cmap='jet')
        self.fig_factor.colorbar(pcm, ax=self.ax_factor)
        self.ax_factor.set_xlim(0, params['x_max']/1000)
        self.ax_factor.set_ylim(0, params['z_max'])
        self.ax_factor.set_xlabel('距离/km')
        self.ax_factor.set_ylabel('高度/m')
        self.ax_factor.set_title(f'传播因子图')
        self.canvas_factor.draw()
        
        # Plot M profile
        M = self.calc_results['M']
        self.fig_M.clear()
        self.ax_M = self.fig_M.add_subplot(111)
        self.ax_M.plot(M, z[:, 0])
        self.ax_M.set_xlabel('修正折射率 M (M-unit)')
        self.ax_M.set_ylabel('高度 / m')
        self.ax_M.set_title('修正折射率 M 剖面图')
        self.ax_M.grid(True)
        self.canvas_M.draw()

        self.update_loss_plot()

    def on_slider_changed(self, value):
        self.lbl_hr_val.setText(f"接收高度: {value} m")
        self.update_loss_plot()

    def update_loss_plot(self):
        if not self.calc_results:
            return
            
        hr = self.slider_hr.value()
        params = self.get_params()
        f_ghz = params['f_mhz'] / 1e3
        z0 = params['z0']
        
        x_km = self.calc_results['x_km']
        z = self.calc_results['z']
        PL = self.calc_results['PL']
        
        hr_idx = np.argmin(np.abs(z[:, 0] - hr))
        PL_hr = PL[hr_idx, :]
        
        # Store current zoom limits if any
        if hasattr(self, 'ax_loss_xlim') and self.ax_loss_xlim:
            xlim = self.ax_loss.get_xlim()
            ylim = self.ax_loss.get_ylim()
        else:
            xlim = (0, params['x_max']/1000)
            ylim = (np.min(PL), np.max(PL))

        self.fig_loss.clear()
        self.ax_loss = self.fig_loss.add_subplot(111)
        self.ax_loss.plot(x_km, PL_hr)
        self.ax_loss.set_xlabel('距离/km')
        self.ax_loss.set_ylabel('PL/dB')
        self.ax_loss.set_title(f'传播损耗图 接收天线高度{hr}m')
        
        self.ax_loss.set_xlim(xlim)
        if ylim:
            self.ax_loss.set_ylim(ylim)
            
        self.ax_loss_xlim = True # Enable limits retaining for subsequent slider moves
        
        self.canvas_loss.draw()

    # --- Menu Actions ---
    def save_preset(self):
        params = self.get_params()
        if not params: return
        file_path, _ = QFileDialog.getSaveFileName(self, "保存预设", "", "JSON Files (*.json)")
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(params, f, indent=4)
            self.status_bar.showMessage("预设已保存", 3000)

    def load_preset(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "加载预设", "", "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    params = json.load(f)
                self.in_f_mhz.setText(str(params.get('f_mhz', 5e3)))
                self.in_theta0.setText(str(params.get('theta0', 0)))
                self.in_z0.setText(str(params.get('z0', 15)))
                self.in_hr.setText(str(params.get('hr', 10)))
                self.in_beta.setText(str(params.get('beta', 3)))
                
                self.in_x_max.setText(str(params.get('x_max', 250000)/1000))
                self.in_dx.setText(str(params.get('dx', 100)/1000))
                self.in_z_max.setText(str(params.get('z_max', 400)))
                self.in_dz_lam.setText(str(params.get('dz_lam', 2)))
                self.in_step.setText(str(params.get('step', 10)))
                
                self.in_M0.setText(str(params.get('M0', 338.5)))
                self.in_C0.setText(str(params.get('C0', 0.117)))
                self.cb_evap.setChecked(params.get('evap', True))
                self.in_width.setText(str(params.get('width', 30)))
                self.cb_impend.setChecked(params.get('impend', True))
                self.in_zb.setText(str(params.get('zb', 200)))
                self.in_z_thick.setText(str(params.get('z_thick', 100)))
                self.in_C1.setText(str(params.get('C1', 0.117)))
                self.in_C2.setText(str(params.get('C2', 0.117)))
                self.in_Md.setText(str(params.get('Md', 30)))
                self.status_bar.showMessage("预设已加载", 3000)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法加载预设: {str(e)}")

    def save_fig(self, fig, default_name):
        file_path, _ = QFileDialog.getSaveFileName(self, "保存图像", default_name, "PNG Images (*.png);;All Files (*)")
        if file_path:
            fig.savefig(file_path, dpi=300, bbox_inches='tight')
            self.status_bar.showMessage("图像已保存", 3000)

    def export_loss_data(self):
        if not self.calc_results:
            QMessageBox.warning(self, "提示", "请先进行计算！")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "导出损耗数据", "PL_data.txt", "Text Files (*.txt)")
        if file_path:
            hr = self.slider_hr.value()
            z = self.calc_results['z']
            hr_idx = np.argmin(np.abs(z[:, 0] - hr))
            PL_hr = self.calc_results['PL'][hr_idx, :]
            x_km = self.calc_results['x_km']
            
            data = np.column_stack((x_km, PL_hr))
            np.savetxt(file_path, data, fmt='%.6f', header='Distance(km)\tPL(dB)', comments='')
            self.status_bar.showMessage("损耗数据已导出", 3000)

    def export_factor_data(self):
        if not self.calc_results:
            QMessageBox.warning(self, "提示", "请先进行计算！")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "导出因子数据", "F_data.txt", "Text Files (*.txt)")
        if file_path:
            x_km = self.calc_results['x_km']
            z = self.calc_results['z'][:, 0]
            F = self.calc_results['F']
            
            # Export format: X, Y, Value. We flatten the grid.
            X, Z = np.meshgrid(x_km, z)
            data = np.column_stack((X.flatten(), Z.flatten(), F.flatten()))
            np.savetxt(file_path, data, fmt='%.6f', header='Distance(km)\tHeight(m)\tFactor(dB)', comments='')
            self.status_bar.showMessage("因子数据已导出", 3000)

    def export_M_data(self):
        if not self.calc_results:
            QMessageBox.warning(self, "提示", "请先进行计算！")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "导出修正折射率M数据", "M_data.txt", "Text Files (*.txt)")
        if file_path:
            z = self.calc_results['z'][:, 0]
            M = self.calc_results['M']
            data = np.column_stack((M, z))
            np.savetxt(file_path, data, fmt='%.6f', header='M(M-unit)\tHeight(m)', comments='')
            self.status_bar.showMessage("修正折射率M数据已导出", 3000)

    def show_about(self):
        about_text = (
            "大气波导模拟 基于 PyQt6 开发\n\n"
            "计算方式：\n"
            "基于抛物方程的分布傅里叶算法\n"
            "极化：水平极化\n"
            "初始条件：通过高斯方向图和初始场的傅里叶变换求解\n"
            "上边界：平滑的吸收层\n"
            "下边界：PEC边界\n"
            "步进算法：Feit-Fleck抛物方程\n\n"
            "by huhaixiang 2026.4"
        )
        QMessageBox.about(self, "关于", about_text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
