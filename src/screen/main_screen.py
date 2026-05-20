import os
import sys
import time
import threading
from datetime import datetime
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
import psutil

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from .splash_screen import SplashScreen
from .group_screen import GroupScreen
from .report_screen import ReportScreen
from .navbar import TopNavbar 

ctk.set_appearance_mode("System")  
ctk.set_default_color_theme("blue") 

FONT_FAMILY = "Segoe UI"
FONT_TITLE = (FONT_FAMILY, 28, "bold")
FONT_SUBTITLE = (FONT_FAMILY, 20, "bold")
FONT_SECTION = (FONT_FAMILY, 15, "bold")
FONT_BODY = (FONT_FAMILY, 13, "normal")
FONT_STATS = (FONT_FAMILY, 12, "bold")
FONT_LOG = ("Consolas", 11, "normal")

# ==========================================
# THREAD SAFE CONSOLE UNTUK LOG UI
# ==========================================
class ThreadSafeConsole:
    def __init__(self, textbox, app):
        self.textbox = textbox
        self.app = app

    def write(self, msg):
        if msg.strip():
            self.app.after(0, self._append, msg)

    def _append(self, msg):
        self.textbox.configure(state="normal")
        self.textbox.insert(tk.END, msg + "\n")
        self.textbox.see(tk.END) 
        self.textbox.configure(state="disabled")

    def flush(self): pass


# ==========================================
# MAIN APPLICATION GUI
# ==========================================
class FaceRecognitionApp(ctk.CTk):
    def __init__(self, recognizer):
        super().__init__()
        self.withdraw()
        
        # Menerima injeksi logika AI dari main.py
        self.recognizer = recognizer
        
        self.title("Face Recognition App - Kelompok 7 - Informatika 2025D")
        self.geometry("1150x750") 
        
        # --- PERBAIKAN PATH (URUTANNYA HARUS SEPERTI INI) ---
        # 1. Definisikan dulu lokasi folder saat ini (folder 'screen')
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Baru gunakan variabel di atas untuk mencari folder 'assets' dan checkpoint model di 'src/dataset'
        self.assets_dir = os.path.join(self.current_dir, "..", "assets")
        self.model_path = os.path.join(self.current_dir, "..", "dataset", "eigen_weights.pt")
        # ----------------------------------------------------
        
        try:
            ico_path = os.path.join(self.assets_dir, "logo_uns.ico")
            png_path = os.path.join(self.assets_dir, "logo_uns.png")
            if os.path.exists(ico_path):
                self.iconbitmap(ico_path)
            elif os.path.exists(png_path):
                self.icon_img = tk.PhotoImage(file=png_path)
                self.after(200, lambda: self.wm_iconphoto(False, self.icon_img))
        except Exception:
            pass

        self.dataset_folder = ""
        self.test_image_path = ""
        self.test_preview_image = None
        self.result_preview_image = None

        self.init_icons()
        self.setup_ui()
        
        sys.stdout = ThreadSafeConsole(self.log_box, self)
        
        SplashScreen(self)
        self.update_hardware_stats()
    def init_icons(self):
        self.icons = {}
        icon_files = {
            "logo_uns": "logo_uns.png", 
            "folder": "ic_folder.png",
            "image": "ic_image.png",
            "analyze": "ic_analyze.png",
            "stats": "ic_stats.png",
            "logs": "ic_logs.png",
            "scan": "ic_scan.png",       
            "member": "ic_member.png"    
        }
        
        for key, filename in icon_files.items():
            path = os.path.join(self.assets_dir, filename)
            if os.path.exists(path):
                img = Image.open(path)
                if key == "logo_uns":
                    self.icons[key] = ctk.CTkImage(light_image=img, dark_image=img, size=(38, 38))
                else:
                    self.icons[key] = ctk.CTkImage(light_image=img, dark_image=img, size=(18, 18))
            else:
                self.icons[key] = None

    def setup_ui(self):
        self.grid_rowconfigure(0, weight=0) 
        self.grid_rowconfigure(1, weight=1) 
        self.grid_columnconfigure(0, weight=1)

        self.top_navbar = TopNavbar(
            self, 
            switch_callback=self.switch_page, 
            logo_img=self.icons.get("logo_uns"),
            icon_scan=self.icons.get("scan"),
            icon_member=self.icons.get("member"),
            icon_report=self.icons.get("stats")
        )
        self.top_navbar.grid(row=0, column=0, sticky="ew")

        self.content_container = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.content_container.grid(row=1, column=0, sticky="nsew")
        self.content_container.grid_rowconfigure(0, weight=1)
        self.content_container.grid_columnconfigure(0, weight=1)

        # ---------------- PAGE 1: DASHBOARD FRAME ----------------
        self.page_dashboard = ctk.CTkFrame(self.content_container, corner_radius=0, fg_color="transparent")
        self.page_dashboard.grid(row=0, column=0, sticky="nsew")
        
        self.page_dashboard.grid_columnconfigure(0, minsize=300, weight=0)
        self.page_dashboard.grid_columnconfigure(1, weight=1)
        self.page_dashboard.grid_rowconfigure(0, weight=1)

        # === SIDEBAR ===
        self.sidebar = ctk.CTkFrame(self.page_dashboard, width=300, corner_radius=0, fg_color=("gray95", "gray13"))
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(9, weight=1) 
        self.sidebar.grid_propagate(False) 

        ctk.CTkLabel(self.sidebar, text="1. Dataset Source", font=FONT_SECTION).grid(row=0, column=0, padx=25, pady=(35, 10), sticky="w")
        self.btn_dataset = ctk.CTkButton(self.sidebar, text="  Choose Folder", font=FONT_BODY, image=self.icons["folder"], compound="left", height=38, corner_radius=8)
        self.btn_dataset.configure(command=self.load_dataset)
        self.btn_dataset.grid(row=1, column=0, padx=25, pady=(0, 5), sticky="ew")
        self.lbl_dataset_status = ctk.CTkLabel(self.sidebar, text="No Folder Chosen", font=FONT_BODY, text_color="gray", width=250, anchor="w")
        self.lbl_dataset_status.grid(row=2, column=0, padx=25, pady=(0, 20), sticky="w")

        ctk.CTkLabel(self.sidebar, text="2. Target Image", font=FONT_SECTION).grid(row=3, column=0, padx=25, pady=(0, 10), sticky="w")
        self.btn_image = ctk.CTkButton(self.sidebar, text="  Choose File", font=FONT_BODY, image=self.icons["image"], compound="left", height=38, corner_radius=8)
        self.btn_image.configure(command=self.load_test_image)
        self.btn_image.grid(row=4, column=0, padx=25, pady=(0, 5), sticky="ew")
        self.lbl_image_status = ctk.CTkLabel(self.sidebar, text="No File Chosen", font=FONT_BODY, text_color="gray", width=250, anchor="w")
        self.lbl_image_status.grid(row=5, column=0, padx=25, pady=(0, 20), sticky="w")

        ctk.CTkLabel(self.sidebar, text="3. Analysis Settings", font=FONT_SECTION).grid(row=6, column=0, padx=25, pady=(0, 10), sticky="w")
        self.frame_settings = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.frame_settings.grid(row=7, column=0, padx=25, pady=(0, 20), sticky="ew")
        self.frame_settings.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(self.frame_settings, text="Tolerance Threshold:", font=FONT_BODY).grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.entry_threshold = ctk.CTkEntry(self.frame_settings, width=70, font=FONT_BODY, justify="center", height=28, corner_radius=6)
        self.entry_threshold.grid(row=0, column=1, sticky="e", pady=(0, 8))
        self.entry_threshold.insert(0, "3000.0") 
        
        ctk.CTkLabel(self.frame_settings, text="Eigen Components:", font=FONT_BODY).grid(row=1, column=0, sticky="w")
        self.entry_components = ctk.CTkEntry(self.frame_settings, width=70, font=FONT_BODY, justify="center", height=28, corner_radius=6)
        self.entry_components.grid(row=1, column=1, sticky="e")
        self.entry_components.insert(0, "50") 

        self.btn_analyze = ctk.CTkButton(
            self.sidebar, text="  START ANALYZE", font=(FONT_FAMILY, 15, "bold"), image=self.icons["analyze"], 
            compound="left", fg_color="gray40", hover_color="#059669", text_color="white", height=45, corner_radius=8, 
            state="disabled", command=self.start_analysis_thread
        )
        self.btn_analyze.grid(row=8, column=0, padx=25, pady=10, sticky="ew")

        self.frame_stats = ctk.CTkFrame(self.sidebar, fg_color=("gray85", "gray16"), corner_radius=12)
        self.frame_stats.grid(row=10, column=0, padx=25, pady=25, sticky="ew")
        
        ctk.CTkLabel(self.frame_stats, text="  System Performance", font=FONT_SECTION, image=self.icons["stats"], compound="left").pack(pady=(15, 10))
        self.lbl_cpu = ctk.CTkLabel(self.frame_stats, text="CPU Usage: 0%", font=FONT_STATS, text_color="#3b82f6", width=200, anchor="w")
        self.lbl_cpu.pack(padx=20)
        self.lbl_ram = ctk.CTkLabel(self.frame_stats, text="RAM Usage: 0%", font=FONT_STATS, text_color="#eab308", width=200, anchor="w")
        self.lbl_ram.pack(padx=20)
        self.lbl_gpu = ctk.CTkLabel(self.frame_stats, text="GPU Load : N/A", font=FONT_STATS, text_color="#10b981", width=200, anchor="w")
        self.lbl_gpu.pack(padx=20, pady=(0, 15))

        # === MAIN CONTENT ===
        self.main_frame = ctk.CTkFrame(self.page_dashboard, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=30, pady=20)
        self.main_frame.grid_columnconfigure((0, 1), weight=1)
        self.main_frame.grid_rowconfigure(2, weight=1) 

        ctk.CTkLabel(self.main_frame, text="Facial Recognition Dashboard", font=FONT_TITLE).grid(row=0, column=0, columnspan=2, pady=(10, 25))

        self.frame_images = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.frame_images.grid(row=1, column=0, columnspan=2, sticky="nsew")
        self.frame_images.grid_columnconfigure((0, 1), weight=1)

        self.box_test = ctk.CTkFrame(self.frame_images, corner_radius=12, fg_color=("gray90", "gray16"), border_width=1, border_color=("gray75", "gray25"))
        self.box_test.grid(row=0, column=0, padx=15, sticky="nsew")
        ctk.CTkLabel(self.box_test, text="Target Face Input", font=FONT_SECTION).pack(pady=(15, 10))
        self.canvas_test = ctk.CTkLabel(self.box_test, text="(Awaiting Target File)", font=FONT_BODY, width=200, height=200, fg_color=("gray82", "gray12"), corner_radius=10)
        self.canvas_test.pack(pady=(0, 25), expand=True)

        self.box_result = ctk.CTkFrame(self.frame_images, corner_radius=12, fg_color=("gray90", "gray16"), border_width=1, border_color=("gray75", "gray25"))
        self.box_result.grid(row=0, column=1, padx=15, sticky="nsew")
        ctk.CTkLabel(self.box_result, text="Mathematic Closest Match", font=FONT_SECTION).pack(pady=(15, 10))
        self.canvas_result = ctk.CTkLabel(self.box_result, text="(Result Preview Area)", font=FONT_BODY, width=200, height=200, fg_color=("gray82", "gray12"), corner_radius=10)
        self.canvas_result.pack(pady=(0, 25), expand=True)

        self.frame_result_info = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.frame_result_info.grid(row=2, column=0, columnspan=2, pady=10, sticky="ew")
        
        self.lbl_result = ctk.CTkLabel(self.frame_result_info, text="System Standby", font=(FONT_FAMILY, 20, "bold"), text_color="gray")
        self.lbl_result.pack(pady=(10, 5))
        self.lbl_time = ctk.CTkLabel(self.frame_result_info, text="Execution Time: 0.00s", font=FONT_BODY)
        self.lbl_time.pack()

        self.progress_bar = ctk.CTkProgressBar(self.frame_result_info, width=500, height=8, corner_radius=5, mode="determinate")
        self.progress_bar.pack(pady=15)
        self.progress_bar.set(0)

        ctk.CTkLabel(self.main_frame, text="  Terminal Logs Engine:", font=FONT_SECTION, image=self.icons["logs"], compound="left").grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 5))
        self.log_box = ctk.CTkTextbox(self.main_frame, height=150, corner_radius=10, border_width=1, border_color=("gray75", "gray25"), state="disabled", font=FONT_LOG)
        self.log_box.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=(0, 10))

        # ---------------- PAGE 2: REPORT SCREEN FRAME ----------------
        self.page_report = ReportScreen(self.content_container)
        self.page_report.grid(row=0, column=0, sticky="nsew")

        # ---------------- PAGE 3: GROUP SCREEN FRAME ----------------
        self.page_group = GroupScreen(self.content_container)
        self.page_group.grid(row=0, column=0, sticky="nsew")

        self.top_navbar.on_tab_click("dashboard")

    def switch_page(self, page_name):
        if page_name == "dashboard":
            self.page_dashboard.tkraise()
        elif page_name == "report":
            self.page_report.tkraise()
        elif page_name == "group":
            self.page_group.tkraise()

    def update_hardware_stats(self):
        try:
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            self.lbl_cpu.configure(text=f"CPU Usage: {cpu}%")
            self.lbl_ram.configure(text=f"RAM Usage: {ram}%")
            if GPU_AVAILABLE:
                gpus = GPUtil.getGPUs()
                if gpus:
                    self.lbl_gpu.configure(text=f"GPU Load : {gpus[0].load*100:.0f}% | VRAM: {gpus[0].memoryUtil*100:.0f}%")
        except Exception:
            pass
        self.after(1000, self.update_hardware_stats)

    def check_ready_to_analyze(self):
        if self.test_image_path:
            self.btn_analyze.configure(state="normal", fg_color="#10b981")

    def _shorten_filename(self, path, max_chars=28):
        name = os.path.basename(path)
        if len(name) <= max_chars:
            return name
        return name[: max_chars - 3] + "..."

    def load_dataset(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.dataset_folder = folder_selected
            self.lbl_dataset_status.configure(text=self._shorten_filename(folder_selected))
            self.check_ready_to_analyze()

    def load_test_image(self):
        file_selected = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_selected:
            self.test_image_path = file_selected
            self.lbl_image_status.configure(text=self._shorten_filename(file_selected))
            img = Image.open(self.test_image_path)
            w, h = img.size
            s = min(w, h)
            img = img.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2))
            self.test_preview_image = ctk.CTkImage(light_image=img, dark_image=img, size=(200, 200))
            self.canvas_test.configure(image=self.test_preview_image, text="")
            self.check_ready_to_analyze()

    def start_analysis_thread(self):
        self.btn_analyze.configure(state="disabled", fg_color="gray40")
        self.lbl_result.configure(text="Analyzing Vector Spaces...", text_color="#3b82f6")
        self.canvas_result.configure(image="", text="Processing...")
        self.progress_bar.set(0)
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", tk.END)
        self.log_box.configure(state="disabled")
        threading.Thread(target=self.run_recognition, daemon=True).start()

    def _collect_system_snapshot(self):
        """Mengambil snapshot performa sistem saat ini."""
        snap = {
            "time": time.time(),
            "cpu": psutil.cpu_percent(interval=0),
            "ram": psutil.virtual_memory().percent,
            "gpu_load": -1,
            "gpu_vram": -1,
        }
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    snap["gpu_load"] = gpus[0].load * 100
                    snap["gpu_vram"] = gpus[0].memoryUtil * 100
            except Exception:
                pass
        return snap

    def _monitor_system(self, snapshots, stop_event):
        """Background thread: kumpulkan snapshot performa setiap 0.5 detik."""
        while not stop_event.is_set():
            snapshots.append(self._collect_system_snapshot())
            stop_event.wait(0.5)

    def run_recognition(self):
            start_time = time.time()
        
            # Mulai monitoring sistem
            system_snapshots = []
            stop_monitor = threading.Event()
            monitor_thread = threading.Thread(
                target=self._monitor_system, args=(system_snapshots, stop_monitor), daemon=True
            )
            monitor_thread.start()
        
            try:
                THRESHOLD = float(self.entry_threshold.get())
            except ValueError:
                print("⚠️ Input Threshold tidak valid! Menggunakan default: 3000.0")
                THRESHOLD = 3000.0
            
            try:
                NUM_COMPONENTS = int(self.entry_components.get())
            except ValueError:
                print("⚠️ Input Komponen Eigen tidak valid! Menggunakan default: 50")
                NUM_COMPONENTS = 50
            
            print("=== MEMULAI ANALISIS RUANG VEKTOR EIGEN ===")
            print(f"Ambang Toleransi (Threshold): {THRESHOLD}")
            print(f"Maks Komponen Eigen         : {NUM_COMPONENTS}\n")

            model_from_cache = self.recognizer.is_loaded

            # 1. LOAD PORTABLE SHARDED MODEL
            if not self.recognizer.is_loaded:
                # --- PERBAIKAN 1: Update UI Progress Bar secara aman melalui self.after ---
                self.after(0, self.progress_bar.set, 0.5)
            
                # --- PERBAIKAN 2: Beri jeda 0.1 detik agar OS merender UI sebelum CPU dimonopoli ---
                time.sleep(0.1)
            
                try:
                    self.recognizer.load_model()
                except Exception as e:
                    print(f"ERROR: {e}")
                    self.after(0, self.show_error, "Model tidak ditemukan! Jalankan 'python build_model.py' di terminal terlebih dahulu.")
                    return
            else:
                print("[INFO] Model Portable sudah tersimpan di VRAM. Eksekusi instan.")
        
            # 2. INFERENCE
            closest_img_rgb, label, min_dist, msg = self.recognizer.recognize(self.test_image_path, THRESHOLD)
        
            # --- PERBAIKAN 3: Update UI Progress Bar akhir secara aman ---
            self.after(0, self.progress_bar.set, 1.0)
        
            print("-" * 45)
            if min_dist != float('inf'):
                print(f"📐 JARAK EUCLIDEAN TERKECIL : {min_dist:.2f}")
            print("-" * 45)

            end_time = time.time()
            exec_time = end_time - start_time
        
            # Hentikan monitoring dan ambil snapshot terakhir
            stop_monitor.set()
            system_snapshots.append(self._collect_system_snapshot())
        
            # Hitung info dataset
            dataset_total_images = len(self.recognizer.dataset_labels) if self.recognizer.is_loaded else 0
            dataset_total_classes = len(set(self.recognizer.dataset_labels)) if self.recognizer.is_loaded else 0
        
            # Hitung match percentage
            match_pct = None
            if msg == "Cocok":
                match_pct = max(0.0, 100.0 * (1 - (min_dist / THRESHOLD)))
        
            # Tentukan device
            import torch
            device_name = "CUDA" if torch.cuda.is_available() else "CPU"
        
            # Kumpulkan report data
            report_data = {
                "timestamp": datetime.now().strftime("%d %B %Y, %H:%M:%S"),
                "threshold": THRESHOLD,
                "num_components": NUM_COMPONENTS,
                "execution_time": exec_time,
                "result_label": label,
                "result_status": msg,
                "min_distance": min_dist,
                "match_percentage": match_pct,
                "test_image_path": self.test_image_path,
                "dataset_folder": self.dataset_folder,
                "dataset_total_images": dataset_total_images,
                "dataset_total_classes": dataset_total_classes,
                "model_loaded_from_cache": model_from_cache,
                "device": device_name,
                "system_snapshots": system_snapshots,
            }
            self.after(0, self.show_final_result, min_dist, THRESHOLD, closest_img_rgb, label, msg, exec_time, report_data)

    def show_final_result(self, min_dist, threshold, closest_img_rgb, label, msg, exec_time, report_data=None):
        self.lbl_time.configure(text=f"Execution Time: {exec_time:.2f}s")
        
        if msg == "Cocok":
            match_percentage = max(0.0, 100.0 * (1 - (min_dist / threshold)))
            print(f"✅ KESIMPULAN KERNEL: Pola cocok dengan individual '{label}' ({match_percentage:.1f}%)")
            self.lbl_result.configure(text=f"MATCH FOUND: {label.upper()} ({match_percentage:.1f}% Cocok)", text_color="#10b981")
            
            res_img = Image.fromarray(closest_img_rgb)
            self.result_preview_image = ctk.CTkImage(light_image=res_img, dark_image=res_img, size=(200, 200))
            self.canvas_result.configure(image=self.result_preview_image, text="")
        else:
            print(f"❌ KESIMPULAN KERNEL: {msg}")
            self.lbl_result.configure(text="UNKNOWN (NO MATCH)", text_color="#ef4444")
            self.canvas_result.configure(image="", text="(No Match Found)")
            
        print("=== ANALISIS DIAGNOSTIK SELESAI ===")
        self.btn_analyze.configure(state="normal", fg_color="#10b981")
        
        # Update Report Screen dan pindah ke tab report
        if report_data:
            self.page_report.update_report(report_data)
            self.switch_page("report")
            self.top_navbar.on_tab_click("report")

    def show_error(self, msg):
        self.lbl_result.configure(text="SYSTEM ERROR", text_color="#ef4444")
        self.canvas_result.configure(image="", text="Error")
        self.btn_analyze.configure(state="normal", fg_color="#10b981")
        messagebox.showerror("Error", msg)