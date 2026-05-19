# src/screen/report_screen.py
import os
import time
import customtkinter as ctk
import tkinter as tk
from PIL import Image
from datetime import datetime

# ==========================================
# KONFIGURASI TIPOGRAFI
# ==========================================
FONT_FAMILY = "Segoe UI"
FONT_TITLE = (FONT_FAMILY, 28, "bold")
FONT_SUBTITLE = (FONT_FAMILY, 14, "normal")
FONT_SECTION = (FONT_FAMILY, 16, "bold")
FONT_BODY = (FONT_FAMILY, 13, "normal")
FONT_BODY_BOLD = (FONT_FAMILY, 13, "bold")
FONT_STATS_LABEL = (FONT_FAMILY, 12, "normal")
FONT_STATS_VALUE = (FONT_FAMILY, 22, "bold")
FONT_SMALL = (FONT_FAMILY, 11, "normal")
FONT_TABLE_HEADER = (FONT_FAMILY, 12, "bold")
FONT_TABLE_CELL = (FONT_FAMILY, 12, "normal")
FONT_BADGE = (FONT_FAMILY, 11, "bold")


class ReportScreen(ctk.CTkFrame):
    """
    Layar Report yang menampilkan hasil analisis terakhir.
    Data di-inject dari luar melalui method update_report().
    """

    def __init__(self, parent):
        super().__init__(parent, fg_color="transparent")

        self.assets_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets"
        )

        # Data report (akan di-update setelah analisis selesai)
        self.report_data = None

        self.setup_ui()

    # ==========================================
    # SETUP UI LAYOUT
    # ==========================================
    def setup_ui(self):
        # Scrollable container utama
        self.scroll = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            corner_radius=0,
        )
        self.scroll.pack(fill="both", expand=True, padx=0, pady=0)
        self.scroll.grid_columnconfigure(0, weight=1)

        # Placeholder awal (belum ada data)
        self.placeholder_frame = ctk.CTkFrame(self.scroll, fg_color="transparent")
        self.placeholder_frame.pack(fill="both", expand=True, pady=100)

        self.lbl_no_data_icon = ctk.CTkLabel(
            self.placeholder_frame,
            text="📋",
            font=(FONT_FAMILY, 64),
        )
        self.lbl_no_data_icon.pack(pady=(0, 15))

        self.lbl_no_data = ctk.CTkLabel(
            self.placeholder_frame,
            text="Belum Ada Data Report",
            font=(FONT_FAMILY, 22, "bold"),
            text_color="gray",
        )
        self.lbl_no_data.pack(pady=(0, 8))

        self.lbl_no_data_sub = ctk.CTkLabel(
            self.placeholder_frame,
            text="Jalankan analisis terlebih dahulu di Dashboard untuk melihat report.",
            font=FONT_SUBTITLE,
            text_color="gray",
        )
        self.lbl_no_data_sub.pack()

        # Frame konten report (awalnya tersembunyi)
        self.report_content = ctk.CTkFrame(self.scroll, fg_color="transparent")

    # ==========================================
    # UPDATE DATA REPORT
    # ==========================================
    def update_report(self, data: dict):
        """
        Menerima data report dari MainScreen setelah analisis selesai.
        
        Expected data keys:
            - timestamp: str (waktu analisis)
            - threshold: float
            - num_components: int
            - execution_time: float (detik)
            - result_label: str atau None
            - result_status: str ("Cocok" / "Tidak Dikenal" / error msg)
            - min_distance: float
            - match_percentage: float atau None
            - test_image_path: str
            - dataset_folder: str
            - dataset_total_images: int
            - dataset_total_classes: int
            - model_loaded_from_cache: bool
            - device: str ("CUDA" / "CPU")
            - system_snapshots: list of dict [{"time": float, "cpu": float, "ram": float, "gpu_load": float, "gpu_vram": float}]
        """
        self.report_data = data

        # Sembunyikan placeholder
        self.placeholder_frame.pack_forget()

        # Bersihkan konten report lama
        for widget in self.report_content.winfo_children():
            widget.destroy()

        self.report_content.pack(fill="both", expand=True, padx=30, pady=(20, 30))

        self._build_header()
        self._build_summary_cards()
        self._build_analysis_detail()
        self._build_system_performance()

    # ==========================================
    # HEADER
    # ==========================================
    def _build_header(self):
        d = self.report_data
        header = ctk.CTkFrame(self.report_content, fg_color="transparent")
        header.pack(fill="x", pady=(10, 5))

        # Judul dan timestamp di baris yang sama
        title_row = ctk.CTkFrame(header, fg_color="transparent")
        title_row.pack(fill="x")
        title_row.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            title_row, text="📊  Analysis Report", font=FONT_TITLE
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            title_row,
            text=d.get("timestamp", ""),
            font=FONT_SMALL,
            text_color="gray",
        ).grid(row=0, column=1, sticky="e", padx=(0, 5))

        # Status badge
        status = d.get("result_status", "")
        is_match = status == "Cocok"

        badge_frame = ctk.CTkFrame(header, fg_color="transparent")
        badge_frame.pack(fill="x", pady=(8, 0))

        badge_color = "#059669" if is_match else "#dc2626"
        badge_text = "✅  MATCH FOUND" if is_match else "❌  NO MATCH"

        badge = ctk.CTkLabel(
            badge_frame,
            text=badge_text,
            font=FONT_BADGE,
            text_color="white",
            fg_color=badge_color,
            corner_radius=6,
            width=140,
            height=28,
        )
        badge.pack(anchor="w")

        # Divider
        ctk.CTkFrame(header, height=1, fg_color=("gray80", "gray25")).pack(
            fill="x", pady=(15, 0)
        )

    # ==========================================
    # KARTU RINGKASAN (4 KARTU)
    # ==========================================
    def _build_summary_cards(self):
        d = self.report_data
        cards_frame = ctk.CTkFrame(self.report_content, fg_color="transparent")
        cards_frame.pack(fill="x", pady=(20, 10))
        cards_frame.grid_columnconfigure((0, 1, 2, 3), weight=1, uniform="card")

        # Data untuk kartu
        exec_time = d.get("execution_time", 0)
        min_dist = d.get("min_distance", float("inf"))
        match_pct = d.get("match_percentage", None)
        num_comp = d.get("num_components", 0)

        card_configs = [
            {
                "icon": "⏱️",
                "label": "Execution Time",
                "value": f"{exec_time:.3f}s",
                "color": "#3b82f6",
            },
            {
                "icon": "📐",
                "label": "Euclidean Distance",
                "value": f"{min_dist:.2f}" if min_dist != float("inf") else "∞",
                "color": "#eab308",
            },
            {
                "icon": "🎯",
                "label": "Match Confidence",
                "value": f"{match_pct:.1f}%" if match_pct is not None else "N/A",
                "color": "#10b981" if match_pct and match_pct > 50 else "#ef4444",
            },
            {
                "icon": "🧮",
                "label": "Eigen Components",
                "value": str(num_comp),
                "color": "#8b5cf6",
            },
        ]

        for idx, cfg in enumerate(card_configs):
            self._create_stat_card(cards_frame, cfg, idx)

    def _create_stat_card(self, parent, config, col):
        card = ctk.CTkFrame(
            parent,
            corner_radius=12,
            fg_color=("gray92", "gray14"),
            border_width=1,
            border_color=("gray80", "gray22"),
        )
        card.grid(row=0, column=col, sticky="nsew", padx=8, pady=5)

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=20, pady=18)

        # Icon + Label
        top_row = ctk.CTkFrame(inner, fg_color="transparent")
        top_row.pack(fill="x")
        ctk.CTkLabel(
            top_row, text=f"{config['icon']}  {config['label']}", font=FONT_STATS_LABEL, text_color="gray"
        ).pack(anchor="w")

        # Value
        ctk.CTkLabel(
            inner,
            text=config["value"],
            font=FONT_STATS_VALUE,
            text_color=config["color"],
        ).pack(anchor="w", pady=(8, 0))

    # ==========================================
    # DETAIL ANALISIS (TABEL)
    # ==========================================
    def _build_analysis_detail(self):
        d = self.report_data
        section = ctk.CTkFrame(self.report_content, fg_color="transparent")
        section.pack(fill="x", pady=(15, 10))

        ctk.CTkLabel(section, text="🔬  Detail Analisis", font=FONT_SECTION).pack(
            anchor="w", pady=(0, 12)
        )

        # Bagi menjadi 2 kolom
        detail_frame = ctk.CTkFrame(section, fg_color="transparent")
        detail_frame.pack(fill="x")
        detail_frame.grid_columnconfigure((0, 1), weight=1, uniform="det")

        # === KOLOM KIRI: Parameter & Hasil ===
        left_card = ctk.CTkFrame(
            detail_frame,
            corner_radius=12,
            fg_color=("gray92", "gray14"),
            border_width=1,
            border_color=("gray80", "gray22"),
        )
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=5)

        left_inner = ctk.CTkFrame(left_card, fg_color="transparent")
        left_inner.pack(fill="both", expand=True, padx=20, pady=18)

        ctk.CTkLabel(left_inner, text="Parameter & Hasil", font=FONT_BODY_BOLD, text_color=("#1a1a2e", "#e0e0e0")).pack(anchor="w", pady=(0, 12))

        param_rows = [
            ("Ambang Toleransi", f"{d.get('threshold', 0):.1f}"),
            ("Komponen Eigen", str(d.get("num_components", "N/A"))),
            ("Jarak Euclidean Min.", f"{d.get('min_distance', float('inf')):.4f}" if d.get('min_distance', float('inf')) != float('inf') else "∞"),
            ("Status Pencocokan", d.get("result_status", "N/A")),
            ("Label Dikenali", d.get("result_label", "—") or "—"),
            ("Confidence", f"{d.get('match_percentage', 0):.1f}%" if d.get("match_percentage") is not None else "N/A"),
            ("Waktu Eksekusi", f"{d.get('execution_time', 0):.4f} detik"),
        ]

        for label, value in param_rows:
            self._add_table_row(left_inner, label, value)

        # === KOLOM KANAN: Info Dataset & Model ===
        right_card = ctk.CTkFrame(
            detail_frame,
            corner_radius=12,
            fg_color=("gray92", "gray14"),
            border_width=1,
            border_color=("gray80", "gray22"),
        )
        right_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=5)

        right_inner = ctk.CTkFrame(right_card, fg_color="transparent")
        right_inner.pack(fill="both", expand=True, padx=20, pady=18)

        ctk.CTkLabel(right_inner, text="Dataset & Model", font=FONT_BODY_BOLD, text_color=("#1a1a2e", "#e0e0e0")).pack(anchor="w", pady=(0, 12))

        test_path = d.get("test_image_path", "")
        test_name = os.path.basename(test_path) if test_path else "N/A"
        dataset_path = d.get("dataset_folder", "")
        dataset_name = os.path.basename(dataset_path) if dataset_path else "Tidak dipilih"
        cached = "Ya (dari VRAM)" if d.get("model_loaded_from_cache", False) else "Tidak (kompilasi baru)"

        dataset_rows = [
            ("Gambar Uji", test_name),
            ("Folder Dataset", dataset_name),
            ("Total Gambar Dataset", str(d.get("dataset_total_images", "N/A"))),
            ("Total Kelas/Individu", str(d.get("dataset_total_classes", "N/A"))),
            ("Model dari Cache", cached),
            ("Perangkat Komputasi", d.get("device", "N/A")),
            ("Timestamp", d.get("timestamp", "N/A")),
        ]

        for label, value in dataset_rows:
            self._add_table_row(right_inner, label, value)

    def _add_table_row(self, parent, label, value):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=3)
        row.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            row, text=label, font=FONT_TABLE_CELL, text_color="gray", anchor="w"
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            row, text=value, font=FONT_TABLE_HEADER, anchor="e"
        ).grid(row=0, column=1, sticky="e")

    # ==========================================
    # PERFORMA SISTEM
    # ==========================================
    def _build_system_performance(self):
        d = self.report_data
        snapshots = d.get("system_snapshots", [])

        section = ctk.CTkFrame(self.report_content, fg_color="transparent")
        section.pack(fill="x", pady=(15, 10))

        ctk.CTkLabel(section, text="🖥️  Performa Sistem Selama Analisis", font=FONT_SECTION).pack(
            anchor="w", pady=(0, 12)
        )

        if not snapshots:
            ctk.CTkLabel(
                section,
                text="Data performa sistem tidak tersedia.",
                font=FONT_BODY,
                text_color="gray",
            ).pack(anchor="w")
            return

        # Hitung statistik aggregat
        cpus = [s["cpu"] for s in snapshots]
        rams = [s["ram"] for s in snapshots]
        gpu_loads = [s.get("gpu_load", -1) for s in snapshots]
        gpu_vrams = [s.get("gpu_vram", -1) for s in snapshots]

        has_gpu = any(g >= 0 for g in gpu_loads)

        # Kartu ringkasan performa
        perf_cards_frame = ctk.CTkFrame(section, fg_color="transparent")
        perf_cards_frame.pack(fill="x", pady=(0, 15))

        num_cols = 4 if has_gpu else 2
        for i in range(num_cols):
            perf_cards_frame.grid_columnconfigure(i, weight=1, uniform="perf")

        # CPU Card
        self._create_perf_card(
            perf_cards_frame, 0,
            icon="🔵", title="CPU Usage",
            avg=sum(cpus) / len(cpus),
            peak=max(cpus),
            unit="%",
            color="#3b82f6",
        )

        # RAM Card
        self._create_perf_card(
            perf_cards_frame, 1,
            icon="🟡", title="RAM Usage",
            avg=sum(rams) / len(rams),
            peak=max(rams),
            unit="%",
            color="#eab308",
        )

        if has_gpu:
            valid_gpu_loads = [g for g in gpu_loads if g >= 0]
            valid_gpu_vrams = [g for g in gpu_vrams if g >= 0]

            # GPU Load Card
            if valid_gpu_loads:
                self._create_perf_card(
                    perf_cards_frame, 2,
                    icon="🟢", title="GPU Load",
                    avg=sum(valid_gpu_loads) / len(valid_gpu_loads),
                    peak=max(valid_gpu_loads),
                    unit="%",
                    color="#10b981",
                )

            # GPU VRAM Card
            if valid_gpu_vrams:
                self._create_perf_card(
                    perf_cards_frame, 3,
                    icon="🟣", title="GPU VRAM",
                    avg=sum(valid_gpu_vrams) / len(valid_gpu_vrams),
                    peak=max(valid_gpu_vrams),
                    unit="%",
                    color="#8b5cf6",
                )

        # Tabel snapshot timeline
        self._build_snapshot_table(section, snapshots, has_gpu)

    def _create_perf_card(self, parent, col, icon, title, avg, peak, unit, color):
        card = ctk.CTkFrame(
            parent,
            corner_radius=12,
            fg_color=("gray92", "gray14"),
            border_width=1,
            border_color=("gray80", "gray22"),
        )
        card.grid(row=0, column=col, sticky="nsew", padx=8, pady=5)

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=18, pady=15)

        ctk.CTkLabel(
            inner, text=f"{icon}  {title}", font=FONT_STATS_LABEL, text_color="gray"
        ).pack(anchor="w")

        # Average
        avg_frame = ctk.CTkFrame(inner, fg_color="transparent")
        avg_frame.pack(fill="x", pady=(8, 2))
        ctk.CTkLabel(avg_frame, text="Rata-rata", font=FONT_SMALL, text_color="gray").pack(side="left")
        ctk.CTkLabel(
            avg_frame, text=f"{avg:.1f}{unit}", font=FONT_BODY_BOLD, text_color=color
        ).pack(side="right")

        # Peak
        peak_frame = ctk.CTkFrame(inner, fg_color="transparent")
        peak_frame.pack(fill="x", pady=(2, 0))
        ctk.CTkLabel(peak_frame, text="Puncak", font=FONT_SMALL, text_color="gray").pack(side="left")
        ctk.CTkLabel(
            peak_frame, text=f"{peak:.1f}{unit}", font=FONT_BODY_BOLD, text_color=color
        ).pack(side="right")

    # ==========================================
    # TABEL SNAPSHOT TIMELINE
    # ==========================================
    def _build_snapshot_table(self, parent, snapshots, has_gpu):
        ctk.CTkLabel(
            parent, text="📈  Timeline Snapshot", font=FONT_BODY_BOLD
        ).pack(anchor="w", pady=(10, 8))

        table_card = ctk.CTkFrame(
            parent,
            corner_radius=12,
            fg_color=("gray92", "gray14"),
            border_width=1,
            border_color=("gray80", "gray22"),
        )
        table_card.pack(fill="x", pady=(0, 10))

        table_inner = ctk.CTkFrame(table_card, fg_color="transparent")
        table_inner.pack(fill="x", padx=18, pady=15)

        # Header
        headers = ["#", "Waktu (s)", "CPU %", "RAM %"]
        if has_gpu:
            headers.extend(["GPU Load %", "GPU VRAM %"])

        col_count = len(headers)
        for i in range(col_count):
            table_inner.grid_columnconfigure(i, weight=1, uniform="tbl")

        for col_idx, h in enumerate(headers):
            lbl = ctk.CTkLabel(
                table_inner, text=h, font=FONT_TABLE_HEADER, text_color=("#1a1a2e", "#a0a0a0")
            )
            lbl.grid(row=0, column=col_idx, sticky="ew", padx=5, pady=(0, 8))

        # Divider header
        div = ctk.CTkFrame(table_inner, height=1, fg_color=("gray75", "gray30"))
        div.grid(row=1, column=0, columnspan=col_count, sticky="ew", pady=(0, 5))

        # Data rows (max 30 untuk UI performance)
        display_snapshots = snapshots
        if len(snapshots) > 30:
            # Ambil sample merata
            step = len(snapshots) / 30
            display_snapshots = [snapshots[int(i * step)] for i in range(30)]

        base_time = snapshots[0]["time"] if snapshots else 0

        for row_idx, snap in enumerate(display_snapshots):
            actual_row = row_idx + 2  # offset karena header + divider

            row_bg = "transparent"

            values = [
                str(row_idx + 1),
                f"{snap['time'] - base_time:.2f}",
                f"{snap['cpu']:.1f}",
                f"{snap['ram']:.1f}",
            ]
            if has_gpu:
                gpu_l = snap.get("gpu_load", -1)
                gpu_v = snap.get("gpu_vram", -1)
                values.append(f"{gpu_l:.1f}" if gpu_l >= 0 else "N/A")
                values.append(f"{gpu_v:.1f}" if gpu_v >= 0 else "N/A")

            colors = [
                ("gray", "gray"),  # #
                ("#555555", "#aaaaaa"),  # waktu
                ("#3b82f6", "#60a5fa"),  # cpu
                ("#eab308", "#fbbf24"),  # ram
            ]
            if has_gpu:
                colors.append(("#10b981", "#34d399"))  # gpu load
                colors.append(("#8b5cf6", "#a78bfa"))  # gpu vram

            for col_idx, val in enumerate(values):
                light_c, dark_c = colors[col_idx] if col_idx < len(colors) else ("gray", "gray")
                lbl = ctk.CTkLabel(
                    table_inner,
                    text=val,
                    font=FONT_TABLE_CELL,
                    text_color=(light_c, dark_c),
                )
                lbl.grid(row=actual_row, column=col_idx, sticky="ew", padx=5, pady=2)

        # Info total snapshot
        if len(snapshots) > 30:
            ctk.CTkLabel(
                table_inner,
                text=f"Menampilkan 30 dari {len(snapshots)} snapshot (disampling merata)",
                font=FONT_SMALL,
                text_color="gray",
            ).grid(row=len(display_snapshots) + 2, column=0, columnspan=col_count, pady=(8, 0), sticky="w")
