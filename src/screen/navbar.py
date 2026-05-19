# src/screen/navbar.py
import customtkinter as ctk

FONT_FAMILY = "Segoe UI"
FONT_HEADER_BY = (FONT_FAMILY, 11, "normal")
FONT_HEADER_GRP = (FONT_FAMILY, 15, "bold")

class TopNavbar(ctk.CTkFrame):
    def __init__(self, parent, switch_callback, logo_img, icon_scan, icon_member, icon_report=None):
        super().__init__(parent, height=55, corner_radius=0, fg_color=("gray85", "#1e1e1e"))
        self.switch_callback = switch_callback
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0) 
        self.grid_columnconfigure(1, weight=1) 
        
        # ================= BAGIAN KIRI (LOGO & NAMA KELOMPOK) =================
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, padx=(25, 30), sticky="w")
        header_frame.grid_rowconfigure(0, weight=1)
        
        # Wrapper 1: Khusus untuk Logo
        logo_wrapper = ctk.CTkFrame(header_frame, fg_color="transparent")
        logo_wrapper.grid(row=0, column=0, padx=(0, 15), sticky="w")
        
        if logo_img:
            lbl_logo = ctk.CTkLabel(logo_wrapper, text="", image=logo_img)
        else:
            lbl_logo = ctk.CTkLabel(logo_wrapper, text="UNS", font=(FONT_FAMILY, 20, "bold"), text_color="#3b82f6")
        lbl_logo.pack(pady=5)
        
        # Wrapper 2: Khusus untuk Teks (Padding 0px agar saling menempel ketat)
        text_wrapper = ctk.CTkFrame(header_frame, fg_color="transparent")
        text_wrapper.grid(row=0, column=1, sticky="w")
        
        ctk.CTkLabel(text_wrapper, text="Kelompok 7", font=FONT_HEADER_GRP).pack(anchor="w", pady=0)

        # ================= BAGIAN TENGAH (TABS ALA ZOOM) =================
        self.active_bg = ("gray75", "#333333") 
        self.inactive_bg = "transparent"
        self.active_text = ("black", "white")
        self.inactive_text = ("gray40", "gray60")
        
        tab_container = ctk.CTkFrame(self, fg_color="transparent")
        tab_container.grid(row=0, column=1, sticky="w") 
        
        # Tombol Dashboard (Menggunakan icon_scan)
        self.btn_dashboard = ctk.CTkButton(
            tab_container, text=" Analysis Dashboard", font=(FONT_FAMILY, 13, "bold"),
            image=icon_scan, compound="left",
            fg_color=self.active_bg, text_color=self.active_text, hover_color=("gray70", "#2d2d2d"),
            corner_radius=8, height=35, command=lambda: self.on_tab_click("dashboard")
        )
        self.btn_dashboard.grid(row=0, column=0, padx=(0, 5))

        # Tombol Report (Menggunakan icon_report)
        self.btn_report = ctk.CTkButton(
            tab_container, text=" Analysis Report", font=(FONT_FAMILY, 13, "bold"),
            image=icon_report, compound="left",
            fg_color=self.inactive_bg, text_color=self.inactive_text, hover_color=("gray70", "#2d2d2d"),
            corner_radius=8, height=35, command=lambda: self.on_tab_click("report")
        )
        self.btn_report.grid(row=0, column=1, padx=5)

        # Tombol Kelompok (Menggunakan icon_member)
        self.btn_group = ctk.CTkButton(
            tab_container, text=" Anggota Kelompok", font=(FONT_FAMILY, 13, "bold"),
            image=icon_member, compound="left",
            fg_color=self.inactive_bg, text_color=self.inactive_text, hover_color=("gray70", "#2d2d2d"),
            corner_radius=8, height=35, command=lambda: self.on_tab_click("group")
        )
        self.btn_group.grid(row=0, column=2, padx=5)

    def on_tab_click(self, page_name):
        # Reset semua tab
        all_btns = [self.btn_dashboard, self.btn_report, self.btn_group]
        for btn in all_btns:
            btn.configure(fg_color=self.inactive_bg, text_color=self.inactive_text)

        # Aktifkan tab yang dipilih
        if page_name == "dashboard":
            self.btn_dashboard.configure(fg_color=self.active_bg, text_color=self.active_text)
        elif page_name == "report":
            self.btn_report.configure(fg_color=self.active_bg, text_color=self.active_text)
        elif page_name == "group":
            self.btn_group.configure(fg_color=self.active_bg, text_color=self.active_text)
        
        self.switch_callback(page_name)