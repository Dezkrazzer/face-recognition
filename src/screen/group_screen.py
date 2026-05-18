# src/screen/group_screen.py
import os
import customtkinter as ctk
from PIL import Image

# ==========================================
# KONFIGURASI TIPOGRAFI
# ==========================================
FONT_FAMILY = "Segoe UI"
FONT_TITLE = (FONT_FAMILY, 34, "bold")
FONT_SUBTITLE = (FONT_FAMILY, 14, "normal")
FONT_NAME = (FONT_FAMILY, 18, "bold")
FONT_ROLE = (FONT_FAMILY, 14, "bold")
FONT_NIM = (FONT_FAMILY, 13, "normal")

class GroupScreen(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color="transparent")
        
        self.assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
        
        # ==========================================
        # DATABASE ANGGOTA KELOMPOK
        # ==========================================
        self.member_data = [
            {
                "name": "Lazuardi Akbar Imani",
                "kelas": "Informatika D",
                "nim": "L0125105",
                "color": "#10b981", # Hijau Emerald
                "emoji": "💻",
                "photo": "foto_member_1"
            },
            {
                "name": "Muhammad Haidar Ramzy",
                "kelas": "Informatika D",
                "nim": "L0125109",
                "color": "#eab308", # Kuning
                "emoji": "📊",
                "photo": "foto_member_2"
            },
            {
                "name": "Egifrid Angelo Mwoleka",
                "kelas": "Informatika D",
                "nim": "L0125XXX",
                "color": "#3b82f6", # Biru
                "emoji": "📝",
                "photo": "foto_member_3"
            }
        ]
        self.setup_ui()

    def load_member_photo(self, photo_name):
        for ext in [".png", ".jpg", ".jpeg"]:
            photo_path = os.path.join(self.assets_dir, photo_name + ext)
            if os.path.exists(photo_path):
                try:
                    img = Image.open(photo_path).convert("RGBA")
                    
                    # Crop gambar agar menjadi persegi presisi di tengah
                    w, h = img.size
                    side = min(w, h)
                    left = (w - side) // 2
                    top = (h - side) // 2
                    img = img.crop((left, top, left + side, top + side))
                    
                    # Resize ke 150x150
                    img = img.resize((150, 150), Image.Resampling.LANCZOS)
                    
                    return ctk.CTkImage(light_image=img, dark_image=img, size=(150, 150))
                except Exception:
                    return None
        return None

    def setup_ui(self):
        # -- BAGIAN HEADER --
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(pady=(40, 10), padx=30)
        
        ctk.CTkLabel(header_frame, text="Anggota Kelompok", font=FONT_TITLE, text_color="#3b82f6").pack(pady=(0, 5))
        ctk.CTkLabel(header_frame, text="Informatika - Faculty of Information Technology and Data Science", font=FONT_SUBTITLE, text_color="gray").pack()

        # -- PEMBATAS --
        ctk.CTkFrame(self, height=1, fg_color=("gray80", "gray20")).pack(fill="x", padx=50, pady=20)

        # -- CONTAINER KARTU --
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(expand=True, padx=40, pady=(0, 20))
        container.grid_columnconfigure((0, 1, 2), weight=1, uniform="col")
        container.grid_rowconfigure(0, weight=1)

        for idx, member in enumerate(self.member_data):
            self.create_member_card(container, member, idx)

    def create_member_card(self, parent, member, col):
        """Merender satu kartu profil bergaya Flat/Minimalist"""
        
        # Kartu Utama dengan warna Solid
        card = ctk.CTkFrame(
            parent, 
            corner_radius=15, 
            fg_color=("gray92", "gray14"), # Warna dasar kartu
            border_width=1, 
            border_color=("gray80", "gray20")
        )
        card.grid(row=0, column=col, sticky="nsew", padx=15, pady=10)

        # Container dalam Kartu untuk mengatur Padding secara merata
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=25, pady=30)

        # 1. AVATAR SECTION
        photo = self.load_member_photo(member["photo"])
        
        if photo:
            avatar_lbl = ctk.CTkLabel(
                inner, 
                text="", 
                image=photo, 
                width=150, 
                height=150, 
                corner_radius=20, # Membuat foto menjadi Rounded Square
                fg_color=("#e2e8f0", "#1e293b") # Warna backdrop jika foto transparan
            )
            avatar_lbl.photo_ref = photo
        else:
            # Fallback Emoji
            avatar_lbl = ctk.CTkLabel(
                inner, 
                text=member["emoji"], 
                font=(FONT_FAMILY, 60), 
                width=150, 
                height=150, 
                corner_radius=20, 
                fg_color=("#e2e8f0", "#1e293b")
            )
            
        avatar_lbl.pack(pady=(0, 25))

        # 2. TEXT SECTION
        ctk.CTkLabel(
            inner, 
            text=member["name"], 
            font=FONT_NAME, 
            text_color=("black", "white")
        ).pack(pady=(0, 5))

        ctk.CTkLabel(
            inner, 
            text=member["kelas"], 
            font=FONT_NIM, 
            text_color="gray"
        ).pack(pady=(0, 20))

        # Aksen Garis Kecil (Warna-warni sesuai nim)
        ctk.CTkFrame(
            inner, 
            height=3, 
            width=40,
            corner_radius=2,
            fg_color=member["color"]
        ).pack(pady=(0, 15))

        ctk.CTkLabel(
            inner, 
            text=member["nim"], 
            font=FONT_ROLE, 
            text_color=member["color"]
        ).pack()