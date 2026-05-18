# src/screen/splash_screen.py
import customtkinter as ctk

class SplashScreen(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("Loading...")
        
        width, height = 400, 250
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        x = (screen_w / 2) - (width / 2)
        y = (screen_h / 2) - (height / 2)
        self.geometry(f"{width}x{height}+{int(x)}+{int(y)}")
        
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        
        ctk.CTkLabel(self, text="Face Recognition Application", font=ctk.CTkFont(size=22, weight="bold"), text_color="#3b82f6").pack(pady=(50, 5))
        ctk.CTkLabel(self, text="Kelompok 7 - Informatika 2025D", font=ctk.CTkFont(size=14)).pack(pady=(0, 30))
        
        self.progress = ctk.CTkProgressBar(self, width=300, mode="determinate")
        self.progress.pack(pady=10)
        self.progress.set(0)
        
        self.lbl_status = ctk.CTkLabel(self, text="Initializing...", font=ctk.CTkFont(size=12), text_color="gray")
        self.lbl_status.pack()

        self.update_progress()

    def update_progress(self, step=0):
        if step <= 100:
            self.progress.set(step / 100)
            
            if step == 20:
                self.lbl_status.configure(text="Loading UI Elements...")
            elif step == 50:
                self.lbl_status.configure(text="Importing Eigenface Modules...")
            elif step == 80:
                self.lbl_status.configure(text="Almost Ready...")
            elif step == 100:
                self.lbl_status.configure(text="Done!")
            
            self.after(25, self.update_progress, step + 1)
        else:
            self.destroy()
            self.parent.deiconify()