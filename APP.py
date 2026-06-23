import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import threading
from ultralytics import YOLO
import os
import sys  # Added for PyInstaller path resolution
import multiprocessing  # Added to prevent infinite window spawning in .exe
from datetime import datetime
import time
import queue
import torch  # Required to run host-level device inspection for standalone deployment
from fpdf import FPDF  # Moved to global scope so PyInstaller bundles it automatically


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # Crucial Fix: Anchors path to the script's actual directory, preventing launch-from-anywhere crashes
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


class ModernButton(tk.Canvas):
    def __init__(self, parent, text, command, bg_color="#3b82f6", hover_color="#2563eb", 
                 fg_color="white", width=280, height=42, radius=8, font_size=10, **kwargs):
        super().__init__(parent, width=width, height=height, bg=parent["bg"], 
                        highlightthickness=0, **kwargs)
        
        self.command = command
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.fg_color = fg_color
        self.current_color = bg_color
        self.text = text
        self.radius = radius
        self.font_size = font_size
        self._width = width
        self._height = height
        self.enabled = True
        
        self.draw_button()
        
        self.bind("<Enter>", self.on_hover)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)
        
    def draw_button(self):
        self.delete("all")
        self.create_rounded_rect(2, 2, self._width-2, self._height-2, 
                                self.radius, fill=self.current_color, outline="")
        self.create_text(self._width//2, self._height//2, text=self.text, 
                        fill=self.fg_color, font=("Segoe UI", self.font_size, "bold"), justify=tk.CENTER)
        
    def create_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1+radius, y1, x2-radius, y1, x2, y1, x2, y1+radius,
            x2, y2-radius, x2, y2, x2-radius, y2, x1+radius, y2,
            x1, y2, x1, y2-radius, x1, y1+radius, x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kwargs)
        
    def on_hover(self, event):
        if self.enabled:
            self.current_color = self.hover_color
            self.draw_button()
        
    def on_leave(self, event):
        if self.enabled:
            self.current_color = self.bg_color
            self.draw_button()
        
    def on_click(self, event):
        if self.enabled and self.command:
            self.command()
            
    def set_state(self, enabled):
        self.enabled = enabled
        if not enabled:
            self.current_color = "#374151"
        else:
            self.current_color = self.bg_color
        self.draw_button()


class HelmetDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Intelligence System")
        self.root.geometry("1600x900")  
        self.root.minsize(1400, 800)
        
        self.colors = {
            'bg_dark': '#0b0f19',       
            'bg_card': '#1e293b',       
            'border': '#334155',        
            'accent_blue': '#3b82f6',   
            'accent_green': '#10b981',  
            'accent_red': '#ef4444',    
            'accent_orange': '#f59e0b', 
            'text_primary': '#ffffff',  
            'text_secondary': '#9ca3af' 
        }
        
        self.root.configure(bg=self.colors['bg_dark'])
        
        self.log_dir = "violations"
        os.makedirs(self.log_dir, exist_ok=True)
        self.violation_records = []
        self.saved_violation_ids = set()
        self.gallery_cards = []  
        
        self.frame_queue = queue.Queue(maxsize=3)
        
        self.model = None
        self.base_model = None
        self.device = "cpu"  # Default device fallthrough container
        
        self.is_webcam_running = False
        self.is_video_running = False
        self.cap = None
        self.video_thread = None
        
        self.total_violations = 0
        self.total_safe = 0
        self.frame_count = 0
        self.process_skip = 2
        self.fps = 0
        
        self.process_width = 416
        self.process_height = 416
        self.display_width = 854
        self.display_height = 480
        
        self.reset_tracks()
        self.setup_gui()
        
        # Load core models directly AFTER GUI creation to update systemic text elements
        self.load_models()
        
        self.update_display_loop()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def load_models(self):
        self.append_to_log("SYSTEM: Probing system processing layout configuration...")
        
        # Step 1: Detect underlying processing core architecture 
        if torch.cuda.is_available():
            self.device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            dialog_title = " Hardware Acceleration Active"
            dialog_msg = f"NVIDIA CUDA GPU Detected!\n\nRunning Inferences on: {gpu_name}\n\nThe software will utilize hardware-accelerated deep matrix operations."
            badge_text = f" CUDA GPU ACTIVE: {gpu_name.split()[0]}"
            badge_bg = self.colors['accent_green']
        else:
            self.device = "cpu"
            dialog_title = "🖥️ CPU Mode Active"
            dialog_msg = "No compatible NVIDIA CUDA GPU detected on this host system.\n\nThe application will run using safe multi-threaded System CPU fallback execution architecture."
            badge_text = "🖥️ ENGINE: SYSTEM CPU FALLBACK"
            badge_bg = self.colors['accent_orange']
            
        # Step 2: Push systemic prompt alert notification out to user 
        messagebox.showinfo(dialog_title, dialog_msg)
        
        # Step 3: Mutate live tracking status badges to match host system specs
        self.lbl_engine_badge.config(text=badge_text, bg=badge_bg)
        self.lbl_engine_bg_frame.config(bg=badge_bg)
        
        # Step 4: Map runtime compilation parameters safely onto targeted system device
        try:
            # Use resource_path so PyInstaller .exe finds the bundled weight file
            best_model_path = resource_path('best.pt')
            self.model = YOLO(best_model_path)
            self.model.to(self.device)
            
            # Crucial Fix: Wrap base model in resource_path to guarantee offline bundling works in .exe
            base_model_path = resource_path('yolov8n.pt')
            self.base_model = YOLO(base_model_path)
            self.base_model.to(self.device)
            
            self.append_to_log(f"SYSTEM: Parallel inference core linked directly to environment: [{self.device.upper()}]")
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed loading parallel inference core weights: {str(e)}")
            
    def setup_gui(self):
        self.root.grid_columnconfigure(0, weight=0)  
        self.root.grid_columnconfigure(1, weight=1)  
        self.root.grid_columnconfigure(2, weight=0)  
        self.root.grid_rowconfigure(1, weight=1)
        
        # ==========================================
        # UPGRADED LIGHTNING-THEMED HEADER PANEL
        # ==========================================
        header_panel = tk.Frame(self.root, bg='#06080e', height=75, highlightbackground=self.colors['border'], highlightthickness=1)
        header_panel.grid(row=0, column=0, columnspan=3, sticky="ew")
        header_panel.pack_propagate(False)
        
        # High tech glowing electric bottom accent strip
        lightning_strip = tk.Frame(header_panel, bg='#00f2fe', height=2)
        lightning_strip.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Depth shadow layer for title texture
        lbl_shadow_title = tk.Label(header_panel, text=" T R A F F I C   I N T E L L I G E N C E   S Y S T E M ", 
                                    font=('Segoe UI', 19, 'bold'), bg='#06080e', fg='#2563eb')
        lbl_shadow_title.place(relx=0.502, rely=0.48, anchor=tk.CENTER)
        
        # Main foreground dynamic electric cyan title text
        lbl_main_title = tk.Label(header_panel, text=" T R A F F I C   I N T E L L I G E N C E   S Y S T E M ", 
                                  font=('Segoe UI', 19, 'bold'), bg='#06080e', fg='#00f2fe')
        lbl_main_title.place(relx=0.5, rely=0.45, anchor=tk.CENTER)
        
        # Live dynamic Hardware processing hardware configuration status badge
        self.lbl_engine_bg_frame = tk.Frame(header_panel, bg=self.colors['accent_blue'], padx=12, pady=5)
        self.lbl_engine_bg_frame.pack(side=tk.RIGHT, padx=25, pady=16)
        
        self.lbl_engine_badge = tk.Label(self.lbl_engine_bg_frame, text="⚙️ INITIALIZING COMPUTER LOGIC...", 
                                         font=('Segoe UI', 8, 'bold'), bg=self.colors['accent_blue'], fg="white")
        self.lbl_engine_badge.pack()

        self.setup_left_panel()
        self.setup_center_panel()
        self.setup_right_panel()
        
        footer_panel = tk.Frame(self.root, bg='#0f0f12', height=35, highlightbackground=self.colors['border'], highlightthickness=1)
        footer_panel.grid(row=2, column=0, columnspan=3, sticky="ew")
        footer_panel.pack_propagate(False)
        
        lbl_footer_left = tk.Label(footer_panel, text="🔒 Trajectory Smoothing Filters & Vehicle Verification Active", font=('Segoe UI', 9), bg='#0f0f12', fg=self.colors['text_secondary'])
        lbl_footer_left.pack(side=tk.LEFT, padx=25, pady=6)
        
        lbl_footer_right = tk.Label(footer_panel, text="Developed by Code_IT", font=('Segoe UI', 9, 'bold'), bg='#0f0f12', fg=self.colors['accent_blue'])
        lbl_footer_right.pack(side=tk.RIGHT, padx=25, pady=6)

    def setup_left_panel(self):
        left_panel = tk.Frame(self.root, bg=self.colors['bg_card'], width=320)
        left_panel.grid(row=1, column=0, sticky="nsew", padx=(20, 10), pady=20)
        left_panel.grid_propagate(False)
        
        title_frame = tk.Frame(left_panel, bg=self.colors['bg_card'])
        title_frame.pack(fill=tk.X, padx=20, pady=(20, 5))
        tk.Label(title_frame, text="HELMET DETECTION", font=('Segoe UI', 14, 'bold'), bg=self.colors['bg_card'], fg=self.colors['text_primary']).pack(anchor='w')
        tk.Label(title_frame, text="Control & Metric Interface", font=('Segoe UI', 9), bg=self.colors['bg_card'], fg=self.colors['text_secondary']).pack(anchor='w')
        
        tk.Frame(left_panel, bg=self.colors['border'], height=1).pack(fill=tk.X, padx=20, pady=15)
        
        tk.Label(left_panel, text="DETECTION MODES", font=('Segoe UI', 9, 'bold'), bg=self.colors['bg_card'], fg=self.colors['text_secondary']).pack(anchor='w', padx=20, pady=(0, 6))
        btn_container = tk.Frame(left_panel, bg=self.colors['bg_card'])
        btn_container.pack(fill=tk.X, padx=20)
        
        self.btn_image = ModernButton(btn_container, text="📷  Detect Image Source", command=self.detect_image, bg_color=self.colors['accent_blue'], hover_color="#2563eb", width=280)
        self.btn_image.pack(pady=5)
        
        self.btn_video = ModernButton(btn_container, text="🎬  Process Video File", command=self.detect_video, bg_color=self.colors['accent_blue'], hover_color="#2563eb", width=280)
        self.btn_video.pack(pady=5)
        
        webcam_frame = tk.Frame(btn_container, bg=self.colors['bg_card'])
        webcam_frame.pack(fill=tk.X, pady=5)
        self.btn_webcam_start = ModernButton(webcam_frame, text="▶  Webcam Live", command=self.start_webcam, bg_color=self.colors['accent_green'], hover_color="#059669", width=135)
        self.btn_webcam_start.pack(side=tk.LEFT, padx=(0, 10))
        self.btn_webcam_stop = ModernButton(webcam_frame, text="⏹  Stop Stream", command=self.stop_all, bg_color=self.colors['accent_red'], hover_color="#dc2626", width=135)
        self.btn_webcam_stop.pack(side=tk.LEFT)
        self.btn_webcam_stop.set_state(False)
        
        tk.Frame(left_panel, bg=self.colors['border'], height=1).pack(fill=tk.X, padx=20, pady=15)
        
        tk.Label(left_panel, text="KPI ENGINE METRICS", font=('Segoe UI', 9, 'bold'), bg=self.colors['bg_card'], fg=self.colors['text_secondary']).pack(anchor='w', padx=20, pady=(0, 8))
        stats_container = tk.Frame(left_panel, bg=self.colors['bg_card'])
        stats_container.pack(fill=tk.X, padx=16)
        
        self.setup_stat_card(stats_container, "FPS", "fps", self.colors['accent_green'], 0, 0)
        self.setup_stat_card(stats_container, "Frames Tracked", "frames", self.colors['accent_blue'], 0, 1)
        self.setup_stat_card(stats_container, "Violations", "violations", self.colors['accent_red'], 1, 0)
        self.setup_stat_card(stats_container, "Safe Enforced", "safe", self.colors['accent_green'], 1, 1)
        
        tk.Frame(left_panel, bg=self.colors['border'], height=1).pack(fill=tk.X, padx=20, pady=15)
        
        settings_container = tk.Frame(left_panel, bg=self.colors['bg_card'])
        settings_container.pack(fill=tk.X, padx=20)
        conf_header = tk.Frame(settings_container, bg=self.colors['bg_card'])
        conf_header.pack(fill=tk.X)
        tk.Label(conf_header, text="Confidence Threshold", font=('Segoe UI', 9), bg=self.colors['bg_card'], fg=self.colors['text_primary']).pack(side=tk.LEFT)
        self.lbl_conf_value = tk.Label(conf_header, text="50%", font=('Segoe UI', 9, 'bold'), bg=self.colors['bg_card'], fg=self.colors['accent_blue'])
        self.lbl_conf_value.pack(side=tk.RIGHT)
        
        self.confidence_var = tk.DoubleVar(value=0.5)
        
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Custom.Horizontal.TScale", 
                        background=self.colors['bg_card'], 
                        troughcolor=self.colors['bg_dark'],
                        bordercolor=self.colors['border'])
        
        self.confidence_scale = ttk.Scale(settings_container, from_=0.1, to=0.9, variable=self.confidence_var, orient=tk.HORIZONTAL, style="Custom.Horizontal.TScale", command=self.update_confidence_label)
        self.confidence_scale.pack(fill=tk.X, pady=(6, 0))

    def setup_stat_card(self, parent, label, key, color, row, col):
        frame = tk.Frame(parent, bg=self.colors['bg_dark'], padx=10, pady=10, highlightbackground=self.colors['border'], highlightthickness=1)
        frame.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
        parent.grid_columnconfigure(col, weight=1)
        val_lbl = tk.Label(frame, text="0", font=('Segoe UI', 16, 'bold'), bg=self.colors['bg_dark'], fg=color)
        val_lbl.pack()
        tk.Label(frame, text=label, font=('Segoe UI', 8), bg=self.colors['bg_dark'], fg=self.colors['text_secondary']).pack()
        setattr(self, f'stat_{key}', val_lbl)

    def setup_center_panel(self):
        center_panel = tk.Frame(self.root, bg=self.colors['bg_dark'])
        center_panel.grid(row=1, column=1, sticky="nsew", padx=10, pady=20)
        
        video_outer = tk.Frame(center_panel, bg=self.colors['border'], padx=1, pady=1)
        video_outer.pack(fill=tk.BOTH, expand=True)
        video_container = tk.Frame(video_outer, bg=self.colors['bg_card'])
        video_container.pack(fill=tk.BOTH, expand=True)
        
        header_frame = tk.Frame(video_container, bg=self.colors['bg_card'])
        header_frame.pack(fill=tk.X, padx=20, pady=12)
        
        live_frame = tk.Frame(header_frame, bg=self.colors['bg_card'])
        live_frame.pack(side=tk.LEFT)
        self.live_dot = tk.Label(live_frame, text="●", font=('Segoe UI', 11), bg=self.colors['bg_card'], fg=self.colors['text_secondary'])
        self.live_dot.pack(side=tk.LEFT)
        self.live_label = tk.Label(live_frame, text=" READY", font=('Segoe UI', 10, 'bold'), bg=self.colors['bg_card'], fg=self.colors['text_secondary'])
        self.live_label.pack(side=tk.LEFT)
        
        self.status_label = tk.Label(header_frame, text="Dual-inference active processing systems offline", font=('Segoe UI', 9), bg=self.colors['bg_card'], fg=self.colors['text_secondary'])
        self.status_label.pack(side=tk.RIGHT)
        
        self.canvas = tk.Canvas(video_container, bg='#121212', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        self.canvas.bind("<Configure>", lambda event: self.draw_placeholder())

    def setup_right_panel(self):
        right_panel = tk.Frame(self.root, bg=self.colors['bg_card'], width=320)
        right_panel.grid(row=1, column=2, sticky="nsew", padx=(10, 20), pady=20)
        right_panel.grid_propagate(False)
        
        lbl_gallery_title = tk.Label(right_panel, text="⚠️ VIOLATION LIVE MATRIX", font=('Segoe UI', 11, 'bold'), bg=self.colors['bg_card'], fg=self.colors['accent_red'])
        lbl_gallery_title.pack(anchor='w', padx=18, pady=(18, 2))
        lbl_gallery_sub = tk.Label(right_panel, text="Bystanders & non-riders safely ignored", font=('Segoe UI', 8), bg=self.colors['bg_card'], fg=self.colors['text_secondary'])
        lbl_gallery_sub.pack(anchor='w', padx=18, pady=(0, 10))
        
        gallery_outer = tk.Frame(right_panel, bg=self.colors['bg_dark'], highlightbackground=self.colors['border'], highlightthickness=1)
        gallery_outer.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)
        
        self.gallery_canvas = tk.Canvas(gallery_outer, bg=self.colors['bg_dark'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(gallery_outer, orient="vertical", command=self.gallery_canvas.yview)
        
        self.gallery_frame = tk.Frame(self.gallery_canvas, bg=self.colors['bg_dark'])
        self.gallery_canvas.create_window((0, 0), window=self.gallery_frame, anchor="nw")
        
        self.gallery_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.gallery_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.gallery_frame.bind("<Configure>", lambda e: self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox("all")))
        
        tk.Label(right_panel, text="SYSTEM LOGGER CONSOLE", font=('Segoe UI', 9, 'bold'), bg=self.colors['bg_card'], fg=self.colors['text_secondary']).pack(anchor='w', padx=18, pady=(12, 4))
        self.log_box = tk.Text(right_panel, height=6, bg=self.colors['bg_dark'], fg=self.colors['accent_green'], font=('Consolas', 8), highlightbackground=self.colors['border'], highlightthickness=1, state=tk.DISABLED)
        self.log_box.pack(fill=tk.X, padx=15, pady=4)
        
        self.btn_reset = ModernButton(right_panel, text="🔄  Reset Evaluation Log", command=self.reset_stats, bg_color=self.colors['accent_orange'], hover_color="#d97706", width=280, height=35, font_size=9)
        self.btn_reset.pack(pady=(12, 4))
        
        self.btn_report = ModernButton(right_panel, text="📊  Export HTML Report", command=self.generate_report, bg_color=self.colors['accent_blue'], hover_color="#2563eb", width=280, height=35, font_size=9)
        self.btn_report.pack(pady=4)

        self.btn_pdf_report = ModernButton(right_panel, text="📄  Export PDF Report", command=self.generate_pdf_report, bg_color=self.colors['accent_green'], hover_color="#059669", width=280, height=35, font_size=9)
        self.btn_pdf_report.pack(pady=(0, 15))

    def find_associated_vehicle(self, helmet_box, vehicles):
        hx1, hy1, hx2, hy2 = helmet_box
        hcx = (hx1 + hx2) / 2
        
        for v in vehicles:
            vx1, vy1, vx2, vy2 = v['bbox']
            x_overlap = max(0, min(hx2, vx2) - max(hx1, vx1))
            y_overlap = max(0, min(hy2, vy2) - max(hy1, vy1))
            if x_overlap > 0 and y_overlap > 0:
                return v
                
            pad_x = (vx2 - vx1) * 0.3
            if (vx1 - pad_x) <= hcx <= (vx2 + pad_x):
                if abs(hy2 - vy1) < (vy2 - vy1) * 0.4:
                    return v
        return None

    def trigger_screenshot_capture(self, frame, bbox, track_id):
        try:
            x1, y1, x2, y2 = map(int, bbox)
            f_h, f_w = frame.shape[:2]
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(f_w - 1, x2), min(f_h - 1, y2)
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: return
            
            ts = datetime.now().strftime("%H%M%S_%f")[:-3]
            filename = f"crop_id_{track_id}_{ts}.jpg"
            filepath = os.path.join(self.log_dir, filename)
            cv2.imwrite(filepath, crop)
            
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.violation_records.append({
                'track_id': track_id,
                'timestamp': timestamp_str,
                'img_path': os.path.abspath(filepath)
            })
            
            self.root.after(0, self.add_violation_thumbnail, filepath, track_id, timestamp_str)
        except Exception as e:
            print(f"Failed handling crop context extraction: {e}")

    def add_violation_thumbnail(self, filepath, track_id, timestamp_str):
        try:
            if len(self.gallery_cards) >= 20:
                oldest_card = self.gallery_cards.pop(0)
                oldest_card.destroy()
                
            pil_img = Image.open(filepath)
            pil_img.thumbnail((120, 120), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(image=pil_img)
            
            card = tk.Frame(self.gallery_frame, bg=self.colors['bg_card'], padx=6, pady=6, highlightbackground=self.colors['border'], highlightthickness=1)
            card.pack(fill=tk.X, padx=10, pady=5)
            
            card.bind("<Enter>", lambda e, c=card: c.config(highlightbackground=self.colors['accent_red']))
            card.bind("<Leave>", lambda e, c=card: c.config(highlightbackground=self.colors['border']))
            
            img_lbl = tk.Label(card, image=tk_img, bg=self.colors['bg_dark'])
            img_lbl.image = tk_img  
            img_lbl.pack(side=tk.LEFT, padx=(0, 10))
            
            meta_frame = tk.Frame(card, bg=self.colors['bg_card'])
            meta_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            tk.Label(meta_frame, text=f"RIDER ID: #{track_id}", font=('Segoe UI', 9, 'bold'), bg=self.colors['bg_card'], fg=self.colors['accent_red']).pack(anchor='w', pady=(2, 0))
            tk.Label(meta_frame, text=f"⏱ {timestamp_str.split()[-1]}", font=('Consolas', 8), bg=self.colors['bg_card'], fg=self.colors['text_secondary']).pack(anchor='w', pady=(2, 0))
            
            self.gallery_cards.append(card)
            self.root.update_idletasks()
            self.gallery_canvas.yview_moveto(1.0)
            
            self.append_to_log(f"ALERT: Rider ID {track_id} helmet safety breach logged.")
        except Exception as e:
            print(f"Error drawing metadata card: {e}")

    def append_to_log(self, text):
        self.log_box.config(state=tk.NORMAL)
        t_str = datetime.now().strftime("%H:%M:%S")
        self.log_box.insert(tk.END, f"[{t_str}] {text}\n")
        self.log_box.see(tk.END)
        self.log_box.config(state=tk.DISABLED)

    def update_confidence_label(self, value):
        self.lbl_conf_value.config(text=f"{int(float(value)*100)}%")

    def set_live_status(self, active, text="READY"):
        if active:
            color = self.colors['accent_green'] if text == "LIVE" else self.colors['accent_blue']
            self.live_dot.config(fg=color)
            self.live_label.config(text=f" {text}", fg=color)
        else:
            self.live_dot.config(fg=self.colors['text_secondary'])
            self.live_label.config(text=f" {text}", fg=self.colors['text_secondary'])

    def reset_stats(self):
        self.total_violations = 0
        self.total_safe = 0
        self.frame_count = 0
        self.fps = 0
        self.reset_tracks()
        self.update_stats()
        self.violation_records.clear()
        self.saved_violation_ids.clear()
        
        for card in self.gallery_cards:
            card.destroy()
        self.gallery_cards.clear()
        
        self.log_box.config(state=tk.NORMAL)
        self.log_box.delete("1.0", tk.END)
        self.log_box.config(state=tk.DISABLED)
        self.status_label.config(text="Statistics dashboard cleared.")
        self.append_to_log("System logs re-initialized.")
        
    def update_stats(self):
        self.stat_fps.config(text=str(self.fps))
        self.stat_frames.config(text=str(self.frame_count))
        self.stat_violations.config(text=str(self.total_violations))
        self.stat_safe.config(text=str(self.total_safe))

    def reset_tracks(self):
        self.active_tracks = []
        self.next_track_id = 1
        self.track_timeout_frames = 12
        self.max_centroid_distance = 110
        self.smoothing_alpha = 0.15
        self.velocity_alpha = 0.50  

    def get_centroid(self, bbox):
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def is_box_in_frame(self, bbox, h, w):
        cx, cy = self.get_centroid(bbox)
        return (0 <= cx <= w) and (0 <= cy <= h)

    def update_tracks(self, detections, frame_counter, frame_h, frame_w, current_raw_frame=None):
        tracks = list(self.active_tracks)
        matched_indices = set()
        new_tracks = []

        for track in tracks:
            if 'centroid' not in track: track['centroid'] = self.get_centroid(track['bbox'])
            if 'velocity' not in track: track['velocity'] = (0.0, 0.0)

        for det in detections:
            det_c = self.get_centroid(det['bbox'])
            best_dist = self.max_centroid_distance + 1
            best_idx = None
            
            for idx, track in enumerate(tracks):
                if idx in matched_indices: continue
                pred_cx = track['centroid'][0] + track['velocity'][0]
                pred_cy = track['centroid'][1] + track['velocity'][1]
                dist = ((pred_cx - det_c[0])**2 + (pred_cy - det_c[1])**2)**0.5
                if dist < best_dist and dist <= self.max_centroid_distance:
                    best_dist = dist
                    best_idx = idx

            if best_idx is not None:
                track = tracks[best_idx]
                old_centroid = track['centroid']
                vx, vy = track['velocity']
                new_vx = self.velocity_alpha * vx + (1 - self.velocity_alpha) * (det_c[0] - old_centroid[0])
                new_vy = self.velocity_alpha * vy + (1 - self.velocity_alpha) * (det_c[1] - old_centroid[1])
                
                alpha_box = 0.35
                ox1, oy1, ox2, oy2 = track['bbox']
                nx1, ny1, nx2, ny2 = det['bbox']
                track['bbox'] = (
                    int(ox1 * (1 - alpha_box) + nx1 * alpha_box),
                    int(oy1 * (1 - alpha_box) + ny1 * alpha_box),
                    int(ox2 * (1 - alpha_box) + nx2 * alpha_box),
                    int(oy2 * (1 - alpha_box) + ny2 * alpha_box)
                )
                
                track['centroid'] = det_c
                track['velocity'] = (new_vx, new_vy)
                track['conf'] = det['conf']
                track['last_seen'] = frame_counter
                track['v_type'] = det.get('v_type', 'motorcycle')
                
                if det['is_violation'] and not track.get('is_violation', False):
                    track['is_violation'] = True
                    if track['track_id'] not in self.saved_violation_ids:
                        self.saved_violation_ids.add(track['track_id'])
                        self.total_violations += 1
                        self.total_safe = max(self.total_safe - 1, 0)
                        if current_raw_frame is not None:
                            crop_box = det.get('vehicle_bbox', det['bbox'])
                            self.trigger_screenshot_capture(current_raw_frame, crop_box, track['track_id'])
                        
                matched_indices.add(best_idx)
            else:
                is_viol = det['is_violation']
                assigned_id = self.next_track_id
                self.next_track_id += 1
                
                new_track = {
                    'track_id': assigned_id,
                    'bbox': det['bbox'],
                    'conf': det['conf'],
                    'is_violation': is_viol,
                    'last_seen': frame_counter,
                    'centroid': det_c,
                    'velocity': (0.0, 0.0),
                    'v_type': det.get('v_type', 'motorcycle')
                }
                new_tracks.append(new_track)
                
                if is_viol:
                    if assigned_id not in self.saved_violation_ids:
                        self.saved_violation_ids.add(assigned_id)
                        self.total_violations += 1
                        if current_raw_frame is not None:
                            crop_box = det.get('vehicle_bbox', det['bbox'])
                            self.trigger_screenshot_capture(current_raw_frame, crop_box, assigned_id)
                else:
                    self.total_safe += 1

        kept = []
        for track in tracks:
            if frame_counter - track['last_seen'] <= self.track_timeout_frames and self.is_box_in_frame(track['bbox'], frame_h, frame_w):
                kept.append(track)

        self.active_tracks = kept + new_tracks
        self.root.after(0, self.update_stats)

    def detect_image(self):
        file_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not file_path: return
            
        self.status_label.config(text="Processing asset frame mapping matrix...")
        self.set_live_status(True, "COMPUTE")
        self.root.update()
        
        try:
            img = cv2.imread(file_path)
            
            base_res = self.base_model.predict(source=img, conf=0.25, classes=[1, 3], verbose=False)[0]
            vehicles = []
            if base_res.boxes is not None:
                for b in base_res.boxes:
                    vehicles.append({
                        'bbox': [int(c) for c in b.xyxy[0].tolist()],
                        'class': int(b.cls[0])
                    })
            
            results = self.model.predict(source=img, conf=self.confidence_var.get(), imgsz=640, save=False, verbose=False)[0]
            annotated = img.copy()
            
            if results.boxes is not None:
                for box in results.boxes:
                    coords = [int(c) for c in box.xyxy[0].tolist()]
                    cls_name = self.model.names[int(box.cls[0])]
                    is_viol = 'without' in cls_name.lower() or 'no' in cls_name.lower()
                    
                    matched_v = self.find_associated_vehicle(coords, vehicles)
                    if matched_v is not None:
                        is_motorcycle = (matched_v['class'] == 3)
                        effective_violation = is_viol if is_motorcycle else False
                        
                        if effective_violation:
                            hx1, hy1, hx2, hy2 = coords
                            vx1, vy1, vx2, vy2 = matched_v['bbox']
                            ux1 = min(hx1, vx1)
                            uy1 = min(hy1, vy1)
                            ux2 = max(hx2, vx2)
                            uy2 = max(hy2, vy2)
                            
                            hp = int((uy2 - uy1) * 0.08)
                            wp = int((ux2 - ux1) * 0.08)
                            matched_vehicle = [ux1 - wp, uy1 - hp, ux2 + wp, uy2 + hp]
                            
                            cv2.rectangle(annotated, (coords[0], coords[1]), (coords[2], coords[3]), (0,0,255), 2)
                            cv2.putText(annotated, "RIDER: NO HELMET (MOTORCYCLE)", (coords[0], coords[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                            self.trigger_screenshot_capture(img, matched_vehicle, "STATIC")
                        else:
                            if not is_motorcycle:
                                cv2.rectangle(annotated, (coords[0], coords[1]), (coords[2], coords[3]), (0,255,0), 2)
                                cv2.putText(annotated, "BICYCLE RIDER", (coords[0], coords[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                            else:
                                self.total_safe += 1
                                cv2.rectangle(annotated, (coords[0], coords[1]), (coords[2], coords[3]), (0,255,0), 2)
                                cv2.putText(annotated, "SAFE MOTORCYCLE", (coords[0], coords[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            self.display_image(annotated_rgb)
            self.frame_count += 1
            self.update_stats()
            self.set_live_status(False, "COMPLETE")
            self.status_label.config(text="Single frame inference pipeline done.")
        except Exception as e:
            messagebox.showerror("Error", f"Detection pipeline failure: {str(e)}")
            self.set_live_status(False, "ERROR")
            
    def detect_video(self):
        file_path = filedialog.askopenfilename(title="Select Video Stream Source", filetypes=[("Video assets", "*.mp4 *.avi *.mov *.mkv")])
        if not file_path: return
            
        self.stop_all()
        self.reset_stats()
        
        self.is_video_running = True
        self.btn_video.set_state(False)
        self.btn_image.set_state(False)
        self.btn_webcam_start.set_state(False)
        self.btn_webcam_stop.set_state(True)
        self.set_live_status(True, "VIDEO")
        self.append_to_log("Asynchronous video stream framework launched.")
        
        self.video_thread = threading.Thread(target=self.process_video_optimized, args=(file_path,), daemon=True)
        self.video_thread.start()
        
    def process_video_optimized(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.is_video_running = False
            return
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_counter = 0
        fps_counter = 0
        fps_start_time = time.time()
        
        while self.is_video_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_counter += 1
            fps_counter += 1
            
            current_time = time.time()
            if current_time - fps_start_time >= 1.0:
                self.fps = fps_counter
                fps_counter = 0
                fps_start_time = current_time
            
            h, w = frame.shape[:2]
            scale = min(self.process_width / w, self.process_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            if frame_counter % self.process_skip == 0:
                self.frame_count = frame_counter
                small_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                base_results = self.base_model.predict(small_frame, conf=0.3, classes=[1, 3], verbose=False)[0]
                vehicles = []
                if base_results.boxes is not None:
                    for b in base_results.boxes:
                        vc = b.xyxy[0].tolist()
                        vehicles.append({
                            'bbox': (int(vc[0]/scale), int(vc[1]/scale), int(vc[2]/scale), int(vc[3]/scale)),
                            'class': int(b.cls[0])
                        })
                
                results = self.model.predict(small_frame, conf=self.confidence_var.get(), imgsz=self.process_width, verbose=False)[0]
                detections = []
                
                if results.boxes is not None:
                    for box in results.boxes:
                        coords = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = int(coords[0]/scale), int(coords[1]/scale), int(coords[2]/scale), int(coords[3]/scale)
                        cls_name = self.model.names[int(box.cls[0])]
                        is_violation = 'without' in cls_name.lower() or 'no' in cls_name.lower()
                        
                        matched_v = self.find_associated_vehicle((x1, y1, x2, y2), vehicles)
                        if matched_v is not None:
                            is_motorcycle = (matched_v['class'] == 3)
                            effective_violation = is_violation if is_motorcycle else False
                            
                            vx1, vy1, vx2, vy2 = matched_v['bbox']
                            ux1 = min(x1, vx1)
                            uy1 = min(y1, vy1)
                            ux2 = max(x2, vx2)
                            uy2 = max(y2, vy2)
                            
                            hp = int((uy2 - uy1) * 0.08)
                            wp = int((ux2 - ux1) * 0.08)
                            matched_union = (ux1 - wp, uy1 - hp, ux2 + wp, uy2 + hp)

                            detections.append({
                                'bbox': (x1, y1, x2, y2),
                                'conf': float(box.conf[0]),
                                'is_violation': effective_violation,
                                'vehicle_bbox': matched_union,
                                'v_type': 'motorcycle' if is_motorcycle else 'bicycle'
                            })
                self.update_tracks(detections, frame_counter, h, w, current_raw_frame=frame)
            else:
                for track in self.active_tracks:
                    if 'centroid' in track and 'velocity' in track:
                        cx = track['centroid'][0] + track['velocity'][0]
                        cy = track['centroid'][1] + track['velocity'][1]
                        track['centroid'] = (cx, cy)
                        dx = track['bbox'][2] - track['bbox'][0]
                        dy = track['bbox'][3] - track['bbox'][1]
                        track['bbox'] = (max(0, int(cx - dx / 2)), max(0, int(cy - dy / 2)), min(w - 1, int(cx + dx / 2)), min(h - 1, int(cy + dy / 2)))
            
            for track in self.active_tracks:
                x1, y1, x2, y2 = track['bbox']
                if track['is_violation']:
                    color = (0, 0, 255)
                    label = f"RIDER ID {track['track_id']} - NO HELMET (MOTORCYCLE)"
                else:
                    color = (0, 255, 0)
                    v_label = "BICYCLE" if track.get('v_type') == 'bicycle' else "MOTORCYCLE"
                    label = f"RIDER ID {track['track_id']} - SAFE ({v_label})"
                    
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                
            display_frame = cv2.resize(frame, (self.display_width, self.display_height))
            display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            try:
                self.frame_queue.put(display_frame_rgb, block=False)
            except queue.Full:
                pass
                
            if frame_counter % 15 == 0:
                self.root.after(0, lambda f=frame_counter: self.status_label.config(text=f"Processing file array: {f}/{total_frames if total_frames > 0 else '?'}"))
                
        cap.release()
        self.is_video_running = False
        self.root.after(0, self.stop_all)
        
    def start_webcam(self):
        self.stop_all()
        self.reset_stats()
        
        self.is_webcam_running = True
        self.btn_webcam_start.set_state(False)
        self.btn_webcam_stop.set_state(True)
        self.btn_video.set_state(False)
        self.btn_image.set_state(False)
        self.set_live_status(True, "LIVE")
        self.append_to_log("Real-time telemetry webcam interface online.")
        
        self.video_thread = threading.Thread(target=self.process_webcam, daemon=True)
        self.video_thread.start()

    def process_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.root.after(0, lambda: self.status_label.config(text="Hardware Error: Capture device missing."))
            self.root.after(0, self.stop_all)
            return
            
        frame_counter = 0
        fps_counter = 0
        fps_start_time = time.time()
        
        while self.is_webcam_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break
                
            frame_counter += 1
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= 1.0:
                self.fps = fps_counter
                fps_counter = 0
                fps_start_time = current_time
                
            h, w = frame.shape[:2]
            scale = min(self.process_width / w, self.process_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            if frame_counter % self.process_skip == 0:
                self.frame_count = frame_counter
                small_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                base_results = self.base_model.predict(small_frame, conf=0.3, classes=[1, 3], verbose=False)[0]
                vehicles = []
                if base_results.boxes is not None:
                    for b in base_results.boxes:
                        vc = b.xyxy[0].tolist()
                        vehicles.append({
                            'bbox': (int(vc[0]/scale), int(vc[1]/scale), int(vc[2]/scale), int(vc[3]/scale)),
                            'class': int(b.cls[0])
                        })
                
                results = self.model.predict(small_frame, conf=self.confidence_var.get(), imgsz=self.process_width, verbose=False)[0]
                detections = []
                
                if results.boxes is not None:
                    for box in results.boxes:
                        coords = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = int(coords[0]/scale), int(coords[1]/scale), int(coords[2]/scale), int(coords[3]/scale)
                        cls_name = self.model.names[int(box.cls[0])]
                        is_violation = 'without' in cls_name.lower() or 'no' in cls_name.lower()
                        
                        matched_v = self.find_associated_vehicle((x1, y1, x2, y2), vehicles)
                        if matched_v is not None:
                            is_motorcycle = (matched_v['class'] == 3)
                            effective_violation = is_violation if is_motorcycle else False
                            
                            vx1, vy1, vx2, vy2 = matched_v['bbox']
                            ux1 = min(x1, vx1)
                            uy1 = min(y1, vy1)
                            ux2 = max(x2, vx2)
                            uy2 = max(y2, vy2)
                            
                            hp = int((uy2 - uy1) * 0.08)
                            wp = int((ux2 - ux1) * 0.08)
                            matched_union = (ux1 - wp, uy1 - hp, ux2 + wp, uy2 + hp)

                            detections.append({
                                'bbox': (x1, y1, x2, y2),
                                'conf': float(box.conf[0]),
                                'is_violation': effective_violation,
                                'vehicle_bbox': matched_union,
                                'v_type': 'motorcycle' if is_motorcycle else 'bicycle'
                            })
                self.update_tracks(detections, frame_counter, h, w, current_raw_frame=frame)
            else:
                for track in self.active_tracks:
                    if 'centroid' in track and 'velocity' in track:
                        cx = track['centroid'][0] + track['velocity'][0]
                        cy = track['centroid'][1] + track['velocity'][1]
                        track['centroid'] = (cx, cy)
                        dx = track['bbox'][2] - track['bbox'][0]
                        dy = track['bbox'][3] - track['bbox'][1]
                        track['bbox'] = (max(0, int(cx - dx / 2)), max(0, int(cy - dy / 2)), min(w - 1, int(cx + dx / 2)), min(h - 1, int(cy + dy / 2)))
                        
            for track in self.active_tracks:
                x1, y1, x2, y2 = track['bbox']
                if track['is_violation']:
                    color = (0, 0, 255)
                    label = f"RIDER ID {track['track_id']} - NO HELMET (MOTORCYCLE)"
                else:
                    color = (0, 255, 0)
                    v_label = "BICYCLE" if track.get('v_type') == 'bicycle' else "MOTORCYCLE"
                    label = f"RIDER ID {track['track_id']} - SAFE ({v_label})"
                    
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                
            display_frame = cv2.resize(frame, (self.display_width, self.display_height))
            display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            try:
                self.frame_queue.put(display_frame_rgb, block=False)
            except queue.Full:
                pass
            time.sleep(0.01)
            
        if self.cap: 
            self.cap.release()

    def stop_all(self):
        self.is_video_running = False
        self.is_webcam_running = False
        
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=1.0)
            
        if self.cap:
            self.cap.release()
            self.cap = None
            
        while not self.frame_queue.empty():
            try: 
                self.frame_queue.get_nowait()
            except queue.Empty: 
                break
                
        self.btn_video.set_state(True)
        self.btn_image.set_state(True)
        self.btn_webcam_start.set_state(True)
        self.btn_webcam_stop.set_state(False)
        self.set_live_status(False, "READY")
        self.draw_placeholder()
        self.append_to_log("Active asynchronous feed channels cleanly terminated.")

    def update_display_loop(self):
        try:
            frame_rgb = None
            while not self.frame_queue.empty():
                frame_rgb = self.frame_queue.get_nowait()
            if frame_rgb is not None:
                self.display_image(frame_rgb)
        except Exception as e:
            print(f"Loop error handling framework exception: {e}")
            
        self.root.after(15, self.update_display_loop)

    def display_image(self, img_rgb):
        img_pil = Image.fromarray(img_rgb)
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w < 10: canvas_w = self.display_width
        if canvas_h < 10: canvas_h = self.display_height
        
        img_pil = img_pil.resize((canvas_w, canvas_h), Image.Resampling.LANCZOS)
        
        self.photo = ImageTk.PhotoImage(image=img_pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def draw_placeholder(self):
        if self.is_video_running or self.is_webcam_running:
            return
            
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        
        if w < 10: w = self.display_width
        if h < 10: h = self.display_height
        
        self.canvas.delete("all")
        self.canvas.create_text(w // 2, h // 2 - 20, text="🤖", font=('Segoe UI Emoji', 40), fill='#334155', anchor=tk.CENTER)
        self.canvas.create_text(w // 2, h // 2 + 35, text="System Core Pipeline Disconnected", font=('Segoe UI', 12, 'bold'), fill='#475569', anchor=tk.CENTER)

    def generate_report(self):
        if not self.violation_records:
            messagebox.showinfo("Export Analytics", "No verified rider spatial violations recorded in the active database session.")
            return
            
        report_path = filedialog.asksaveasfilename(defaultextension=".html", filetypes=[("HTML Report Summary", "*.html")], title="Save Analytics Report File")
        if not report_path: return
        
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Traffic Intelligence System - Session Audit Report</title>
                <style>
                    body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #121212; color: #e2e8f0; margin: 0; padding: 40px; }}
                    .container {{ max-width: 1100px; margin: 0 auto; }}
                    h1 {{ color: #ffffff; border-bottom: 2px solid #2d2d2d; padding-bottom: 12px; font-size: 24px; }}
                    .metrics {{ display: flex; gap: 15px; margin: 20px 0; }}
                    .card {{ background: #1e293b; border: 1px solid #2d2d2d; padding: 20px; border-radius: 6px; flex: 1; text-align: center; }}
                    .card h3 {{ margin: 0 0 8px 0; color: #9ca3af; font-size: 12px; text-transform: uppercase; }}
                    .card p {{ margin: 0; font-size: 28px; font-weight: bold; }}
                    .text-red {{ color: #ef4444; }} .text-green {{ color: #10b981; }} .text-blue {{ color: #3b82f6; }}
                    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 20px; margin-top: 30px; }}
                    .log-card {{ background: #1e293b; border: 1px solid #2d2d2d; border-radius: 6px; overflow: hidden; transition: transform 0.2s ease, box-shadow 0.2s ease; }}
                    .log-card:hover {{ transform: translateY(-4px); box-shadow: 0 12px 20px rgba(0,0,0,0.4); }}
                    .log-card img {{ width: 100%; height: 200px; object-fit: contain; background: #0b0f19; display: block; border-bottom: 1px solid #2d2d2d; transition: transform 0.3s ease; }}
                    .log-card:hover img {{ transform: scale(1.1); }}
                    .meta {{ padding: 12px; font-size: 13px; }}
                    .meta-row {{ display: flex; justify-content: space-between; margin-bottom: 4px; }}
                    .lbl {{ color: #9ca3af; }}
                    .val {{ font-weight: bold; color: #ffffff; }}
                    footer {{ margin-top: 50px; text-align: center; font-size: 11px; color: #4b5563; border-top: 1px solid #2d2d2d; padding-top: 15px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>📊 Traffic Intelligence Audit Summary</h1>
                    <p style="color: #9ca3af;">Session Timestamp Log: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <div class="metrics">
                        <div class="card"><h3>Frames Analyzed</h3><p class="text-blue">{self.frame_count}</p></div>
                        <div class="card"><h3>Breach Infractions</h3><p class="text-red">{self.total_violations}</p></div>
                        <div class="card"><h3>Safe Verified Riders</h3><p class="text-green">{self.total_safe}</p></div>
                    </div>
                    
                    <h2>📋 Evidence Bounding Repository</h2>
                    <div class="grid">
            """
            for item in self.violation_records:
                html_content += f"""
                        <div class="log-card">
                            <img src="file:///{item['img_path']}" alt="Rider Crop Element">
                            <div class="meta">
                                <div class="meta-row"><span class="lbl">Rider Track ID:</span><span class="val text-red">#{item['track_id']}</span></div>
                                <div class="meta-row"><span class="lbl">Timestamp:</span><span class="val">{item['timestamp'].split()[-1]}</span></div>
                            </div>
                        </div>
                """
            html_content += """
                    </div>
                    <footer>Developed by Code_IT</footer>
                </div>
            </body>
            </html>
            """
            with open(report_path, "w", encoding="utf-8") as file:
                file.write(html_content)
            messagebox.showinfo("Report Exported", f"Audit metrics exported successfully to:\n{report_path}")
            self.append_to_log("Session HTML analytics report written successfully.")
        except Exception as e:
            messagebox.showerror("Report Error", f"Failed compiling report metrics: {str(e)}")

    def generate_pdf_report(self):
        if not self.violation_records:
            messagebox.showinfo("Export Analytics", "No verified rider spatial violations recorded in the active database session.")
            return
            
        report_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Report Summary", "*.pdf")], title="Save Analytics PDF Report")
        if not report_path: return
        
        try:
            pdf = FPDF()
            pdf.add_page()
            
            pdf.set_font("Helvetica", "B", 18)
            pdf.cell(0, 15, "Traffic Intelligence Session Audit Report", ln=1, align="C")
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1, align="C")
            pdf.ln(5)
            
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 10, "Summary Metrics:", ln=1)
            pdf.set_font("Helvetica", "", 11)
            pdf.cell(60, 8, f"Frames Analyzed: {self.frame_count}", border=1, align="C")
            pdf.cell(60, 8, f"Breach Infractions: {self.total_violations}", border=1, align="C")
            pdf.cell(60, 8, f"Safe Verified Riders: {self.total_safe}", border=1, align="C")
            pdf.ln(15)
            
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, "Evidence Bounding Repository:", ln=1)
            pdf.ln(5)
            
            for item in self.violation_records:
                if pdf.get_y() > 200:
                    pdf.add_page()
                    
                pdf.set_font("Helvetica", "B", 11)
                pdf.cell(0, 8, f"Rider Track ID: #{item['track_id']}   |   Timestamp: {item['timestamp'].split()[-1]}", ln=1)
                
                if os.path.exists(item['img_path']):
                    try:
                        pdf.image(item['img_path'], w=60)
                    except Exception as img_err:
                        pdf.set_font("Helvetica", "I", 9)
                        pdf.cell(0, 6, f"[Image display error: {str(img_err)}]", ln=1)
                else:
                    pdf.set_font("Helvetica", "I", 9)
                    pdf.cell(0, 6, "[Evidence image file missing]", ln=1)
                pdf.ln(10)
                
            pdf.set_y(-20)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 10, "Developed by Code_IT", align="C")
            
            pdf.output(report_path)
            messagebox.showinfo("Report Exported", f"Audit PDF report exported successfully to:\n{report_path}")
            self.append_to_log("Session PDF analytics report written successfully.")
            self.status_label.config(text="PDF Report export completed.")
        except Exception as e:
            messagebox.showerror("Report Error", f"Failed compiling PDF report metrics: {str(e)}")

    def on_closing(self):
        self.stop_all()
        self.root.destroy()


if __name__ == "__main__":
    # CRITICAL: Ensures internal multiprocessing streams do not recursively loop your Tkinter setup inside an executable
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = HelmetDetectionGUI(root)
    root.mainloop()