import customtkinter as ctk
import cv2
import threading
import time
import csv
from tkinter import messagebox, filedialog
from PIL import Image
import numpy as np

# Import our backend modules
import db_manager as db
from vector_index import VectorIndex
from face_engine import FaceEngine

class BiometricDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # 1. Palette & Modern Dark Aesthetic
        self.title("AURA-FACE | Biometric Kiosk & Management")
        self.geometry("1240x780")
        self.minsize(1240, 780)
        
        ctk.set_appearance_mode("Dark")
        # Define custom colors
        self.colors = {
            "bg_deep": "#0B0F19",
            "surface": "#151D2E",
            "border": "#22304A",
            "green": "#10B981",
            "blue": "#3B82F6",
            "red": "#EF4444",
            "purple": "#8B5CF6",
            "text_main": "#F8FAFC",
            "text_muted": "#94A3B8"
        }
        self.configure(fg_color=self.colors["bg_deep"])
        
        # Application State
        self.mode = "KIOSK"
        self.cap = None
        self.camera_running = False
        self.enrollment_mode = False
        self.enrollment_crops = []
        self.enrollment_target = 5
        self.enrollment_user_id = None
        
        # Initialize Backend Engines
        try:
            self.face_engine = FaceEngine()
            self.vector_index = VectorIndex()
        except Exception as e:
            messagebox.showerror("Engine Initialization Error", f"Failed to load AI models:\n{e}")
            
        # UI Setup
        self.setup_header()
        self.setup_kiosk_view()
        self.update_clock()
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_header(self):
        self.header_frame = ctk.CTkFrame(self, fg_color=self.colors["surface"], corner_radius=0, height=60)
        self.header_frame.pack(fill="x", side="top")
        
        # Branding
        title = ctk.CTkLabel(self.header_frame, text="AURA-FACE | Biometric Kiosk & Management", 
                             font=ctk.CTkFont(size=20, weight="bold"), text_color=self.colors["text_main"])
        title.pack(side="left", padx=20, pady=15)
        
        # Controls (Right side)
        self.admin_btn = ctk.CTkButton(self.header_frame, text="Admin Portal", fg_color="transparent",
                                       border_width=2, border_color=self.colors["blue"], text_color=self.colors["blue"],
                                       hover_color="#1E3A8A", command=self.open_admin_auth)
        self.admin_btn.pack(side="right", padx=20, pady=15)
        
        self.clock_label = ctk.CTkLabel(self.header_frame, text="00:00:00", font=ctk.CTkFont(family="Consolas", size=18), text_color=self.colors["text_muted"])
        self.clock_label.pack(side="right", padx=20, pady=15)
        
    def update_clock(self):
        current_time = time.strftime("%H:%M:%S")
        self.clock_label.configure(text=current_time)
        self.after(1000, self.update_clock)
        
    def setup_kiosk_view(self):
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # --- Left Card (Camera Viewport) ---
        self.left_card = ctk.CTkFrame(self.main_container, fg_color=self.colors["surface"], border_width=1, border_color=self.colors["border"])
        self.left_card.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        cam_header = ctk.CTkFrame(self.left_card, fg_color="transparent")
        cam_header.pack(fill="x", padx=20, pady=15)
        ctk.CTkLabel(cam_header, text="Live Scanner Terminal", font=ctk.CTkFont(size=18, weight="bold")).pack(side="left")
        self.cam_badge = ctk.CTkLabel(cam_header, text="● STANDBY", text_color=self.colors["text_muted"], font=ctk.CTkFont(weight="bold"))
        self.cam_badge.pack(side="right")
        
        self.video_label = ctk.CTkLabel(self.left_card, text="Camera is Idle.\nClick 'Start Attendance Scan' to begin", 
                                        font=ctk.CTkFont(size=16), text_color=self.colors["text_muted"])
        self.video_label.pack(fill="both", expand=True, padx=20, pady=10)
        
        cam_controls = ctk.CTkFrame(self.left_card, fg_color="transparent")
        cam_controls.pack(fill="x", padx=20, pady=20)
        
        self.start_scan_btn = ctk.CTkButton(cam_controls, text="▶ Start Attendance Scan", fg_color=self.colors["green"], 
                                            hover_color="#059669", height=40, font=ctk.CTkFont(weight="bold"), command=self.start_kiosk_camera)
        self.start_scan_btn.pack(side="left", expand=True, fill="x", padx=(0, 10))
        
        self.stop_scan_btn = ctk.CTkButton(cam_controls, text="⏹ Stop", fg_color=self.colors["red"], 
                                           hover_color="#B91C1C", height=40, width=100, font=ctk.CTkFont(weight="bold"), command=self.stop_camera)
        self.stop_scan_btn.pack(side="right")
        
        # --- Right Card (Dashboard Analytics & Feed) ---
        self.right_card = ctk.CTkFrame(self.main_container, fg_color=self.colors["surface"], border_width=1, border_color=self.colors["border"])
        self.right_card.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        ctk.CTkLabel(self.right_card, text="Daily Analytics", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=20, pady=(15, 5))
        
        self.kpi_grid = ctk.CTkFrame(self.right_card, fg_color="transparent")
        self.kpi_grid.pack(fill="x", padx=20, pady=10)
        self.kpi_grid.columnconfigure((0, 1), weight=1)
        
        self.kpi_total = self.create_kpi_card(self.kpi_grid, "Registered Total", "0", self.colors["blue"], 0, 0)
        self.kpi_present = self.create_kpi_card(self.kpi_grid, "Present Today", "0", self.colors["green"], 0, 1)
        self.kpi_absent = self.create_kpi_card(self.kpi_grid, "Absent Today", "0", self.colors["red"], 1, 0)
        self.kpi_rate = self.create_kpi_card(self.kpi_grid, "Attendance Rate %", "0%", self.colors["purple"], 1, 1)
        
        ctk.CTkLabel(self.right_card, text="Live Punch Stream", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=20, pady=(20, 5))
        
        self.stream_scroll = ctk.CTkScrollableFrame(self.right_card, fg_color="transparent")
        self.stream_scroll.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        self.refresh_dashboard()
        
    def create_kpi_card(self, parent, title, value, color, row, col):
        card = ctk.CTkFrame(parent, fg_color=self.colors["bg_deep"], corner_radius=8, border_width=1, border_color=color)
        card.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)
        ctk.CTkLabel(card, text=title, text_color=self.colors["text_muted"], font=ctk.CTkFont(size=12)).pack(anchor="w", padx=10, pady=(10, 0))
        val_label = ctk.CTkLabel(card, text=value, text_color=color, font=ctk.CTkFont(size=24, weight="bold"))
        val_label.pack(anchor="w", padx=10, pady=(0, 10))
        return val_label
        
    def refresh_dashboard(self):
        stats = db.get_daily_attendance_stats()
        self.kpi_total.configure(text=str(stats['total']))
        self.kpi_present.configure(text=str(stats['present']))
        self.kpi_absent.configure(text=str(stats['absent']))
        self.kpi_rate.configure(text=f"{stats['percentage']}%")
        
        for widget in self.stream_scroll.winfo_children():
            widget.destroy()
            
        records = db.get_today_attendance()
        for rec in records:
            # rec = (user_code, name, date, time)
            card = ctk.CTkFrame(self.stream_scroll, fg_color=self.colors["bg_deep"], corner_radius=6, border_width=1, border_color=self.colors["border"])
            card.pack(fill="x", pady=4)
            info_frame = ctk.CTkFrame(card, fg_color="transparent")
            info_frame.pack(side="left", padx=10, pady=10)
            ctk.CTkLabel(info_frame, text=rec[1], font=ctk.CTkFont(weight="bold", size=14)).pack(anchor="w")
            ctk.CTkLabel(info_frame, text=f"ID: {rec[0]}", font=ctk.CTkFont(size=12), text_color=self.colors["text_muted"]).pack(anchor="w")
            
            badge = ctk.CTkFrame(card, fg_color=self.colors["green"], corner_radius=4)
            badge.pack(side="right", padx=10)
            ctk.CTkLabel(badge, text=rec[3], font=ctk.CTkFont(size=12, weight="bold"), text_color="#000000").pack(padx=8, pady=4)

    # --- KIOSK Camera Logic ---
    def start_kiosk_camera(self):
        if not self.camera_running:
            self.cap = cv2.VideoCapture(0)
            self.camera_running = True
            self.mode = "KIOSK"
            self.cam_badge.configure(text="● SCANNING ACTIVE", text_color=self.colors["green"])
            self.video_label.configure(text="")
            self.update_kiosk_camera()
            
    def update_kiosk_camera(self):
        if not self.camera_running or self.mode != "KIOSK":
            return
            
        ret, frame = self.cap.read()
        if ret:
            # Detect faces
            try:
                faces = self.face_engine.detect_faces(frame)
            except Exception:
                faces = None
                
            verified_info = None
            
            if faces is not None:
                for face in faces:
                    x, y, w, h = face[:4].astype(int)
                    # Expand bounding box slightly for the crop
                    pad_x = int(w * 0.1)
                    pad_y = int(h * 0.1)
                    x1 = max(0, x - pad_x)
                    y1 = max(0, y - pad_y)
                    x2 = min(frame.shape[1], x + w + pad_x)
                    y2 = min(frame.shape[0], y + h + pad_y)
                    
                    face_crop = frame[y1:y2, x1:x2]
                    
                    if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                        try:
                            # Extract embedding
                            embedding = self.face_engine.extract_embedding(face_crop)
                            
                            # Search in vector index
                            uid, dist = self.vector_index.search(embedding, threshold=0.55)
                            
                            if uid is not None:
                                user_info = db.get_user_by_id(int(uid))
                                if user_info:
                                    user_code, name = user_info
                                    verified_info = (user_code, name)
                                    
                                    # Draw HUD Box
                                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                    cv2.putText(frame, f"{name} ({user_code})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                else:
                                    # UID in index but not in DB?
                                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                                    cv2.putText(frame, "Unknown Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            else:
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                                cv2.putText(frame, "Unknown Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        except Exception as e:
                            print(f"Embedding extraction error: {e}")
                            
            # Convert to RGB and display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(frame.shape[1], frame.shape[0]))
            
            self.video_label.imgtk = ctk_img
            self.video_label.configure(image=ctk_img, text="")
            
            if verified_info:
                # Mark attendance
                success, timestamp = db.mark_attendance(verified_info[0], verified_info[1])
                self.cam_badge.configure(text=f"✔ VERIFIED: {verified_info[1]}", text_color=self.colors["green"])
                self.refresh_dashboard()
                
                # Auto-Off Mechanism
                self.camera_running = False
                self.after(1500, self.stop_camera)
                return
                
        self.after(15, self.update_kiosk_camera)
        
    def stop_camera(self):
        self.camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.configure(image="", text="Camera is Idle.\nClick 'Start Attendance Scan' to begin")
        self.cam_badge.configure(text="● STANDBY", text_color=self.colors["text_muted"])
        
    def on_closing(self):
        self.stop_camera()
        self.destroy()
        
    # --- Admin Authentication ---
    def open_admin_auth(self):
        if hasattr(self, 'admin_window') and self.admin_window is not None and self.admin_window.winfo_exists():
            self.admin_window.focus()
            return
            
        dialog = ctk.CTkToplevel(self)
        dialog.title("Admin Authentication")
        dialog.geometry("320x350")
        dialog.transient(self)
        dialog.grab_set()
        dialog.configure(fg_color=self.colors["bg_deep"])
        
        ctk.CTkLabel(dialog, text="🔒 Admin Portal", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(30, 20))
        
        user_entry = ctk.CTkEntry(dialog, placeholder_text="Username", width=200)
        user_entry.pack(pady=10)
        
        pass_entry = ctk.CTkEntry(dialog, placeholder_text="Password", show="*", width=200)
        pass_entry.pack(pady=10)
        
        def attempt_login():
            if db.verify_admin(user_entry.get(), pass_entry.get()):
                dialog.destroy()
                self.stop_camera()
                self.open_admin_dashboard()
            else:
                messagebox.showerror("Auth Error", "Invalid Credentials", parent=dialog)
                
        ctk.CTkButton(dialog, text="Login", fg_color=self.colors["blue"], hover_color="#1E3A8A", command=attempt_login, width=200).pack(pady=(20, 10))
        
    # --- Admin Dashboard ---
    def open_admin_dashboard(self):
        self.admin_window = ctk.CTkToplevel(self)
        self.admin_window.title("Admin Dashboard - AURA-FACE")
        self.admin_window.geometry("1000x700")
        self.admin_window.minsize(1000, 700)
        self.admin_window.transient(self)
        self.admin_window.grab_set()
        self.admin_window.configure(fg_color=self.colors["bg_deep"])
        
        # Header
        header = ctk.CTkFrame(self.admin_window, fg_color=self.colors["surface"], corner_radius=0, height=60)
        header.pack(fill="x", side="top")
        ctk.CTkLabel(header, text="Admin Control Panel", font=ctk.CTkFont(size=20, weight="bold")).pack(side="left", padx=20, pady=15)
        
        ctk.CTkButton(header, text="Return to Kiosk", fg_color="transparent", border_width=1, border_color=self.colors["text_muted"], 
                      command=self.admin_window.destroy).pack(side="right", padx=20, pady=15)
                      
        # Tabview
        self.admin_tabs = ctk.CTkTabview(self.admin_window, fg_color=self.colors["bg_deep"])
        self.admin_tabs.pack(fill="both", expand=True, padx=20, pady=20)
        
        self.admin_tabs.add("User Management & Enrollment")
        self.admin_tabs.add("Attendance Analytics & Export")
        
        self.setup_enrollment_tab()
        self.setup_analytics_tab()
        
    def setup_enrollment_tab(self):
        tab = self.admin_tabs.tab("User Management & Enrollment")
        
        # Enrollment Form
        form_frame = ctk.CTkFrame(tab, fg_color=self.colors["surface"], corner_radius=8, border_width=1, border_color=self.colors["border"])
        form_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(form_frame, text="User ID / Roll No:").grid(row=0, column=0, padx=10, pady=(15, 5), sticky="w")
        self.reg_id_entry = ctk.CTkEntry(form_frame, width=150)
        self.reg_id_entry.grid(row=1, column=0, padx=10, pady=(0, 15))
        
        ctk.CTkLabel(form_frame, text="Full Name:").grid(row=0, column=1, padx=10, pady=(15, 5), sticky="w")
        self.reg_name_entry = ctk.CTkEntry(form_frame, width=200)
        self.reg_name_entry.grid(row=1, column=1, padx=10, pady=(0, 15))
        
        ctk.CTkLabel(form_frame, text="Registration Date:").grid(row=0, column=2, padx=10, pady=(15, 5), sticky="w")
        self.reg_date_entry = ctk.CTkEntry(form_frame, width=150)
        self.reg_date_entry.insert(0, time.strftime("%Y-%m-%d"))
        self.reg_date_entry.grid(row=1, column=2, padx=10, pady=(0, 15))
        
        self.enroll_btn = ctk.CTkButton(form_frame, text="Enroll Face Biometrics", fg_color=self.colors["purple"], hover_color="#7C3AED", command=self.start_enrollment_process)
        self.enroll_btn.grid(row=1, column=3, padx=20, pady=(0, 15))
        
        self.enroll_status = ctk.CTkLabel(form_frame, text="", text_color=self.colors["green"])
        self.enroll_status.grid(row=2, column=0, columnspan=4, pady=(0, 10))
        
        # Enrollment Camera View (Hidden by default)
        self.enroll_cam_frame = ctk.CTkFrame(tab, fg_color="transparent")
        self.enroll_cam_label = ctk.CTkLabel(self.enroll_cam_frame, text="")
        self.enroll_cam_label.pack()
        
        # Table View
        self.users_textbox = ctk.CTkTextbox(tab, fg_color=self.colors["surface"], border_width=1, border_color=self.colors["border"], font=ctk.CTkFont(family="Consolas", size=13))
        self.users_textbox.pack(fill="both", expand=True, pady=(0, 10))
        self.refresh_users_table()
        
        # User Management View
        manage_frame = ctk.CTkFrame(tab, fg_color=self.colors["surface"], corner_radius=8, border_width=1, border_color=self.colors["border"])
        manage_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(manage_frame, text="Manage User (Enter UID):").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.manage_uid_entry = ctk.CTkEntry(manage_frame, width=80)
        self.manage_uid_entry.grid(row=0, column=1, padx=10, pady=10)
        
        ctk.CTkButton(manage_frame, text="Edit Details", fg_color=self.colors["blue"], command=self.open_edit_user_dialog).grid(row=0, column=2, padx=10, pady=10)
        ctk.CTkButton(manage_frame, text="Delete User", fg_color=self.colors["red"], command=self.delete_user_action).grid(row=0, column=3, padx=10, pady=10)
        
    def refresh_users_table(self):
        if not hasattr(self, 'users_textbox'): return
        self.users_textbox.configure(state="normal")
        self.users_textbox.delete("1.0", "end")
        self.users_textbox.insert("end", f"{'UID':<6} | {'User Code':<15} | {'Full Name':<25} | {'Reg Date'}\n")
        self.users_textbox.insert("end", "-"*70 + "\n")
        users = db.get_all_users()
        for u in users:
            self.users_textbox.insert("end", f"{u[0]:<6} | {u[1]:<15} | {u[2]:<25} | {u[3]}\n")
        self.users_textbox.configure(state="disabled")
        
    def open_edit_user_dialog(self):
        uid_str = self.manage_uid_entry.get().strip()
        if not uid_str.isdigit():
            messagebox.showerror("Error", "Please enter a valid numeric UID")
            return
            
        uid = int(uid_str)
        user = db.get_user_by_id(uid)
        if not user:
            messagebox.showerror("Error", "User not found")
            return
            
        dialog = ctk.CTkToplevel(self.admin_window)
        dialog.title("Edit User")
        dialog.geometry("350x300")
        dialog.transient(self.admin_window)
        dialog.grab_set()
        
        ctk.CTkLabel(dialog, text=f"Editing User UID: {uid}", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=20)
        
        code_entry = ctk.CTkEntry(dialog, placeholder_text="User Code", width=200)
        code_entry.insert(0, user[0])
        code_entry.pack(pady=10)
        
        name_entry = ctk.CTkEntry(dialog, placeholder_text="Full Name", width=200)
        name_entry.insert(0, user[1])
        name_entry.pack(pady=10)
        
        def save_changes():
            success, msg = db.update_user(uid, code_entry.get().strip(), name_entry.get().strip())
            if success:
                messagebox.showinfo("Success", "User updated successfully", parent=dialog)
                dialog.destroy()
                self.refresh_users_table()
                self.manage_uid_entry.delete(0, 'end')
            else:
                messagebox.showerror("Error", msg, parent=dialog)
                
        ctk.CTkButton(dialog, text="Save Changes", command=save_changes, fg_color=self.colors["green"]).pack(pady=20)

    def delete_user_action(self):
        uid_str = self.manage_uid_entry.get().strip()
        if not uid_str.isdigit():
            messagebox.showerror("Error", "Please enter a valid numeric UID")
            return
            
        uid = int(uid_str)
        user = db.get_user_by_id(uid)
        if not user:
            messagebox.showerror("Error", "User not found")
            return
            
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete user {user[1]} ({user[0]})?\nThis will also remove their attendance and biometric data."):
            success, msg = db.delete_user(uid)
            if success:
                self.vector_index.remove_vector(uid)
                messagebox.showinfo("Success", "User deleted successfully")
                self.refresh_users_table()
                self.manage_uid_entry.delete(0, 'end')
            else:
                messagebox.showerror("Error", msg)

        
    def start_enrollment_process(self):
        code = self.reg_id_entry.get().strip()
        name = self.reg_name_entry.get().strip()
        reg_date = self.reg_date_entry.get().strip()
        
        if not code or not name:
            messagebox.showerror("Error", "Please fill in User ID and Name")
            return
            
        # Register in DB
        numeric_id = db.add_user(code, name, reg_date)
        if numeric_id == -1:
            messagebox.showerror("Error", "User Code already exists!")
            return
            
        self.enrollment_user_id = numeric_id
        self.enrollment_crops = []
        
        self.enroll_status.configure(text="Please look at the camera. Capturing 5 face samples...", text_color=self.colors["blue"])
        self.enroll_btn.configure(state="disabled")
        
        self.enroll_cam_frame.pack(fill="x", pady=10)
        self.users_textbox.pack_forget()
        
        self.mode = "ENROLL"
        self.cap = cv2.VideoCapture(0)
        self.camera_running = True
        self.update_enrollment_camera()
        
    def update_enrollment_camera(self):
        if not self.camera_running or self.mode != "ENROLL":
            return
            
        ret, frame = self.cap.read()
        if ret:
            try:
                faces = self.face_engine.detect_faces(frame)
            except Exception:
                faces = None
                
            if faces is not None and len(faces) > 0:
                # Use the largest face
                face = max(faces, key=lambda f: f[2]*f[3])
                x, y, w, h = face[:4].astype(int)
                
                pad_x = int(w * 0.1)
                pad_y = int(h * 0.1)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(frame.shape[1], x + w + pad_x)
                y2 = min(frame.shape[0], y + h + pad_y)
                
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                    embedding = self.face_engine.extract_embedding(face_crop)
                    self.enrollment_crops.append(embedding)
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Sample {len(self.enrollment_crops)}/{self.enrollment_target}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show preview
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
            self.enroll_cam_label.configure(image=ctk_img)
            
            if len(self.enrollment_crops) >= self.enrollment_target:
                self.finish_enrollment()
                return
                
        # 300ms delay to capture different expressions
        self.after(300, self.update_enrollment_camera)
        
    def finish_enrollment(self):
        self.camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
            
        self.enroll_cam_frame.pack_forget()
        self.users_textbox.pack(fill="both", expand=True)
        self.enroll_btn.configure(state="normal")
        
        # Calculate mean embedding
        mean_emb = np.mean(self.enrollment_crops, axis=0)
        # Re-normalize
        mean_emb = mean_emb / np.linalg.norm(mean_emb)
        mean_emb = mean_emb.astype(np.float32)
        
        # Check for duplicate face before adding
        try:
            uid, dist = self.vector_index.search(mean_emb, threshold=0.55)
            if uid is not None:
                # Duplicate found! Rollback the DB entry
                db.delete_user(self.enrollment_user_id) 
                self.enroll_status.configure(text=f"Error: Face already registered to ID {uid}", text_color=self.colors["red"])
                self.refresh_users_table()
                return
        except Exception as e:
            print(f"Error checking for duplicates: {e}")
        
        # Add to FAISS Index
        try:
            self.vector_index.add_vector(self.enrollment_user_id, mean_emb)
            self.enroll_status.configure(text="✔ User Biometrics Enrolled Successfully!", text_color=self.colors["green"])
            self.reg_id_entry.delete(0, 'end')
            self.reg_name_entry.delete(0, 'end')
        except Exception as e:
            self.enroll_status.configure(text=f"Error saving to index: {e}", text_color=self.colors["red"])
            
        self.refresh_users_table()
        
    def setup_analytics_tab(self):
        tab = self.admin_tabs.tab("Attendance Analytics & Export")
        
        ctrl_frame = ctk.CTkFrame(tab, fg_color="transparent")
        ctrl_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkButton(ctrl_frame, text="Export Today's Sheet (CSV)", command=self.export_today_csv, fg_color=self.colors["blue"]).pack(side="left", padx=5)
        ctk.CTkButton(ctrl_frame, text="Export Summary (CSV)", command=self.export_summary_csv, fg_color=self.colors["purple"]).pack(side="left", padx=5)
        
        self.analytics_textbox = ctk.CTkTextbox(tab, fg_color=self.colors["surface"], border_width=1, border_color=self.colors["border"], font=ctk.CTkFont(family="Consolas", size=13))
        self.analytics_textbox.pack(fill="both", expand=True)
        
        # Load Analytics Data
        self.refresh_analytics_table()
        
    def refresh_analytics_table(self):
        if not hasattr(self, 'analytics_textbox'): return
        summary = db.get_user_attendance_summary()
        self.analytics_textbox.configure(state="normal")
        self.analytics_textbox.delete("1.0", "end")
        self.analytics_textbox.insert("end", f"{'User Code':<15} | {'Name':<25} | {'Total Days':<12} | {'Present':<10} | {'Absent':<10} | {'Attendance %'}\n")
        self.analytics_textbox.insert("end", "-"*95 + "\n")
        for s in summary:
            self.analytics_textbox.insert("end", f"{s['user_code']:<15} | {s['name']:<25} | {s['total_days']:<12} | {s['present']:<10} | {s['absent']:<10} | {s['percentage']}%\n")
        self.analytics_textbox.configure(state="disabled")

    def export_today_csv(self):
        records = db.get_today_attendance()
        if not records:
            messagebox.showinfo("Info", "No attendance records today.")
            return
        
        today = time.strftime("%Y-%m-%d")
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=f"Attendance_{today}.csv")
        if path:
            with open(path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(["User Code", "Name", "Date", "Time"])
                w.writerows(records)
            messagebox.showinfo("Success", f"Exported to {path}")
            
    def export_summary_csv(self):
        summary = db.get_user_attendance_summary()
        if not summary:
            messagebox.showinfo("Info", "No data to export.")
            return
            
        today = time.strftime("%Y-%m-%d")
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=f"Attendance_Summary_{today}.csv")
        if path:
            with open(path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(["User Code", "Name", "Total Days", "Present", "Absent", "Attendance %"])
                for s in summary:
                    w.writerow([s['user_code'], s['name'], s['total_days'], s['present'], s['absent'], s['percentage']])
            messagebox.showinfo("Success", f"Exported to {path}")

if __name__ == "__main__":
    app = BiometricDashboard()
    app.mainloop()
