import os
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import time
import winsound
import torch
import threading


class YOLOApp:
    #Cu bind key sa se opreasca programu
    def bind_keys(self):
        self.root.bind("<Escape>", self.stop_detection)
    #Face Cleanup
    def stop_detection(self, event=None):
        print("Detectia orpita de mine.")
        self.running = False
        self.cleanup()

    #initializeaza aplicatia , creeaza intefata, incarca modelul
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("YOLO Detection App")
        self.fps_list = []
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        self.last_time = time.time()
        self.fps = 0

        print("CUDA available:", torch.cuda.is_available())
        print("cuDNN enabled:", torch.backends.cudnn.enabled)

        #incarca modelul direct pe gpu
        self.model = YOLO(model_path, task='detect').to('cuda')
        print("Model device:", next(self.model.model.parameters()).device)

        # Numele claselor detectate de mode
        self.labels = self.model.names
        #sursa, record,path
        self.cap = None
        self.record = False
        self.recorder = None
        self.output_path = None
        self.running = False
        self.original_fps = 30

        #ui pt afisare
        self.panel = tk.Label(root)
        self.panel.pack()

        #butoane
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        #lista de clase detectabile
        self.selected_classes = []
        self.class_listbox = tk.Listbox(root, selectmode="multiple", exportselection=False, height=10)
        self.class_listbox.pack(pady=5)

        for name in self.labels.values():
            self.class_listbox.insert(tk.END, name)

        tk.Button(root, text="Update Clasa ", command=self.update_selected_classes).pack(pady=5)
        tk.Button(btn_frame, text=" Imagine", command=self.load_image).pack(side="left", padx=5)
        tk.Button(btn_frame, text=" Video", command=self.load_video).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Live ", command=self.live_detection_popup).pack(side="left", padx=5)

        self.record_var = tk.IntVar()
        tk.Checkbutton(btn_frame, text="Record ", variable=self.record_var).pack(side="left", padx=5)

        self.select_output_button = tk.Button(self.root, text="Select Output ", command=self.select_output_file)
        self.select_output_button.pack(pady=10)

        # adauga sound checkbox
        self.sound_var = tk.IntVar()
        tk.Checkbutton(root, text=" Sound", variable=self.sound_var).pack(pady=5)
        self.sound_enabled = False
        self.sound_cooldown = {}  # cooldown pe clasa

        self.bind_keys()

   # estimare distanta fata de obiect

    def estimate_distance(self, pixel_height, real_height_m=0.6, focal_length_px=700):
        if pixel_height <= 0:
            return None
        return round((focal_length_px * real_height_m) / pixel_height, 2)

    #face update la lista cu clase selectate pt detectie
    def update_selected_classes(self):
        selected_indices = self.class_listbox.curselection()
        self.selected_classes = [int(i) for i in selected_indices]
        print("Clasa selectata id:", self.selected_classes)

    #selecteaza path u unde salveaza detectia
    def select_output_file(self):
        output_file = filedialog.asksaveasfilename(defaultextension=".avi",
                                                   filetypes=[("AVI files", "*.avi"), ("MP4 files", "*.mp4")])
        if output_file:
            self.output_path = output_file
            print(f"Selected output file: {self.output_path}")

    #incarca imaginea pt detectie
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        image = cv2.imread(path)
        result = self.model(image, verbose=False)[0]
        self.display_frame(image, result)

    #incarca fisier video (proceseaza frame by frame)
    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv")])
        if not path:
            return
        self.cap = cv2.VideoCapture(path)
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.running = True
        self.prepare_video_writer()
        self.process_video()

    #gui popup pt detectia live
    def live_detection_popup(self):
        popup = tk.Toplevel(self.root)
        popup.title("Live Detection Settings")
        popup.geometry("300x200")

        source_var = tk.StringVar(value="usb")

        def toggle_inputs():
            state = "normal" if source_var.get() == "droidcam" else "disabled"
            ip_entry.config(state=state)
            port_entry.config(state=state)

        tk.Label(popup, text="Select Sursa").pack(pady=5)
        tk.Radiobutton(popup, text="USB Camera", variable=source_var, value="usb", command=toggle_inputs).pack()
        tk.Radiobutton(popup, text="DroidCam (IP)", variable=source_var, value="droidcam", command=toggle_inputs).pack()

        tk.Label(popup, text="IP Address:").pack()
        ip_entry = tk.Entry(popup)
        ip_entry.insert(0, "192.168.100.144")
        ip_entry.pack()

        tk.Label(popup, text="Port:").pack()
        port_entry = tk.Entry(popup)
        port_entry.insert(0, "4747")
        port_entry.pack()

        #porneste detectia pe live in fucntie de sursa
        def start_live():
            if source_var.get() == "usb":
                self.cap = cv2.VideoCapture(0)
            else:
                ip = ip_entry.get().strip()
                port = port_entry.get().strip()
                stream_url = f"http://{ip}:{port}/video"
                self.cap = cv2.VideoCapture(stream_url)
            self.original_fps = 30
            popup.destroy()
            self.running = True
            self.prepare_video_writer()
            self.process_video()

        tk.Button(popup, text="Start Live ", command=start_live).pack(pady=10)
        toggle_inputs()

    #pt inregistrarea video
    def prepare_video_writer(self):
        self.record = bool(self.record_var.get())
        if not self.cap or not self.cap.isOpened():
            print("Eroare la capura video")
            return

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.resolution = (width, height)

        if self.record:
            if not self.output_path:
                print("Recording e enabled dar nu e selectat Output File.")
                return

            ext = os.path.splitext(self.output_path)[1].lower()
            if ext == '.avi':
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            elif ext == '.mp4':
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            else:
                print("Unsupported file type.")
                return

            self.recorder = cv2.VideoWriter(self.output_path, fourcc, self.original_fps, self.resolution)
            if self.recorder.isOpened():
                print(f"Started recording to: {self.output_path}")
            else:
                print("Eroare la record")

    #proceseaza frame-urile si face detectia
    def process_video(self):
        if not self.running or self.cap is None or not self.cap.isOpened():
            self.cleanup()
            return

        ret, frame = self.cap.read()
        if not ret:
            self.cleanup()
            return
        #calculeaza fps urile
        current_time = time.time()
        instant_fps = 1.0 / (current_time - self.last_time) if self.last_time else 0
        self.fps_list.append(instant_fps)
        if len(self.fps_list) > 10:
            self.fps_list.pop(0)
        self.fps = sum(self.fps_list) / len(self.fps_list) if self.fps_list else 0
        self.last_time = current_time

        #ruleaza modelu pe cadru curent
        results = self.model(frame, verbose=False)[0]
        display_frame = frame.copy()
        self.display_frame(display_frame, results)
        #daca record e activat salveaza cadru
        if self.record and self.recorder:
            self.recorder.write(display_frame)
        #se programeaza urmatoru cadru
        self.root.after(1, self.process_video)
    #face cleanup pe toate resursele
    def cleanup(self):
        if self.cap:
            self.cap.release()
        if self.recorder:
            self.recorder.release()
            if self.record and self.output_path:
                print(f"Recording salvat la: {self.output_path}")
        self.cap = None
        self.recorder = None
        self.running = False
        self.panel.config(image='', text="Detectie terminata")
        self.panel.image = None

    #face sunet pentru clasa detectata
    def play_beep_sound(self, cls):
        def beep():
            print(f"Playing sound pt clasa {cls}")
            winsound.Beep(1000, 500)  # frq 1000Hz, durata 500ms
            self.sound_cooldown[cls] = time.time() + 5  # cooldown de 5 secunde
        threading.Thread(target=beep, daemon=True).start()

    #afisseaza frame u curent cu bodunding box urile si numele clasei
    def display_frame(self, frame, results):
        current_time = time.time()
        self.sound_enabled = bool(self.sound_var.get())
        for det in results.boxes:
            xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
            conf = det.conf.item()
            cls = int(det.cls.item())

            #ignora clasel neselectte
            if self.selected_classes and cls not in self.selected_classes:
                continue

                #treshold
            if conf > 0.5:
                box_height = xyxy[3] - xyxy[1]
                distance_m = self.estimate_distance(box_height)
                label = f"{self.labels[cls]} {int(conf * 100)}% {distance_m}m"
                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # sunet + cooldown
                if self.sound_enabled:
                    if current_time >= self.sound_cooldown.get(cls, 0):
                        self.sound_cooldown[cls] = current_time + 5  # 5 secunde cooldwon
                        self.play_beep_sound(cls)

              
        #afiseaza fps urile
        fps_text = f"FPS: {self.fps:.2f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        
        #face conversia cadrului la rgb si i da display in ui
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.panel.imgtk = imgtk
        self.panel.configure(image=imgtk)

#main 
if __name__ == "__main__":
    model_path = r"C:\Users\ASUS ROG\Desktop\Model For the app\model_for_the_app\my_model.pt"
    if not os.path.exists(model_path):
        print("Model ne gasit!!!.")
    else:
        root = tk.Tk()
        app = YOLOApp(root, model_path)
        root.mainloop()

    #modele
#"C:\Users\ASUS ROG\Desktop\Model For the app\model_for_the_app\my_model.pt"

#"C:\Users\ASUS ROG\Desktop\Model For the app\model_for_the_app\my_model.pt" yolo8L
#"C:\Users\ASUS ROG\Desktop\modelyolo8nano\my_model (3)\my_model.pt" yolo8nano