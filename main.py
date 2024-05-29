import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import glob
import queue
import pydicom
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import ReXNNPS

def load_dicom_directory():
    global path
    directory = filedialog.askdirectory()
    if directory:
        path = directory
        global files, current_index
        files = glob.glob(os.path.join(directory, '*.dcm'))
        if files:
            current_index = 0
            load_dicom_image(files[current_index])
            update_buttons()
            log_message("DICOM files loaded successfully from: " + path)
        else:
            messagebox.showerror("Error", "No DICOM files found in the directory.")

def load_dicom_image(file_path):
    global canvas_widget
    dicom_data = pydicom.dcmread(file_path)
    fig, ax = plt.subplots()
    ax.imshow(dicom_data.pixel_array, cmap='gray')
    if canvas_widget is not None:
        canvas_widget.destroy()
    canvas = FigureCanvasTkAgg(fig, master=image_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    update_buttons()

def next_image():
    global current_index
    if current_index < len(files) - 1:
        current_index += 1
        load_dicom_image(files[current_index])
        log_message(f"Showing image {current_index + 1} of {len(files)}")

def previous_image():
    global current_index
    if current_index > 0:
        current_index -= 1
        load_dicom_image(files[current_index])
        log_message(f"Showing image {current_index + 1} of {len(files)}")

def update_buttons():
    prev_button.config(state=tk.NORMAL if current_index > 0 else tk.DISABLED)
    next_button.config(state=tk.NORMAL if current_index < len(files) - 1 else tk.DISABLED)

def log_message(message):
    log.config(state=tk.NORMAL)
    log.insert(tk.END, message + "\n")
    log.config(state=tk.DISABLED)
    log.yview(tk.END)

def execute_calculation():
    global path, conversion, export_format, a_value, b_value
    a = a_value.get()
    b = b_value.get()
    if not all([path, a, b]):
        messagebox.showerror("Error", "Please ensure all fields are filled and a directory is selected.")
        return
    try:
        a = float(a)
        b = float(b)
        threading.Thread(target=lambda: ReXNNPS.calculateNNPS(path, conversion.get(), a, b, export_format.get(), update_progress, log_queue), daemon=True).start()
    except ValueError:
        messagebox.showerror("Error", "Invalid numerical input for 'a' or 'b'.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def update_progress(percentage):
    progress['value'] = percentage
    root.update_idletasks()

def update_log_from_queue():
    while not log_queue.empty():
        message = log_queue.get()
        log_message(message)
    root.after(100, update_log_from_queue)

root = tk.Tk()
root.title("ReX - Noise Power Spectrum Analysis")
root.iconbitmap('dinosauricon.ico')
root.geometry('900x800')

# Crear una instancia de Queue
log_queue = queue.Queue()

# Variables globales para almacenar configuraciones
path = ''
conversion = tk.StringVar(value='linear')  # Opciones de conversión
export_format = tk.StringVar(value='excel')  # Opciones de formato de exportación

# Frames y widgets
image_frame = tk.Frame(root, width=600, height=450)
image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
canvas_widget = None

navigation_frame = tk.Frame(image_frame)
navigation_frame.pack(side=tk.BOTTOM, fill=tk.X)

prev_button = tk.Button(navigation_frame, text="Previous", command=previous_image)
prev_button.pack(side=tk.LEFT, padx=10)

next_button = tk.Button(navigation_frame, text="Next", command=next_image)
next_button.pack(side=tk.RIGHT, padx=10)

status_frame = tk.Frame(root, height=150)
status_frame.pack(side=tk.BOTTOM, fill=tk.X)

log = tk.Text(status_frame, state=tk.DISABLED, height=5)
log.pack(side=tk.TOP, fill=tk.X, expand=True)

progress = ttk.Progressbar(status_frame, orient="horizontal", mode="determinate")
progress.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=10)

load_button = tk.Button(navigation_frame, text="Select folder", command=load_dicom_directory)
load_button.pack(side=tk.TOP, pady=10)

a_value = tk.Entry(navigation_frame, width=10)
a_value.pack(side=tk.LEFT, padx=10)
b_value = tk.Entry(navigation_frame, width=10)
b_value.pack(side=tk.LEFT, padx=10)

calculate_button = tk.Button(navigation_frame, text="Calculate NPS/NNPS", command=execute_calculation)
calculate_button.pack(side=tk.LEFT, padx=20)

conversion_menu = tk.OptionMenu(root, conversion, 'linear', 'log')
conversion_menu.pack(side=tk.LEFT, padx=10)

format_menu = tk.OptionMenu(root, export_format, 'csv', 'excel')
format_menu.pack(side=tk.RIGHT, padx=10)

root.after(100, update_log_from_queue)  # Iniciar la revisión de la cola
root.mainloop()
