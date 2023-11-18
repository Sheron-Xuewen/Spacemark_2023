# GUI代码
import tkinter as tk
from tkinter import ttk
import os
from tkinter import filedialog
import config
import subprocess
import sys
import pytesseract


def browse_video():
    config.VIDEO_PATH = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    video_path_label.config(text=f"Selected: {config.VIDEO_PATH}")

def browse_image():
    config.USER_IMAGE_PATH = filedialog.askopenfilename(filetypes=[("JPG files", "*.jpg")])
    image_path_label.config(text=f"Selected: {config.USER_IMAGE_PATH}")

def browse_direction_image():  # New Function
    config.DIRECTION_IMAGE_PATH = filedialog.askopenfilename(filetypes=[("PNG files", "*.PNG")])
    direction_image_path_label.config(text=f"Selected: {config.DIRECTION_IMAGE_PATH}")

def browse_output():
    config.OUTPUT_VIDEO_PATH = filedialog.askdirectory()
    output_path_label.config(text=f"Selected: {config.OUTPUT_VIDEO_PATH}")

def update_direction_input_method(event):  # New Function
    selected_method = direction_input_method_combobox.get()
    if selected_method == "Manual Input":
        user_direction_entry.pack()
        direction_image_path_button.pack_forget()
    elif selected_method == "Upload Image":
        user_direction_entry.pack_forget()
        direction_image_path_button.pack()

def run_main_program():
    config.USER_TEXT = user_text_entry.get()

    # 根据下拉菜单的选择来确定方向输入方式
    user_input_method = direction_input_method_combobox.get()
    if user_input_method == "Manual Input":
        config.USER_DIRECTION_INPUT = user_direction_entry.get()
    elif user_input_method == "Upload Image":
        # 运行图片识别程序并获取输出
        result = subprocess.run(["python", "Image_Recognition.py"], capture_output=True, text=True, encoding='utf-8')
        if result.returncode == 0:
            output = result.stdout.strip()
            direction_info = output.split(":")[1].strip()
            config.USER_DIRECTION_INPUT = direction_info
        else:
            print("Image recognition failed:", result.stderr)
            return

    config.VIDEO_PATH = video_path_label.cget("text").replace("Selected: ", "").strip()
    config.USER_IMAGE_PATH = image_path_label.cget("text").replace("Selected: ", "").strip()
    config.OUTPUT_VIDEO_PATH = output_path_label.cget("text").replace("Selected: ", "").strip()

    full_output_path = os.path.join(config.OUTPUT_VIDEO_PATH, 'output.avi')
    config.OUTPUT_VIDEO_PATH = full_output_path

    config.VIDEO_PATH = config.VIDEO_PATH.replace("/", "\\")
    config.USER_IMAGE_PATH = config.USER_IMAGE_PATH.replace("/", "\\")
    config.OUTPUT_VIDEO_PATH = config.OUTPUT_VIDEO_PATH.replace("/", "\\")

    print(f"User Text: {config.USER_TEXT}")
    print(f"User Direction: {config.USER_DIRECTION_INPUT}")
    print(f"Video Path: {config.VIDEO_PATH}")
    print(f"Image Path: {config.USER_IMAGE_PATH}")
    print(f"Output Path: {config.OUTPUT_VIDEO_PATH}")

    if not os.path.exists(config.VIDEO_PATH):
        print(f"Video file does not exist: {config.VIDEO_PATH}")
        return
    if not os.path.exists(config.USER_IMAGE_PATH):
        print(f"Image file does not exist: {config.USER_IMAGE_PATH}")
        return
    output_directory = os.path.dirname(config.OUTPUT_VIDEO_PATH)
    if not os.path.exists(output_directory):
        print(f"Output directory does not exist: {output_directory}")
        return

    if all([config.VIDEO_PATH, config.USER_IMAGE_PATH, config.OUTPUT_VIDEO_PATH, config.USER_TEXT, config.USER_DIRECTION_INPUT]):
        config.write_config()
        result = subprocess.run([sys.executable, "E:\\AR_Wayfinding_Project\\src - 副本\\GUI\\AR_Wayfinding_Main.py"], capture_output=True, text=True, encoding='utf-8')
        print("Standard Output:", result.stdout)
        print("Standard Error:", result.stderr)
    else:
        print("All fields must be filled out before running the main program.")
        return



root = tk.Tk()
root.title("AR Wayfinding")

user_text_label = tk.Label(root, text="Enter User Text:")
user_text_label.pack()
user_text_entry = tk.Entry(root)
user_text_entry.pack()

direction_input_method_label = tk.Label(root, text="Direction Input Method:")  # New Widget
direction_input_method_label.pack()  # New Widget
direction_input_method_combobox = ttk.Combobox(root, values=["Manual Input", "Upload Image"])  # New Widget
direction_input_method_combobox.pack()  # New Widget
direction_input_method_combobox.current(0)  # New Widget
direction_input_method_combobox.bind("<<ComboboxSelected>>", update_direction_input_method)  # New Widget

user_direction_label = tk.Label(root, text="Enter User Direction:")
user_direction_label.pack()
user_direction_entry = tk.Entry(root)
user_direction_entry.pack()

direction_image_path_label = tk.Label(root, text="Select Direction Image Path:")  # New Widget
direction_image_path_label.pack()  # New Widget
direction_image_path_button = tk.Button(root, text="Browse", command=browse_direction_image)  # New Widget
direction_image_path_button.pack_forget()  # Initially hidden  # New Widget

image_path_label = tk.Label(root, text="Select User Image Path:")
image_path_label.pack()
image_path_button = tk.Button(root, text="Browse", command=browse_image)
image_path_button.pack()

video_path_label = tk.Label(root, text="Select Video Path:")
video_path_label.pack()
video_path_button = tk.Button(root, text="Browse", command=browse_video)
video_path_button.pack()

output_path_label = tk.Label(root, text="Select Output Video Path:")
output_path_label.pack()
output_path_button = tk.Button(root, text="Browse", command=browse_output)
output_path_button.pack()

run_button = tk.Button(root, text="Run Program", command=run_main_program)
run_button.pack()

root.mainloop()