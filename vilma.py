import cv2
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import numpy as np
import mss
import threading
import os
import time
from datetime import datetime
import atexit
import tkinter as tk
from tkinter import filedialog
import platform
import subprocess
from colorama import init, Fore, Style
import ctypes
import pyautogui

# Initialize colorama (for colored terminal output)
init()

class ViLMA:
    """
    A class to monitor the desktop screen (or a specific window) and perform binary
    inference on the captured images using a pre-trained vision-language model.
    It then reacts (log out, screenshot, etc.) based on the model's inferred text.
    """

    def __init__(self):
        """
        Initialize the ViLMA object with default settings, but do not load any model.
        """
        # Decide which device (CPU/GPU) to use for inference
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model and processor references (initialized to None until load_model())
        self.model = None
        self.processor = None

        # A list of prompts to use during screen monitoring. Each prompt is a string.
        self.prompts = []

        # Booleans controlling various functionalities
        self.blank_window_open = False     # Whether the "blank screen" is active
        self.logout_on_trigger = False     # If True, log out upon "yes" inference
        self.dummy_mode = False            # If True, do not perform certain real actions
        self.blank_screen_on_trigger = False
        self.screenshot_on_trigger = False
        self.record_on_trigger = False

        # For custom trigger logic (open a file/app when a certain text is detected)
        self.custom_trigger_path = None
        self.custom_trigger_enabled = False
        self.custom_trigger_output = "yes"

        # Thread/Event-based recording variables
        #   _recording_event is an Event: if "set", we stop recording; if "clear", we keep recording
        self._recording_event = threading.Event()
        self.current_video_filename = None
        self.video_writer = None

        # Rate of inference (images per second), can be None to run freely
        self.inference_rate = None

        # Resolution setting for image processing/resizing: "640p", "720p", "1080p", "native"
        self.resolution = "720p"

        # Keyboard trigger logic (type a sequence if a certain text is detected)
        self.keyboard_trigger_enabled = False
        self.keyboard_trigger_sequence = ""
        self.keyboard_trigger_activated = False
        self.keyboard_trigger_output = "yes"

        # If set, only capture from a single window with a known title
        self.target_window = None

        # Ensure we close the blank screen if the program exits unexpectedly
        atexit.register(self.ensure_blank_window_closed)

    @property
    def is_recording(self):
        """
        Property that returns True if the recording thread is active.
        We interpret "active" as _recording_event NOT being set.
        - Event is "cleared" => record_desktop() loops.
        - Event is "set" => record_desktop() thread is signaled to stop.
        """
        return not self._recording_event.is_set()

    def load_model(self, model_path):
        """
        Loads the model and processor from a given model directory.
        - model_path (str): path to the HuggingFace model directory on disk.
        """
        try:
            # Load the Causal LM model in eval mode
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True
            ).eval().to(self.device)

            # If we have a GPU available, convert the model to half-precision for speed
            if self.device.type == "cuda":
                self.model.half()

            # Load the corresponding processor (tokenizer + image processor)
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

            print(Fore.GREEN + "Model loaded successfully." + Style.RESET_ALL)
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

    def prepare_inputs(self, task_prompt, image):
        """
        Prepares the text and image for the model using the processor.
        - task_prompt (str): text prompt
        - image (PIL.Image): screenshot or window capture
        Returns a dict of tensors ready for inference.
        """
        try:
            # Processor will handle tokenization and any image transformation
            inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(self.device)

            # On CUDA, optionally cast float tensors to half precision
            if self.device.type == "cuda":
                for k, v in inputs.items():
                    if torch.is_floating_point(v):
                        inputs[k] = v.half()
            return inputs
        except Exception as e:
            raise RuntimeError(f"Error preparing inputs: {e}")

    def run_model(self, inputs):
        """
        Run the model forward pass (generate) on the prepared inputs.
        - inputs: output of prepare_inputs()
        Returns the generated token ids from the model.
        """
        try:
            # Use automatic mixed precision on GPU
            with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs.get("pixel_values"),
                    max_new_tokens=1024,
                    early_stopping=False,
                    do_sample=False,
                    num_beams=1,
                )
            return generated_ids
        except Exception as e:
            raise RuntimeError(f"Error running model: {e}")

    def process_outputs(self, generated_ids):
        """
        Decodes the model's output token IDs back into text.
        - generated_ids: the output from run_model()
        Returns a string.
        """
        try:
            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            return generated_text
        except Exception as e:
            raise RuntimeError(f"Error processing outputs: {e}")

    def run_inference(self, image, prompt):
        """
        Full pipeline of: prepare -> run -> decode on a single image+prompt pair.
        Prints debug info about raw and cleaned inference text.
        Returns the cleaned text or 'Error' if something fails.
        """
        try:
            # Prepare inputs from the user-defined prompt plus the screenshot
            inputs = self.prepare_inputs(prompt, image)
            # Generate token IDs from the model
            generated_ids = self.run_model(inputs)
            # Decode back into text
            generated_text = self.process_outputs(generated_ids)

            # Clean up known artifacts or tokens
            cleaned_text = generated_text.replace("</s><s>", "").strip()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            print(f"{timestamp} - Raw Inference result: {generated_text}")
            print(f"{timestamp} - Cleaned Inference result: {cleaned_text}")

            # Check if the custom trigger is enabled (e.g., open a file on certain output)
            if self.custom_trigger_enabled and self.custom_trigger_path:
                print(f"{timestamp} - Checking custom trigger: '{self.custom_trigger_output.lower()}' in '{cleaned_text.lower()}'")
                if self.custom_trigger_output.lower() in cleaned_text.lower():
                    print(f"{timestamp} - Custom trigger matched. Running custom trigger.")
                    self.run_custom_trigger()
                    # After the first match, disable custom trigger automatically
                    self.custom_trigger_enabled = False
                else:
                    print(f"{timestamp} - Custom trigger did not match.")

            # Check if the "keyboard trigger" is enabled
            #   If it sees the designated keyword in text and hasn't yet triggered, do it
            if self.keyboard_trigger_enabled and (self.keyboard_trigger_output.lower() in cleaned_text.lower()) and not self.keyboard_trigger_activated:
                print(f"{timestamp} - Keyboard trigger matched. Running keyboard trigger.")
                self.run_keyboard_trigger()
                self.keyboard_trigger_activated = True

            # If the text no longer contains the target, reset the activation
            if self.keyboard_trigger_output.lower() not in cleaned_text.lower():
                self.keyboard_trigger_activated = False

            return cleaned_text
        except Exception as e:
            print(f"Error during inference: {e}")
            return "Error"

    def capture_desktop(self):
        """
        Captures either the full desktop or a specific window, based on self.target_window.
        Uses mss to grab the screen contents, returns a PIL.Image.
        """
        try:
            with mss.mss() as sct:
                if self.target_window:
                    # If a target window is specified, find a window matching that title via PyAutoGUI
                    window = pyautogui.getWindowsWithTitle(self.target_window)
                    if window:
                        window = window[0]
                        left, top, width, height = window.left, window.top, window.width, window.height
                        monitor = {"top": top, "left": left, "width": width, "height": height}
                    else:
                        print(Fore.YELLOW + f"Window '{self.target_window}' not found. Capturing full desktop." + Style.RESET_ALL)
                        monitor = sct.monitors[1]
                else:
                    # Otherwise capture the primary monitor
                    monitor = sct.monitors[1]

                screenshot = sct.grab(monitor)
                # Convert raw BGRA data from MSS into a PIL image
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            return img
        except Exception as e:
            raise RuntimeError(f"Error capturing desktop: {e}")

    def take_screenshot(self):
        """
        Takes a screenshot (same logic as capture_desktop) and saves it to disk with a timestamped filename.
        """
        try:
            screenshot = self.capture_desktop()
            filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            screenshot.save(filename)
            print(Fore.GREEN + f"Screenshot taken: {filename}" + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"Error taking screenshot: {e}" + Style.RESET_ALL)

    def start_recording(self):
        """
        Starts recording the screen (desktop or window) in a separate thread, if not already running.
        - Clears the _recording_event so the recording thread will loop.
        - Creates a VideoWriter to store frames in an .mp4 file.
        - Prints the filename/directory for user reference.
        """
        try:
            # Ensure the event is cleared so the recording thread runs
            self._recording_event.clear()

            # Prepare VideoWriter with mp4v codec at ~20 FPS
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            width, height = self.get_resolution_dimensions()
            self.current_video_filename = f'recording_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'

            self.video_writer = cv2.VideoWriter(self.current_video_filename, fourcc, 20.0, (width, height))

            print(Fore.GREEN + f"Recording started. Output file: {self.current_video_filename}" + Style.RESET_ALL)
            print(Fore.GREEN + f"Stored in directory: {os.getcwd()}" + Style.RESET_ALL)

            # Launch the dedicated recording thread
            record_thread = threading.Thread(target=self.record_desktop, daemon=True)
            record_thread.start()
        except Exception as e:
            print(Fore.RED + f"Error starting recording: {e}" + Style.RESET_ALL)

    def record_desktop(self):
        """
        Continuously captures frames while _recording_event is NOT set.
        Writes frames to the self.video_writer.
        """
        try:
            with mss.mss() as sct:
                while not self._recording_event.is_set():
                    try:
                        # Attempt to capture from desktop or specific window
                        screenshot = self.capture_desktop()
                    except Exception as capture_err:
                        print(Fore.RED + f"Error capturing desktop while recording: {capture_err}" + Style.RESET_ALL)
                        continue

                    # Convert from PIL to numpy (RGB -> BGR for OpenCV)
                    frame = np.array(screenshot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # Write frames out if the writer is still open
                    if self.video_writer:
                        self.video_writer.write(frame)

                    # Limit framerate to ~20 FPS
                    time.sleep(1/20)
        except Exception as e:
            print(Fore.RED + f"Error recording desktop: {e}" + Style.RESET_ALL)
        finally:
            # Ensure release of the video writer resource
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            print(Fore.GREEN + "Recording thread has exited." + Style.RESET_ALL)

    def stop_recording(self):
        """
        Signals the recording thread to stop if it is currently running.
        - Setting _recording_event means the record_desktop() loop will exit.
        """
        try:
            if self.is_recording:
                self._recording_event.set()
                print(Fore.GREEN + f"Recording stop requested. Final file: {self.current_video_filename}" + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"Error stopping recording: {e}" + Style.RESET_ALL)

    def run_custom_trigger(self):
        """
        If a custom trigger is enabled and the inference matches it, we run an OS-specific command
        to open the associated file/path (like open a script, doc, or .exe).
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{timestamp} - Running custom trigger: {self.custom_trigger_path}")

            # Use different commands depending on Windows/Darwin/Linux
            if platform.system() == "Windows":
                os.startfile(self.custom_trigger_path)
            elif platform.system() == "Darwin":
                subprocess.run(["open", self.custom_trigger_path], check=True)
            else:
                subprocess.run(["xdg-open", self.custom_trigger_path], check=True)

            print(f"{timestamp} - Custom trigger executed successfully.")
        except Exception as e:
            print(f"{timestamp} - Error running custom trigger: {e}")

    def run_keyboard_trigger(self):
        """
        If a keyboard trigger is detected, we simulate typing using PyAutoGUI.
        """
        try:
            pyautogui.typewrite(self.keyboard_trigger_sequence)
            print(Fore.GREEN + f"Executed keyboard sequence: {self.keyboard_trigger_sequence}" + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"Error executing keyboard trigger: {e}" + Style.RESET_ALL)

    def show_blank_window(self):
        """
        Creates a fullscreen black window (via OpenCV) to obscure the screen.
        User can press 'q' to close it manually, or we can close it programmatically.
        """
        try:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Showing blank window")

            self.blank_window_open = True

            # Create an all-black image. Dimensions are somewhat arbitrary,
            # but 1920x1080 ensures full coverage on typical screens.
            blank_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)

            cv2.namedWindow("Blank Screen", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Blank Screen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            # On Windows, force the window to front/topmost
            if platform.system() == "Windows":
                hwnd = ctypes.windll.user32.FindWindowW(None, "Blank Screen")
                if hwnd:
                    ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)

            # Continue showing the window until user presses 'q' or blank_window_open becomes False
            while self.blank_window_open:
                cv2.imshow("Blank Screen", blank_screen)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    self.blank_window_open = False
                    break

            # Once done, destroy the window
            cv2.destroyWindow("Blank Screen")
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Closed blank window")
        except Exception as e:
            print(f"Error showing blank window: {e}")

    def ensure_blank_window_closed(self):
        """
        Safely close the blank window if it’s still open. Called on program exit.
        """
        try:
            self.blank_window_open = False
            cv2.destroyWindow("Blank Screen")
        except:
            pass

    def logout(self):
        """
        Logs the user out of the system on Windows, macOS, or Linux.
        If the OS is unsupported, we print a warning.
        """
        try:
            system_platform = platform.system()
            if system_platform == "Windows":
                subprocess.run(["shutdown", "/l"], check=True)
            elif system_platform in ("Linux", "Darwin"):
                subprocess.run(["pkill", "-KILL", "-u", os.getlogin()], check=True)
            else:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Unsupported operating system: {system_platform}")
        except Exception as e:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{timestamp} - Error logging out: {e}")

    def start_monitoring(self):
        """
        Main loop to capture the screen, run inference, and perform actions based on "yes"/"no" triggers.
        Continues until user presses 'q' or an error occurs.
        """
        if self.model is None:
            print(Fore.RED + "Error: No model loaded. Please load a model before starting monitoring." + Style.RESET_ALL)
            return

        if not self.prompts:
            print(Fore.RED + "Error: No inference prompts set. Please add at least one inference prompt before starting monitoring." + Style.RESET_ALL)
            return

        try:
            while True:
                start_time = time.time()
                try:
                    screen = self.capture_desktop()
                except RuntimeError as cap_err:
                    # If capturing fails, wait briefly and retry
                    print(Fore.RED + f"Error capturing desktop: {cap_err}" + Style.RESET_ALL)
                    time.sleep(1)
                    continue

                # Convert the screenshot to RGB -> numpy -> resized -> back to PIL
                screen_rgb = screen.convert("RGB")
                screen_np = np.array(screen_rgb)
                width, height = self.get_resolution_dimensions()
                screen_resized = cv2.resize(screen_np, (width, height))
                pil_image = Image.fromarray(screen_resized)

                # Run each prompt in the prompts list
                for prompt in self.prompts:
                    result = self.run_inference(pil_image, prompt)

                    # If we get "yes" and not in dummy mode
                    if result.lower() == "yes" and not self.dummy_mode:
                        # Optionally log out
                        if self.logout_on_trigger:
                            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Trigger detected, logging out")
                            self.logout()
                            break  # Immediately break from the for-loop

                        # Optionally show blank window
                        if self.blank_screen_on_trigger and not self.blank_window_open:
                            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Trigger detected, opening blank window")
                            self.show_blank_window()

                        # Optionally take a screenshot
                        if self.screenshot_on_trigger:
                            self.take_screenshot()

                        # If record_on_trigger is on, start recording if not already
                        if self.record_on_trigger and not self.is_recording:
                            self.start_recording()

                    # If we get "no", we close the blank window and stop recording if they are active
                    elif result.lower() == "no":
                        if self.blank_window_open:
                            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - No trigger detected, closing blank window")
                            self.ensure_blank_window_closed()

                        if self.record_on_trigger and self.is_recording:
                            self.stop_recording()

                # Respect the inference_rate setting if it’s not None
                elapsed_time = time.time() - start_time
                if self.inference_rate:
                    time_to_wait = max(1.0 / self.inference_rate - elapsed_time, 0)
                    time.sleep(time_to_wait)

                # Check if user pressed 'q' in any open CV window
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except Exception as e:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{timestamp} - Error during monitoring: {e}")
        finally:
            # Make sure we close the blank window if open
            self.ensure_blank_window_closed()

            # Also stop recording if currently active
            if self.is_recording:
                self.stop_recording()

    def get_resolution_dimensions(self):
        """
        Returns (width, height) based on:
          - self.target_window size (if specified & found)
          - or the chosen resolution string (640p, 720p, 1080p, native)
          - if 'native', we get actual monitor size via mss.
        """
        # If a specific window is targeted, try to get that window's geometry
        if self.target_window:
            window = pyautogui.getWindowsWithTitle(self.target_window)
            if window:
                window = window[0]
                return window.width, window.height

        # Otherwise, pick a resolution from our set of known strings
        if self.resolution == "640p":
            return 640, 360
        elif self.resolution == "720p":
            return 1280, 720
        elif self.resolution == "1080p":
            return 1920, 1080
        elif self.resolution == "native":
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                return monitor["width"], monitor["height"]
        else:
            # Default fallback = 720p
            return 1280, 720

    def toggle_resolution(self):
        """
        Cycles through a list of supported resolutions:
        ["640p", "720p", "1080p", "native"]
        and sets self.resolution to the next one in the list.
        Prints the new resolution to the user.
        """
        resolutions = ["640p", "720p", "1080p", "native"]
        current_index = resolutions.index(self.resolution)
        new_index = (current_index + 1) % len(resolutions)
        self.resolution = resolutions[new_index]
        print(Fore.GREEN + f"Resolution set to {self.resolution}." + Style.RESET_ALL)

    def set_target_window(self):
        """
        Sets the target window by enumerating open windows (via pyautogui.getAllWindows())
        and letting the user pick from a numbered list. If user leaves it blank,
        we capture the full desktop. This avoids having to type the exact window title.
        """
        all_windows = pyautogui.getAllWindows()

        # Filter out windows with empty or whitespace-only titles
        titled_windows = []
        for idx, w in enumerate(all_windows):
            title = w.title.strip()
            if title:
                titled_windows.append((idx, title))

        if not titled_windows:
            print("No titled windows found. Defaulting to full desktop capture.")
            self.target_window = None
            return

        print(Fore.CYAN + "Open Windows:" + Style.RESET_ALL)
        for local_index, (original_idx, title) in enumerate(titled_windows):
            print(f"[{local_index}] {title}")

        selection = input("\nEnter the index of the window you want to monitor (leave blank for full desktop): ")

        if selection.strip() == "":
            print("No selection, capturing full desktop.")
            self.target_window = None
            return

        if not selection.isdigit():
            print("Invalid input. Defaulting to full desktop.")
            self.target_window = None
            return

        selection_index = int(selection)
        if 0 <= selection_index < len(titled_windows):
            chosen_title = titled_windows[selection_index][1]
            self.target_window = chosen_title
            print(Fore.GREEN + f"Target window set to: {self.target_window}" + Style.RESET_ALL)
        else:
            print("Invalid index. Capturing full desktop.")
            self.target_window = None

        def terminal_menu(self):
            """
            Displays the terminal menu for user interaction.
            """
            print(Fore.CYAN + "\n=== Welcome to ViLMA (Vision-Language Model-based Active Monitoring) ===" + Style.RESET_ALL)
            while True:
                print(Fore.CYAN + "\n=== Menu ===" + Style.RESET_ALL)

                print(Fore.LIGHTGREEN_EX + "1. Start Screen Monitoring" + Style.RESET_ALL)

                print(Fore.MAGENTA + "\nModel Operations:" + Style.RESET_ALL)
                print(Fore.LIGHTMAGENTA_EX + "2. Load Florence-2" + Style.RESET_ALL)

                print(Fore.BLUE + "\nMonitoring Settings:" + Style.RESET_ALL)
                print(Fore.LIGHTBLUE_EX + "3. Add Inference Prompt" + Style.RESET_ALL)
                print(Fore.LIGHTBLUE_EX + "4. Remove Inference Prompt" + Style.RESET_ALL)
                print(Fore.LIGHTBLUE_EX + "5. List Inference Prompts" + Style.RESET_ALL)
                print(Fore.LIGHTBLUE_EX + "6. Set Inference Rate (current: " +
                      (Fore.GREEN + str(self.inference_rate) if self.inference_rate else Fore.RED + "None") +
                      Style.RESET_ALL + ")" + Style.RESET_ALL)
                print(Fore.LIGHTBLUE_EX + "7. Toggle Processing Resolution (current: " +
                      Fore.GREEN + self.resolution + Style.RESET_ALL + ")" + Style.RESET_ALL)
                print(Fore.LIGHTBLUE_EX + "8. Set Target Window (current: " +
                      (Fore.GREEN + self.target_window if self.target_window else Fore.RED + "Full Desktop") +
                      Style.RESET_ALL + ")" + Style.RESET_ALL)

                # CHANGED SECTION
                print(Fore.GREEN + "\nToggles & Triggers:" + Style.RESET_ALL)
                print(Fore.LIGHTGREEN_EX + "9.  Logout (current: " +
                      (Fore.GREEN + "ON" if self.logout_on_trigger else Fore.RED + "OFF") +
                      Style.RESET_ALL + ")" + Style.RESET_ALL)
                print(Fore.LIGHTGREEN_EX + "10. Dummy Mode (current: " +
                      (Fore.GREEN + "ON" if self.dummy_mode else Fore.RED + "OFF") +
                      Style.RESET_ALL + ")" + Style.RESET_ALL)
                print(Fore.LIGHTGREEN_EX + "11. Blank Screen (current: " +
                      (Fore.GREEN + "ON" if self.blank_screen_on_trigger else Fore.RED + "OFF") +
                      Style.RESET_ALL + ")" + Style.RESET_ALL)
                print(Fore.LIGHTGREEN_EX + "12. Screenshot (current: " +
                      (Fore.GREEN + "ON" if self.screenshot_on_trigger else Fore.RED + "OFF") +
                      Style.RESET_ALL + ")" + Style.RESET_ALL)
                print(Fore.LIGHTGREEN_EX + "13. Record (current: " +
                      (Fore.GREEN + "ON" if self.record_on_trigger else Fore.RED + "OFF") +
                      Style.RESET_ALL + ")" + Style.RESET_ALL)
                print(Fore.LIGHTGREEN_EX + "14. Custom (current: " +
                      (Fore.GREEN + "ON" if self.custom_trigger_enabled else Fore.RED + "OFF") +
                      Style.RESET_ALL + ")" + Style.RESET_ALL)
                print(Fore.LIGHTGREEN_EX + "15. Keyboard Command (current: " +
                      (Fore.GREEN + "ON" if self.keyboard_trigger_enabled else Fore.RED + "OFF") +
                      Style.RESET_ALL + ")" + Style.RESET_ALL)

                print(Fore.YELLOW + "\nGeneral:" + Style.RESET_ALL)
                print(Fore.LIGHTYELLOW_EX + "16. Quit" + Style.RESET_ALL)

                print(Fore.CYAN + "\n==========================" + Style.RESET_ALL)
                choice = input("Enter your choice: ")

            try:
                if choice == "1":
                    print(Fore.CYAN + "Starting screen monitoring..." + Style.RESET_ALL)
                    self.start_monitoring()

                elif choice == "2":
                    # Prompt to load the model
                    self.load_model_menu()

                elif choice == "3":
                    # Add an inference prompt
                    prompt = input("Enter the inference prompt to add: ")
                    self.prompts.append(prompt)
                    print(Fore.GREEN + f"Added inference prompt: {prompt}" + Style.RESET_ALL)

                elif choice == "4":
                    # Remove an inference prompt
                    self.list_prompts()
                    index = int(input("Enter the prompt number to remove: ")) - 1
                    if 0 <= index < len(self.prompts):
                        removed_prompt = self.prompts.pop(index)
                        print(Fore.GREEN + f"Removed inference prompt: {removed_prompt}" + Style.RESET_ALL)
                    else:
                        print(Fore.RED + "Invalid prompt number." + Style.RESET_ALL)

                elif choice == "5":
                    # List inference prompts
                    self.list_prompts()

                elif choice == "6":
                    # Set how many frames/second we want to process
                    self.set_inference_rate()

                elif choice == "7":
                    # Toggle resolution among 640p,720p,1080p,native
                    self.toggle_resolution()

                elif choice == "8":
                    # Enumerate open windows and pick one by index
                    self.set_target_window()

                elif choice == "9":
                    # Toggle whether we log out on "yes"
                    self.logout_on_trigger = not self.logout_on_trigger
                    print(Fore.GREEN + f"Logout on Trigger is now {'ON' if self.logout_on_trigger else 'OFF'}" + Style.RESET_ALL)

                elif choice == "10":
                    # Toggle dummy mode (suppress certain real actions)
                    self.dummy_mode = not self.dummy_mode
                    print(Fore.GREEN + f"Dummy mode is now {'ON' if self.dummy_mode else 'OFF'}" + Style.RESET_ALL)

                elif choice == "11":
                    # Toggle blank screen overlay on "yes"
                    self.blank_screen_on_trigger = not self.blank_screen_on_trigger
                    print(Fore.GREEN + f"Blank Screen on Trigger is now {'ON' if self.blank_screen_on_trigger else 'OFF'}" + Style.RESET_ALL)

                elif choice == "12":
                    # Toggle taking a screenshot on "yes"
                    self.screenshot_on_trigger = not self.screenshot_on_trigger
                    print(Fore.GREEN + f"Screenshot on Trigger is now {'ON' if self.screenshot_on_trigger else 'OFF'}" + Style.RESET_ALL)

                elif choice == "13":
                    # Toggle screen recording on "yes"
                    self.record_on_trigger = not self.record_on_trigger
                    print(Fore.GREEN + f"Record on Trigger is now {'ON' if self.record_on_trigger else 'OFF'}" + Style.RESET_ALL)

                elif choice == "14":
                    # Toggle custom trigger
                    if not self.custom_trigger_enabled:
                        self.custom_trigger_path = filedialog.askopenfilename(title="Select File to Open on Trigger")
                        if self.custom_trigger_path:
                            self.custom_trigger_output = input("Enter the output that triggers the custom action (e.g., 'yes', 'no', 'open file', etc.): ")
                            print(f"Setting custom_trigger_output to: {self.custom_trigger_output}")
                            self.custom_trigger_enabled = True
                            print(Fore.GREEN + f"Custom Trigger set to open {self.custom_trigger_path} on output: {self.custom_trigger_output}" + Style.RESET_ALL)
                        else:
                            print(Fore.RED + "Custom Trigger path selection cancelled." + Style.RESET_ALL)
                    else:
                        self.custom_trigger_enabled = False
                        self.custom_trigger_path = None
                        self.custom_trigger_output = "yes"
                        print(Fore.GREEN + "Custom Trigger is now OFF" + Style.RESET_ALL)

                elif choice == "15":
                    # Toggle keyboard trigger
                    if not self.keyboard_trigger_enabled:
                        self.keyboard_trigger_sequence = input("Enter the keyboard sequence to type on trigger: ")
                        self.keyboard_trigger_output = input("Enter the output that triggers the keyboard action (e.g., 'yes', 'no', 'low health', etc.): ")
                        self.keyboard_trigger_enabled = True
                        self.keyboard_trigger_activated = False
                        print(Fore.GREEN + f"Keyboard Trigger set to type: {self.keyboard_trigger_sequence} on output: {self.keyboard_trigger_output}" + Style.RESET_ALL)
                    else:
                        self.keyboard_trigger_enabled = False
                        self.keyboard_trigger_sequence = ""
                        self.keyboard_trigger_output = "yes"
                        print(Fore.GREEN + "Keyboard Trigger is now OFF" + Style.RESET_ALL)

                elif choice == "16":
                    # Quit the program
                    print(Fore.CYAN + "Quitting..." + Style.RESET_ALL)
                    break

                else:
                    # Catch-all for invalid menu choices
                    print(Fore.RED + "Invalid choice. Please try again." + Style.RESET_ALL)

            except Exception as e:
                # Catch any errors that occur during menu handling
                print(Fore.RED + f"Error: {e}" + Style.RESET_ALL)

    def load_model_menu(self):
        """
        Helper method to open a file dialog so the user can pick the model directory.
        Then calls load_model with that path.
        """
        model_path = select_model_path()
        if not model_path:
            print(Fore.RED + "Model path selection cancelled." + Style.RESET_ALL)
        else:
            self.load_model(model_path)

    def set_inference_rate(self):
        """
        Prompts the user for an inference rate (1-25) or 'None', then sets self.inference_rate.
        """
        try:
            rate = input("Enter the desired inference rate (1-25, or None for default): ")
            if rate.lower() == 'none':
                self.inference_rate = None
                print(Fore.GREEN + "Inference rate set to default (None)." + Style.RESET_ALL)
            else:
                rate = int(rate)
                if 1 <= rate <= 25:
                    self.inference_rate = rate
                    print(Fore.GREEN + f"Inference rate set to {rate} IPS." + Style.RESET_ALL)
                else:
                    print(Fore.RED + "Invalid rate. Please enter a number between 1 and 25." + Style.RESET_ALL)
        except ValueError:
            print(Fore.RED + "Invalid input. Please enter a number between 1 and 25, or 'None'." + Style.RESET_ALL)

    def list_prompts(self):
        """
        Prints out the current inference prompts in self.prompts, enumerated by index.
        """
        print(Fore.CYAN + "\nCurrent inference prompts:" + Style.RESET_ALL)
        for i, prompt in enumerate(self.prompts, 1):
            print(Fore.GREEN + f"{i}. {prompt}" + Style.RESET_ALL)


def select_model_path():
    """
    Utility function to open a dialog for selecting a model directory (using tkinter).
    Returns the chosen directory path as a string, or empty if cancelled.
    """
    root = tk.Tk()
    root.withdraw()
    model_path = filedialog.askdirectory(title="Select Model Directory")
    root.destroy()
    return model_path

if __name__ == "__main__":
    try:
        # Create an instance of ViLMA and display the menu
        vilma = ViLMA()
        print(Fore.CYAN + "Starting ViLMA (Vision-Language Model-based Active Monitoring)." + Style.RESET_ALL)
        vilma.terminal_menu()
    except Exception as e:
        # If something fails on initialization, log it here
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(Fore.RED + f"{timestamp} - Error initializing ViLMA: {e}" + Style.RESET_ALL)
