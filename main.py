#!/usr/bin/env python3

import os
import sys
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'
warnings.filterwarnings("ignore", category=FutureWarning) 

if os.name == 'nt':
    os.system('')

class Color:
    ORANGE = '\033[38;5;208m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    WHITE = '\033[97m'
    RESET = '\033[0m'

print(f"{Color.ORANGE}[Start]{Color.RESET} {Color.WHITE}Starting Hand Gesture Controller{Color.RESET}")

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import json, time, math, threading, urllib.request, collections, subprocess
import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    HAS_NVIDIA_GPU = True
    print(f"{Color.GREEN}[System]{Color.RESET} {Color.WHITE}NVIDIA GPU detected and monitoring enabled.{Color.RESET}")
except Exception as e:
    HAS_NVIDIA_GPU = False
    print(f"{Color.YELLOW}[System]{Color.RESET} {Color.WHITE}Could not initialize NVIDIA NVML for GPU tracking: {e}{Color.RESET}")

import tkinter as tk
from tkinter import ttk, StringVar, messagebox
from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Controller as KeyboardController, Key

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOpts = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"{Color.YELLOW}[Model]{Color.RESET} {Color.WHITE}Downloading hand_landmarker.task (~8 MB)...{Color.RESET}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"{Color.GREEN}[Model]{Color.RESET} {Color.WHITE}Download complete.{Color.RESET}")
    else:
        print(f"{Color.GREEN}[Model]{Color.RESET} {Color.WHITE}Hand tracking model found.{Color.RESET}")

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
]

DEFAULT_CONFIG = {
    "camera_index": 0,
    "smoothing": 5,           
    "click_threshold": 0.266,  
    "scroll_speed": 114,
    "show_camera": True,
    "control_area": 50.0,     
    "camera_zoom": 1.463917525773196,       
    "tracked_hand": "Right Hand Only",
    "enable_volume": True,           
    "volume_threshold": 1.0,         
    "both_hands_open_frames": 3,
    "gestures": {
        "thumb_index_pinch":  "left_click",
        "thumb_middle_pinch": "right_click",
        "thumb_ring_pinch":   "double_click",
        "thumb_pinky_pinch":  "middle_click",
        "two_fingers_up":     "scroll_mode",
        "fist":               "win_tab",
        "open_hand":          "none",
        "both_hands_open":    "desktop_keyboard"
    }
}

CONFIG_FILE = "gesture_config.json"

ACTION_DESCRIPTIONS = {
    "left_click":       "Left click (hold = drag)",
    "right_click":      "Right click",
    "double_click":     "Double click",
    "middle_click":     "Middle click",
    "scroll_mode":      "Scroll",
    "drag_toggle":      "Hold to Drag (Fist)",
    "win_tab":          "Task View (Win + Tab)",
    "desktop_keyboard": "Open On-Screen Keyboard (Win+Ctrl+O)",
    "win_d":            "Show Desktop (Win + D)",
    "alt_f":            "Alt + F",
    "screenshot":       "Save screenshot",
    "none":             "No action"
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
        for k, v in DEFAULT_CONFIG.items():
            if k not in cfg:
                cfg[k] = v
        # Merge any new gesture keys added in newer versions
        for g, a in DEFAULT_CONFIG["gestures"].items():
            if g not in cfg["gestures"]:
                cfg["gestures"][g] = a
        print(f"{Color.GREEN}[Config]{Color.RESET} {Color.WHITE}Loaded settings from {CONFIG_FILE}.{Color.RESET}")
        return cfg
    print(f"{Color.YELLOW}[Config]{Color.RESET} {Color.WHITE}No existing config found. Using defaults.{Color.RESET}")
    return DEFAULT_CONFIG.copy()

def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

def fingers_up(lm):
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    up = [lm[tips[0]].x > lm[pips[0]].x]
    for i in range(1, 5):
        up.append(lm[tips[i]].y < lm[pips[i]].y)
    return up

def detect_circle(pts):
    if len(pts) < 10: return 0
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    
    radii = [math.hypot(p[0]-cx, p[1]-cy) for p in pts]
    if sum(radii)/len(radii) < 0.02: return 0 

    angles = [math.atan2(p[1]-cy, p[0]-cx) for p in pts]
    total_delta = 0
    for i in range(1, len(angles)):
        delta = angles[i] - angles[i-1]
        delta = (delta + math.pi) % (2 * math.pi) - math.pi
        if abs(delta) > math.pi / 2: 
            continue
        total_delta += delta
    return total_delta

def detect_gesture(lm, threshold):
    ref_size = max(0.01, math.hypot(lm[9].x - lm[0].x, (lm[9].y - lm[0].y) * 0.75))
    
    def n_dist(a, b): 
        return math.hypot(a.x - b.x, (a.y - b.y) * 0.75) / ref_size

    if n_dist(lm[4], lm[8])  < threshold: return "thumb_index_pinch"
    if n_dist(lm[4], lm[12]) < threshold: return "thumb_middle_pinch"
    if n_dist(lm[4], lm[16]) < threshold: return "thumb_ring_pinch"
    if n_dist(lm[4], lm[20]) < threshold: return "thumb_pinky_pinch"

    th, idx, mid, rng, pnk = fingers_up(lm)

    if idx and not mid and not rng and not pnk: return "index_up"
    if idx and mid and not rng and not pnk: return "two_fingers_up"
    if not any([idx, mid, rng, pnk]):       return "fist"
    if all([th, idx, mid, rng, pnk]):       return "open_hand"
    return "none"


class HandMouseTracker:
    def __init__(self, config):
        self.config = config
        self.running = False
        self.paused = False
        self.mouse = MouseController()
        self.keyboard = KeyboardController()
        self.sw, self.sh = pyautogui.size()

        self.smooth_x = 0.0
        self.smooth_y = 0.0
        self.last_gesture = "none"
        self.click_cd = 0.0
        
        self.lmb_held = False          
        self.freeze_until = 0.0        
        self.frozen_pos = (0, 0)       
        self.fist_held = False   
        
        self.lmb_press_time = 0.0
        self.fist_press_time = 0.0
        self.lmb_release_time = 0.0
        self.fist_release_time = 0.0
        self.hold_threshold = 0.50 
        self.drop_grace_duration = 0.25 
        
        self.scroll_accum = 0.0
        self.scroll_ref = None
        self.index_history = collections.deque(maxlen=25)

        self.status_text = "Waiting for hand..."
        self.thread = None
        
        self.cpu_usage = 0.0
        self.ram_usage = 0.0
        self.gpu_usage = 0.0

    def start(self):
        self.running = True
        self.paused = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print(f"{Color.GREEN}[Tracker]{Color.RESET} {Color.WHITE}Camera feed and tracking loop started.{Color.RESET}")

    def stop(self):
        self.running = False

    def _release_lmb(self):
        if self.lmb_held:
            pyautogui.mouseUp(button='left')
            self.lmb_held = False
            print(f"{Color.ORANGE}[Action]{Color.RESET} {Color.WHITE}Left Click Released{Color.RESET}")

    def _act_once(self, action):
        now = time.time()
        if now < self.click_cd: return

        if action == "right_click":
            self.mouse.click(Button.right)
            print(f"{Color.ORANGE}[Action]{Color.RESET} {Color.WHITE}Right Click{Color.RESET}")
            self.click_cd = now + 0.40
        elif action == "double_click":
            self.mouse.click(Button.left, 2)
            print(f"{Color.ORANGE}[Action]{Color.RESET} {Color.WHITE}Double Click{Color.RESET}")
            self.click_cd = now + 0.65
        elif action == "middle_click":
            self.mouse.click(Button.middle)
            print(f"{Color.ORANGE}[Action]{Color.RESET} {Color.WHITE}Middle Click{Color.RESET}")
            self.click_cd = now + 0.40
        elif action == "win_tab":
            self.keyboard.press(Key.cmd)
            self.keyboard.press(Key.tab)
            self.keyboard.release(Key.tab)
            self.keyboard.release(Key.cmd)
            print(f"{Color.ORANGE}[Action]{Color.RESET} {Color.WHITE}Win + Tab (Task View){Color.RESET}")
            self.click_cd = now + 1.0 
        elif action == "desktop_keyboard":
            osk_path = r"C:\Windows\SysNative\osk.exe" if os.path.exists(r"C:\Windows\SysNative\osk.exe") else r"C:\Windows\System32\osk.exe"
            subprocess.Popen(osk_path, shell=True)
            try:
                pyautogui.hotkey('alt', 'esc') 
            except:
                pass
            print(f"{Color.ORANGE}[Action]{Color.RESET} {Color.WHITE}Open On-Screen Keyboard (osk.exe){Color.RESET}")
            self.click_cd = now + 1.5
        elif action == "win_d":
            self.keyboard.press(Key.cmd)
            self.keyboard.press('d')
            self.keyboard.release('d')
            self.keyboard.release(Key.cmd)
            print(f"{Color.ORANGE}[Action]{Color.RESET} {Color.WHITE}Win + D (Show Desktop){Color.RESET}")
            self.click_cd = now + 1.0
        elif action == "alt_f":
            self.keyboard.press(Key.alt)
            self.keyboard.press('f')
            self.keyboard.release('f')
            self.keyboard.release(Key.alt)
            print(f"{Color.ORANGE}[Action]{Color.RESET} {Color.WHITE}Alt + F{Color.RESET}")
            self.click_cd = now + 0.5
        elif action == "screenshot":
            fname = f"screenshot_{int(now)}.png"
            pyautogui.screenshot(fname)
            print(f"{Color.ORANGE}[Action]{Color.RESET} {Color.WHITE}Screenshot -> {fname}{Color.RESET}")
            self.click_cd = now + 1.20

    def _loop(self):
        ensure_model()
        
        opts = HandLandmarkerOpts(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.50, 
            min_hand_presence_confidence=0.50,  
            min_tracking_confidence=0.50,       
        )

        cap = cv2.VideoCapture(self.config["camera_index"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        t0 = time.time()
        last_perf_check = 0.0

        with HandLandmarker.create_from_options(opts) as lmk:
            while self.running:
                ret, frame = cap.read()
                if not ret: break

                now = time.time()
                cfg = self.config

                if now - last_perf_check > 1.5:
                    self.cpu_usage = psutil.cpu_percent(interval=None)
                    self.ram_usage = psutil.virtual_memory().percent
                    if HAS_NVIDIA_GPU:
                        try:
                            self.gpu_usage = pynvml.nvmlDeviceGetUtilizationRates(GPU_HANDLE).gpu
                        except pynvml.NVMLError:
                            self.gpu_usage = 0.0
                    last_perf_check = now

                if self.paused:
                    self._release_lmb()
                    if self.fist_held:
                        pyautogui.mouseUp(button='left')
                        self.fist_held = False
                    
                    self.status_text = "Tracking Paused"
                    
                    if cfg.get("show_camera", True):
                        frame = cv2.flip(frame, 1)
                        cv2.rectangle(frame, (0, fh - 26), (fw, fh), (15, 15, 25), -1)
                        cv2.putText(frame, f"{self.status_text}", (8, fh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
                        
                        gpu_str = f"{self.gpu_usage:.1f}%" if HAS_NVIDIA_GPU else "N/A"
                        perf_str = f"CPU: {self.cpu_usage:.1f}% | RAM: {self.ram_usage:.1f}% | GPU: {gpu_str}"
                        cv2.putText(frame, perf_str, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                        
                        cv2.putText(frame, "PAUSED", (int(fw/2)-60, int(fh/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

                        scale = cfg.get("camera_zoom", 1.5)
                        display_frame = cv2.resize(frame, (int(fw * scale), int(fh * scale)), interpolation=cv2.INTER_LINEAR)
                        cv2.imshow("Gesture Controller (Q = quit)", display_frame)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            self.running = False
                            break
                    continue 

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms = int((time.time() - t0) * 1000)
                
                result = lmk.detect_for_video(mp_img, ts_ms)

                # Detect both-hands-open regardless of tracked_hand mode 
                # Only check the 4 non-thumb fingers [1:] â€” thumb direction
                # flips between hands in a mirrored frame and causes false negatives
                both_open_detected = False
                if result.hand_landmarks and len(result.hand_landmarks) >= 2:
                    hand1_x = result.hand_landmarks[0][0].x
                    hand2_x = result.hand_landmarks[1][0].x
                    x_distance = abs(hand1_x - hand2_x)

                    if x_distance > 0.30:
                        open_cnt = sum(
                            1 for h_lm in result.hand_landmarks
                            if all(fingers_up(h_lm)[1:])
                        )
                        if open_cnt >= 2:
                            both_open_detected = True

                base_smooth = max(1, cfg["smoothing"])
                thresh = cfg["click_threshold"]
                gesture = "none"
                action  = "none"

                area = cfg.get("control_area", 50.0) / 100.0
                margin = (1.0 - area) / 2.0
                zx1, zx2 = int(fw * margin), int(fw * (1.0 - margin))
                zy1, zy2 = int(fh * margin), int(fh * (1.0 - margin))

                valid_lm = None
                
                if result.hand_landmarks:
                    target_hand = cfg.get("tracked_hand", "Right Hand Only")
                    for i, handedness in enumerate(result.handedness):
                        phys_hand = "Right" if handedness[0].category_name == "Left" else "Left"
                        if target_hand == "Right Hand Only" and phys_hand != "Right": continue
                        if target_hand == "Left Hand Only" and phys_hand != "Left": continue
                        valid_lm = result.hand_landmarks[i]
                        break 

                if valid_lm:
                    lm = valid_lm
                    raw_x = np.interp(lm[9].x * fw, [zx1, zx2], [0, self.sw])
                    raw_y = np.interp(lm[9].y * fh, [zy1, zy2], [0, self.sh])

                    dist_moved = math.hypot(raw_x - self.smooth_x, raw_y - self.smooth_y)
                    dyn_smooth = base_smooth + 3.0 if dist_moved < 30 else base_smooth
                    
                    self.smooth_x += (raw_x - self.smooth_x) / dyn_smooth
                    self.smooth_y += (raw_y - self.smooth_y) / dyn_smooth

                    mx = int(np.clip(self.smooth_x, 0, self.sw - 1))
                    my = int(np.clip(self.smooth_y, 0, self.sh - 1))

                    gesture = detect_gesture(lm, thresh)
                    action  = cfg["gestures"].get(gesture, "none")

                    # Override with both_hands_open when both hands are open (any mode)
                    if both_open_detected:
                        gesture = "both_hands_open"
                        action  = cfg["gestures"].get("both_hands_open", "none")

                    if action == "pause_toggle" and now > self.click_cd:
                        self.paused = not self.paused
                        self.click_cd = now + 1.0
                        state_str = "Paused" if self.paused else "Resumed"
                        print(f"{Color.ORANGE}[System]{Color.RESET} {Color.WHITE}Tracking {state_str}{Color.RESET}")

                    if not self.paused:
                        if gesture in ("open_hand", "both_hands_open"):
                            self.lmb_release_time = 0.0
                            self.fist_release_time = 0.0
                        
                        if gesture == "index_up" and cfg.get("enable_volume", True):
                            self.index_history.append((lm[8].x, lm[8].y))
                            rot_delta = detect_circle(self.index_history)
                            vol_thresh = cfg.get("volume_threshold", 2.5)
                            
                            if rot_delta > vol_thresh:
                                self.keyboard.press(Key.media_volume_up)
                                self.keyboard.release(Key.media_volume_up)
                                self.index_history.clear()
                                action = "Volume Up"
                                print(f"{Color.ORANGE}[Action]{Color.RESET} {Color.WHITE}Volume Up{Color.RESET}")
                            elif rot_delta < -vol_thresh:
                                self.keyboard.press(Key.media_volume_down)
                                self.keyboard.release(Key.media_volume_down)
                                self.index_history.clear()
                                action = "Volume Down"
                                print(f"{Color.ORANGE}[Action]{Color.RESET} {Color.WHITE}Volume Down{Color.RESET}")
                        else:
                            self.index_history.clear()

                        if action == "drag_toggle" and self.fist_held:
                            if now - self.fist_press_time >= self.hold_threshold:
                                self.fist_release_time = now + self.drop_grace_duration
                            else:
                                self.fist_release_time = now

                        if gesture == "thumb_index_pinch" and action == "left_click" and self.lmb_held:
                            if now - self.lmb_press_time >= self.hold_threshold:
                                self.lmb_release_time = now + self.drop_grace_duration
                            else:
                                self.lmb_release_time = now

                        if action == "drag_toggle":
                            if not self.fist_held:
                                self._release_lmb()
                                pyautogui.mouseDown(button='left')
                                self.fist_held = True
                                self.fist_press_time = now
                                print(f"{Color.ORANGE}[Action]{Color.RESET} {Color.WHITE}Fist Hold Started{Color.RESET}")
                            self.mouse.position = (mx, my)
                        elif self.fist_held:
                            if now >= self.fist_release_time:
                                pyautogui.mouseUp(button='left')
                                self.fist_held = False
                                print(f"{Color.ORANGE}[Action]{Color.RESET} {Color.WHITE}Fist Hold Released{Color.RESET}")
                            else:
                                self.mouse.position = (mx, my) 
                                
                        if gesture == "thumb_index_pinch" and action == "left_click":
                            if not self.lmb_held and now > self.click_cd:
                                self.frozen_pos = (mx, my)
                                self.freeze_until = now + 0.14
                                self.mouse.position = self.frozen_pos
                                pyautogui.mouseDown(button='left')
                                self.lmb_held = True
                                self.lmb_press_time = now
                                print(f"{Color.ORANGE}[Action]{Color.RESET} {Color.WHITE}Left Click Pressed{Color.RESET}")
                            elif self.lmb_held:
                                if now < self.freeze_until:
                                    self.mouse.position = self.frozen_pos
                                else:
                                    self.mouse.position = (mx, my)
                        elif self.lmb_held:
                            if now >= self.lmb_release_time:
                                pyautogui.mouseUp(button='left')
                                self.lmb_held = False
                                print(f"{Color.ORANGE}[Action]{Color.RESET} {Color.WHITE}Left Click Released{Color.RESET}")
                                self.click_cd = now + 0.15 
                            else:
                                if now < self.freeze_until:
                                    self.mouse.position = self.frozen_pos
                                else:
                                    self.mouse.position = (mx, my)

                        if not self.lmb_held and not self.fist_held:
                            if gesture == "two_fingers_up" and action == "scroll_mode":
                                self.mouse.position = (mx, my)
                                if self.scroll_ref is None:
                                    self.scroll_ref = lm[9].y
                                else:
                                    dy = lm[9].y - self.scroll_ref
                                    self.scroll_accum += dy * cfg["scroll_speed"]
                                    
                                    if abs(self.scroll_accum) > 1.0:
                                        ticks = int(self.scroll_accum)
                                        pyautogui.scroll(-ticks * 120)
                                        dir_str = "Up" if ticks < 0 else "Down"
                                        print(f"{Color.ORANGE}[Action]{Color.RESET} {Color.WHITE}Scroll {dir_str}{Color.RESET}")
                                        self.scroll_accum -= ticks
                                        self.scroll_ref = lm[9].y
                            elif gesture.endswith("pinch"):
                                self.scroll_ref = None
                                self.scroll_accum = 0.0
                                self.mouse.position = (mx, my)
                                if gesture != self.last_gesture:
                                    self._act_once(action)
                            else:
                                self.scroll_ref = None
                                self.scroll_accum = 0.0
                                self.mouse.position = (mx, my)
                                if gesture != self.last_gesture:
                                    self._act_once(action)

                        self.last_gesture = gesture
                        
                        bridging = False
                        if self.lmb_held and gesture != "thumb_index_pinch" and now < self.lmb_release_time: bridging = True
                        if self.fist_held and action != "drag_toggle" and now < self.fist_release_time: bridging = True
                        
                        if bridging:
                            self.status_text = "Bridging tracking loss..."
                        else:
                            self.status_text = f"{gesture} -> {action}"

                else:
                    self._release_lmb()
                    if self.fist_held:
                        pyautogui.mouseUp(button='left')
                        self.fist_held = False
                        print(f"{Color.ORANGE}[Action]{Color.RESET} {Color.WHITE}Fist Hold Released (Hand Lost){Color.RESET}")
                        
                    # Both-hands-open can fire even without a single tracked hand
                    if both_open_detected and not self.paused:
                        bh_action = cfg["gestures"].get("both_hands_open", "none")
                        if self.last_gesture != "both_hands_open":
                            self._act_once(bh_action)
                        self.last_gesture = "both_hands_open"
                        self.status_text = f"both_hands_open -> {bh_action}"
                    elif not self.lmb_held and not self.fist_held:
                        self.last_gesture = "none"
                        self.scroll_ref = None
                        self.scroll_accum = 0.0
                        self.status_text = "No hand detected" if not self.paused else "Tracking Paused"
                    else:
                        self.status_text = "Bridging tracking loss..."

                if cfg.get("show_camera", True):
                    if valid_lm:
                        for a, b in HAND_CONNECTIONS:
                            cv2.line(frame, 
                                     (int(lm[a].x * fw), int(lm[a].y * fh)),
                                     (int(lm[b].x * fw), int(lm[b].y * fh)),
                                     (130, 130, 130), 1)
                        for pt in lm:
                            cv2.circle(frame, (int(pt.x * fw), int(pt.y * fh)), 3, (70, 180, 255), -1)

                        cv2.circle(frame, (int(lm[9].x * fw), int(lm[9].y * fh)), 9, (0, 220, 80),  -1)
                        cv2.circle(frame, (int(lm[4].x * fw), int(lm[4].y * fh)), 9, (0, 100, 255), -1)

                    cv2.rectangle(frame, (0, fh - 26), (fw, fh), (15, 15, 25), -1)
                    cv2.putText(frame, f"{self.status_text}", (8, fh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 210, 255), 1)
                    
                    gpu_str = f"{self.gpu_usage:.1f}%" if HAS_NVIDIA_GPU else "N/A"
                    perf_str = f"CPU: {self.cpu_usage:.1f}% | RAM: {self.ram_usage:.1f}% | GPU: {gpu_str}"
                    cv2.putText(frame, perf_str, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                    if self.lmb_held or self.fist_held:
                        cv2.putText(frame, "HOLDING", (fw - 90, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 80, 255), 2)
                    
                    if not valid_lm and not self.paused:
                        cv2.putText(frame, "No valid hand", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 70, 220), 2)
                    elif self.paused:
                        cv2.putText(frame, "PAUSED", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (50, 50, 160), 1)

                    scale = cfg.get("camera_zoom", 1.5)
                    display_frame = cv2.resize(frame, (int(fw * scale), int(fh * scale)), interpolation=cv2.INTER_LINEAR)

                    cv2.imshow("Gesture Controller (Q = quit)", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break

        cap.release()
        cv2.destroyAllWindows()
        self._release_lmb()
        if self.fist_held:
            self.mouse.release(Button.left)
        print(f"{Color.YELLOW}[Tracker]{Color.RESET} {Color.WHITE}Camera feed and tracking stopped.{Color.RESET}")


class GestureApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Gesture Mouse Controller")
        self.root.resizable(False, False)

        self.style = ttk.Style(self.root)
        self.style.theme_use("clam")

        self.cfg = load_config()
        self.tracker = HandMouseTracker(self.cfg)
        
        self.status_var = tk.StringVar(value="Stopped")
        self.gesture_var = tk.StringVar(value="--")
        self.perf_var = tk.StringVar(value="CPU: --% | RAM: --% | GPU: --%")

        self.apply_white_theme()
        self.create_widgets()
        self.create_overlay()
        
        psutil.cpu_percent()
        self._poll()

    def apply_white_theme(self):
        bg = "#ffffff"
        fg = "#000000"
        btn_bg = "#f0f0f0"
        btn_active = "#e0e0e0"

        self.root.configure(bg=bg)
        self.style.configure(".", background=bg, foreground=fg, fieldbackground=bg)
        self.style.configure("TFrame", background=bg)
        self.style.configure("TLabel", background=bg, foreground=fg)
        self.style.configure("TButton", background=btn_bg, foreground=fg)
        self.style.map("TButton", background=[("active", btn_active)], foreground=[("active", fg)])
        self.style.configure("TLabelframe", background=bg, foreground=fg)
        self.style.configure("TLabelframe.Label", background=bg, foreground=fg)
        self.style.configure("TNotebook", background=btn_bg)
        self.style.configure("TNotebook.Tab", background=btn_bg, foreground=fg)
        self.style.map("TNotebook.Tab", background=[("selected", bg)], foreground=[("selected", fg)])

    def create_overlay(self):
        self.overlay = tk.Toplevel(self.root)
        self.overlay.overrideredirect(True) 
        self.overlay.attributes("-topmost", True) 
        
        if sys.platform == "win32":
            self.overlay.attributes("-transparentcolor", "black")
            
        self.overlay.config(bg="black")
        self.overlay.geometry("+20+20")
        
        self.overlay_lbl = tk.Label(
            self.overlay, 
            text="Gesture Tracking Active", 
            font=("Segoe UI", 12, "bold"), 
            fg="white", 
            bg="black"
        )
        self.overlay_lbl.pack()
        self.overlay.withdraw()

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.main_frame = ttk.Frame(self.notebook, padding=10)
        self.settings_frame = ttk.Frame(self.notebook, padding=10)

        self.notebook.add(self.main_frame, text="Main")
        self.notebook.add(self.settings_frame, text="Settings")

        top_bar = ttk.Frame(self.main_frame)
        top_bar.grid(row=0, column=0, columnspan=3, sticky="we")
        ttk.Button(top_bar, text="Help / Info", command=self.show_help).pack(side="left")

        row = 1
        status_frame = ttk.LabelFrame(self.main_frame, text="Status & Performance")
        status_frame.grid(row=row, column=0, columnspan=3, sticky="we", padx=5, pady=10)
        
        ttk.Label(status_frame, text="Tracking Status:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Label(status_frame, textvariable=self.status_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(status_frame, text="Current Gesture:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Label(status_frame, textvariable=self.gesture_var).grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(status_frame, text="System Load:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        ttk.Label(status_frame, textvariable=self.perf_var).grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        row += 1
        self.btn_start = ttk.Button(self.main_frame, text="Start", command=self._start)
        self.btn_start.grid(row=row, column=0, sticky="we", padx=5, pady=5)
        
        self.btn_pause = ttk.Button(self.main_frame, text="Pause", command=self._toggle_pause)
        self.btn_pause.grid(row=row, column=1, sticky="we", padx=5, pady=5)
        self.btn_pause.state(['disabled'])
        
        self.btn_stop = ttk.Button(self.main_frame, text="Stop", command=self._stop)
        self.btn_stop.grid(row=row, column=2, sticky="we", padx=5, pady=5)
        self.btn_stop.state(['disabled'])
        
        row += 1
        ref_frame = ttk.LabelFrame(self.main_frame, text="Quick Reference")
        ref_frame.grid(row=row, column=0, columnspan=3, sticky="we", padx=5, pady=10)
        
        tips = [
            "Index up          ->  none / Volume control",
            "Index circle      ->  Volume control",
            "Thumb + Index     ->  Click / hold to drag",
            "Thumb + Middle    ->  Right click",
            "Thumb + Ring      ->  Double click",
            "Two fingers up    ->  Scroll up/down",
            "Fist              ->  Win + Tab",
            "Both hands        ->  Keyboard",
            "Open Hand         ->  none",
        ]
        for i, tip in enumerate(tips):
            ttk.Label(ref_frame, text=tip, font=("Consolas", 10)).grid(row=i, column=0, sticky="w", padx=5, pady=2)
            
        self.smooth_var = tk.DoubleVar(value=self.cfg.get("smoothing", 5))
        self.thresh_var = tk.DoubleVar(value=self.cfg.get("click_threshold", 0.20))
        self.scroll_var = tk.DoubleVar(value=self.cfg.get("scroll_speed", 60))
        self.area_var   = tk.DoubleVar(value=self.cfg.get("control_area", 50.0))
        self.zoom_var   = tk.DoubleVar(value=self.cfg.get("camera_zoom", 1.5))
        self.hand_var   = tk.StringVar(value=self.cfg.get("tracked_hand", "Right Hand Only"))
        self.cam_var    = tk.BooleanVar(value=self.cfg.get("show_camera", True))
        self.vol_en_var = tk.BooleanVar(value=self.cfg.get("enable_volume", True))
        self.vol_thresh_var = tk.DoubleVar(value=self.cfg.get("volume_threshold", 2.5))
        
        s_row = 0
        sens_frame = ttk.LabelFrame(self.settings_frame, text="Sensitivity & Camera")
        sens_frame.grid(row=s_row, column=0, sticky="we", padx=5, pady=5)
        
        ttk.Label(sens_frame, text="Mouse Smoothing").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Scale(sens_frame, from_=1, to=15, variable=self.smooth_var, orient="horizontal").grid(row=0, column=1, sticky="we", padx=5, pady=5)
        
        ttk.Label(sens_frame, text="Pinch Threshold").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Scale(sens_frame, from_=0.05, to=0.35, variable=self.thresh_var, orient="horizontal").grid(row=1, column=1, sticky="we", padx=5, pady=5)
        
        ttk.Label(sens_frame, text="Scroll Speed").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        ttk.Scale(sens_frame, from_=5, to=150, variable=self.scroll_var, orient="horizontal").grid(row=2, column=1, sticky="we", padx=5, pady=5)

        ttk.Label(sens_frame, text="Control Area (%)").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        ttk.Scale(sens_frame, from_=30, to=100, variable=self.area_var, orient="horizontal").grid(row=3, column=1, sticky="we", padx=5, pady=5)

        ttk.Label(sens_frame, text="Camera Zoom").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        ttk.Scale(sens_frame, from_=1, to=4, variable=self.zoom_var, orient="horizontal").grid(row=4, column=1, sticky="we", padx=5, pady=5)

        ttk.Label(sens_frame, text="Tracked Hand").grid(row=5, column=0, sticky="w", padx=5, pady=5)
        ttk.Combobox(sens_frame, values=["Right Hand Only", "Left Hand Only", "Both Hands"], textvariable=self.hand_var, state="readonly", width=18).grid(row=5, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Checkbutton(sens_frame, text="Show camera window", variable=self.cam_var).grid(row=6, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        s_row += 1
        vol_frame = ttk.LabelFrame(self.settings_frame, text="Volume Control (Index Circle)")
        vol_frame.grid(row=s_row, column=0, sticky="we", padx=5, pady=5)
        
        ttk.Checkbutton(vol_frame, text="Enable Volume Gesture", variable=self.vol_en_var).grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        ttk.Label(vol_frame, text="Rotation Threshold").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Scale(vol_frame, from_=1.0, to=6.0, variable=self.vol_thresh_var, orient="horizontal").grid(row=1, column=1, sticky="we", padx=5, pady=5)
        
        s_row += 1
        map_frame = ttk.LabelFrame(self.settings_frame, text="Gesture Mapping")
        map_frame.grid(row=s_row, column=0, sticky="we", padx=5, pady=5)
        
        GESTURE_LABELS = {
            "thumb_index_pinch":  "Thumb + Index Pinch",
            "thumb_middle_pinch": "Thumb + Middle Pinch",
            "thumb_ring_pinch":   "Thumb + Ring Pinch",
            "thumb_pinky_pinch":  "Thumb + Pinky Pinch",
            "two_fingers_up":     "Index + Middle Up",
            "fist":               "Fist",
            "open_hand":          "Open Hand",
            "both_hands_open":    "Both Hands Open"
        }
        
        ACTION_OPTIONS = list(ACTION_DESCRIPTIONS.keys())
        self.g_vars = {}
        
        for i, (gesture, label) in enumerate(GESTURE_LABELS.items()):
            ttk.Label(map_frame, text=label).grid(row=i, column=0, sticky="w", padx=5, pady=2)
            var = tk.StringVar(value=self.cfg["gestures"].get(gesture, "none"))
            cb = ttk.Combobox(map_frame, values=ACTION_OPTIONS, textvariable=var, state="readonly", width=20)
            cb.grid(row=i, column=1, sticky="w", padx=5, pady=2)
            self.g_vars[gesture] = var
            
        s_row += 1
        btn_frame = ttk.Frame(self.settings_frame)
        btn_frame.grid(row=s_row, column=0, sticky="we", padx=5, pady=10)
        
        ttk.Button(btn_frame, text="Save Settings", command=self.save_settings).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Reset Defaults", command=self.reset_settings).pack(side="left", padx=5)

    def show_help(self):
        text = (
            "Gesture Mouse Controller\n\n"
            "This app lets you control your mouse using hand gestures.\n"
            "- Volume: Hold your index finger up and draw a circle in the air (clockwise = UP)\n"
            "- Performance stats (CPU/RAM/GPU) are shown to monitor the tracking overhead.\n"
        )
        messagebox.showinfo("Help", text)

    def _start(self):
        self.tracker = HandMouseTracker(self.cfg)
        self.tracker.start()
        self.btn_start.state(['disabled'])
        self.btn_pause.state(['!disabled'])
        self.btn_stop.state(['!disabled'])
        self.status_var.set("Running")
        self.overlay_lbl.config(text="Gesture Tracking Active", fg="white")
        self.overlay.deiconify()

    def _toggle_pause(self):
        if self.tracker and self.tracker.running:
            self.tracker.paused = not self.tracker.paused
            if self.tracker.paused:
                self.btn_pause.config(text="Resume")
                self.status_var.set("Paused")
                self.overlay_lbl.config(text="Tracking Paused", fg="yellow")
                print(f"{Color.ORANGE}[System]{Color.RESET} {Color.WHITE}Tracking Paused.{Color.RESET}")
            else:
                self.btn_pause.config(text="Pause")
                self.status_var.set("Running")
                self.overlay_lbl.config(text="Gesture Tracking Active", fg="white")
                print(f"{Color.ORANGE}[System]{Color.RESET} {Color.WHITE}Tracking Resumed.{Color.RESET}")

    def _stop(self):
        self.tracker.stop()
        self.btn_start.state(['!disabled'])
        self.btn_pause.state(['disabled'])
        self.btn_pause.config(text="Pause")
        self.btn_stop.state(['disabled'])
        self.status_var.set("Stopped")
        self.gesture_var.set("--")
        self.overlay.withdraw()

    def _poll(self):
        if self.tracker.running:
            self.gesture_var.set(self.tracker.status_text)
            gpu_str = f"{self.tracker.gpu_usage:.1f}%" if HAS_NVIDIA_GPU else "N/A"
            self.perf_var.set(f"CPU: {self.tracker.cpu_usage:.1f}% | RAM: {self.tracker.ram_usage:.1f}% | GPU: {gpu_str}")
            if not self.tracker.thread.is_alive():
                self._stop()
        else:
            idle_cpu = psutil.cpu_percent(interval=None)
            idle_ram = psutil.virtual_memory().percent
            idle_gpu = 0.0
            if HAS_NVIDIA_GPU:
                try:
                    idle_gpu = pynvml.nvmlDeviceGetUtilizationRates(GPU_HANDLE).gpu
                except pynvml.NVMLError:
                    idle_gpu = 0.0
            gpu_str = f"{idle_gpu:.1f}%" if HAS_NVIDIA_GPU else "N/A"
            self.perf_var.set(f"CPU: {idle_cpu:.1f}% | RAM: {idle_ram:.1f}% | GPU: {gpu_str}")
            
        self.root.after(150, self._poll)

    def save_settings(self):
        self.cfg["smoothing"] = int(self.smooth_var.get())
        self.cfg["click_threshold"] = round(self.thresh_var.get(), 3)
        self.cfg["scroll_speed"] = int(self.scroll_var.get())
        self.cfg["control_area"] = float(self.area_var.get())
        self.cfg["camera_zoom"] = float(self.zoom_var.get())
        self.cfg["tracked_hand"] = self.hand_var.get()
        self.cfg["show_camera"] = bool(self.cam_var.get())
        self.cfg["enable_volume"] = bool(self.vol_en_var.get())
        self.cfg["volume_threshold"] = float(self.vol_thresh_var.get())
        
        for g, v in self.g_vars.items():
            self.cfg["gestures"][g] = v.get()
            
        save_config(self.cfg)
        if self.tracker.running:
            self.tracker.config = self.cfg
        print(f"{Color.GREEN}[Config]{Color.RESET} {Color.WHITE}Settings successfully saved to JSON.{Color.RESET}")
        messagebox.showinfo("Saved", "Settings saved and applied.")

    def reset_settings(self):
        self.cfg.update(DEFAULT_CONFIG)
        save_config(self.cfg)
        
        self.smooth_var.set(self.cfg["smoothing"])
        self.thresh_var.set(self.cfg["click_threshold"])
        self.scroll_var.set(self.cfg["scroll_speed"])
        self.area_var.set(self.cfg["control_area"])
        self.zoom_var.set(self.cfg["camera_zoom"])
        self.hand_var.set(self.cfg["tracked_hand"])
        self.cam_var.set(self.cfg["show_camera"])
        self.vol_en_var.set(self.cfg["enable_volume"])
        self.vol_thresh_var.set(self.cfg["volume_threshold"])
        
        for g, var in self.g_vars.items():
            var.set(self.cfg["gestures"].get(g, "none"))
            
        if self.tracker.running:
            self.tracker.config = self.cfg
            
        print(f"{Color.YELLOW}[Config]{Color.RESET} {Color.WHITE}Settings reset to default.{Color.RESET}")
        messagebox.showinfo("Reset", "Settings reset to defaults.")

def main():
    try:
        p = psutil.Process(os.getpid())
        if sys.platform == "win32":
            p.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            p.nice(-10) 
        print(f"{Color.GREEN}[System]{Color.RESET} {Color.WHITE}Process priority elevated to High.{Color.RESET}")
    except Exception as e:
        print(f"{Color.YELLOW}[System]{Color.RESET} {Color.WHITE}Could not elevate priority: {e}{Color.RESET}")

    root = tk.Tk()
    app = GestureApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app._stop(), root.destroy()))
    
    def on_close():
        app._stop()
        if HAS_NVIDIA_GPU:
            try: pynvml.nvmlShutdown()
            except: pass
        print(f"{Color.ORANGE}[System]{Color.RESET} {Color.WHITE}Shutting down. Goodbye!{Color.RESET}")
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

if __name__ == "__main__":
    main()