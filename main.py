import sys,traceback
import os

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

def _crash_handler(exc_type, exc_value, exc_tb):
    with open("jarvis_crash.log", "w") as f:
        traceback.print_exception(exc_type, exc_value, exc_tb, file=f)
    print("CRASHED:", exc_value)

sys.excepthook = _crash_handler

import queue,threading,subprocess,tempfile,time,math,ollama
import webbrowser,json

from enum import Enum, auto
from pathlib import Path
from datetime import datetime
import tkinter as tk

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel
from openwakeword.model import Model

WAKE_WORD_SENSITIVITY = 0.5

SYSTEM_PROMPT = """You are J.A.R.V.I.S. (Just A Rather Very Intelligent System).
Speak with dry British wit and understated confidence. Address the user as "sir" or "ma'am".
Be extremely concise. 1 to 2 sentences maximum. Never leak reasoning, tool names, or code into your replies.

TOOL USE RULES:
Only use tools when the user explicitly requests a system action. Never use tools for conversation or greetings.

EXAMPLES:
- "code a snake game and run it": manage_file(action="write", filename="snake.py", content="...") then terminal_command(command="start python snake.py")
- "open chrome": launch_app(app="chrome")
- "play nothing else matters": play_music(query="nothing else matters metallica")
- "what time is it": get_datetime()
- "take a screenshot": screenshot(filename="screenshot.png")
- "set volume to 50": set_volume(level=50)
- "search for python tutorials": web_search(query="python tutorials")
- "set a 5 minute timer": set_timer(seconds=300, label="5 min")
- "open youtube": open_website(url="youtube.com")
- "kill chrome" or "how much RAM": use terminal_command with taskkill, systeminfo, etc.

RULES:
- You run on Windows. Use CMD syntax for terminal_command.
- Never show code, JSON, or tool calls in your spoken reply.
- Never explain what you are doing. Just do it and confirm briefly.
- When writing code, essays, or long reports, write COMPLETE functional content in the tool call. No placeholders. You MUST provide the full text in your `manage_file` tool call `content` parameter. This is critical.
- NEVER fabricate URLs. Use play_music(query=...) for music, web_search(query=...) for searches.
- For anything not covered by a specific tool, use terminal_command.
"""

conversation_history = []
MAX_HISTORY = 14


class State(Enum):
    SLEEP      = auto()
    LISTENING  = auto()
    PROCESSING = auto()
    THINKING   = auto()
    SPEAKING   = auto()

STATE_LABELS = {
    State.SLEEP:      "STANDBY",
    State.LISTENING:  "LISTENING",
    State.PROCESSING: "PROCESSING",
    State.THINKING:   "ANALYZING",
    State.SPEAKING:   "RESPONDING",
}

STATE_COLORS = {
    State.SLEEP:      "#0a4a6b",
    State.LISTENING:  "#00d4ff",
    State.PROCESSING: "#ffd700",
    State.THINKING:   "#ff8c00",
    State.SPEAKING:   "#00ff9f",
}


class JarvisState:
    def __init__(self):
        self._state = State.SLEEP
        self._lock = threading.Lock()
        self._callbacks = []

    @property
    def value(self):
        with self._lock:
            return self._state

    @value.setter
    def value(self, new_state):
        with self._lock:
            self._state = new_state
        for cb in self._callbacks:
            try: cb(new_state)
            except: pass

    def is_one_of(self, *states):
        with self._lock:
            return self._state in states

    def on_change(self, callback):
        self._callbacks.append(callback)


SR = 44100

def _make_tone(freqs, duration_s, shape="sine", fade_ms=30):
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    wave = np.zeros_like(t)
    for f in freqs:
        if shape == "sine":
            wave += np.sin(2 * np.pi * f * t)
        elif shape == "tri":
            wave += 2 * np.abs(2 * (t * f - np.floor(t * f + 0.5))) - 1
        elif shape == "saw":
            wave += 2 * (t * f - np.floor(0.5 + t * f))
    wave /= max(len(freqs), 1)
    fade_n = int(SR * fade_ms / 1000)
    wave[:fade_n]  *= np.linspace(0, 1, fade_n)
    wave[-fade_n:] *= np.linspace(1, 0, fade_n)
    return (wave * 0.35).astype(np.float32)

def _sweep(f_start, f_end, duration_s, shape="sine", fade_ms=20):
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    freqs = np.linspace(f_start, f_end, len(t))
    if shape == "saw":
        wave = 2 * (t * freqs - np.floor(0.5 + t * freqs))
    else:
        wave = np.sin(2 * np.pi * freqs * t)
    fade_n = int(SR * fade_ms / 1000)
    wave[:fade_n]  *= np.linspace(0, 1, fade_n)
    wave[-fade_n:] *= np.linspace(1, 0, fade_n)
    return (wave * 0.35).astype(np.float32)

_out_lock = threading.Lock()

def _play_numpy(arr, fs=SR):
    try:
        with _out_lock:
            with sd.OutputStream(samplerate=fs, channels=1, dtype="float32") as stream:
                chunk = 1024
                for i in range(0, len(arr), chunk):
                    stream.write(arr[i:i+chunk].reshape(-1, 1))
    except:
        pass

def _play_async(arr):
    threading.Thread(target=_play_numpy, args=(arr,), daemon=True).start()

_play = _play_numpy
_silence = lambda ms: np.zeros(int(SR * ms / 1000), dtype=np.float32)

TONE_WAKE = np.concatenate([
    _sweep(80, 1200, 0.12, shape="saw"), _silence(30),
    _make_tone([1200, 2400], 0.06), _silence(20),
    _make_tone([2400], 0.04, fade_ms=5),
])
TONE_STARTUP = np.concatenate([
    _sweep(60, 400, 0.18, shape="saw"), _silence(40),
    _sweep(300, 1800, 0.14, shape="sine"), _silence(30),
    _make_tone([900, 1800], 0.08), _silence(25),
    _make_tone([1200, 2400], 0.05), _silence(20),
    _make_tone([2400], 0.12, fade_ms=60),
])
TONE_RESPOND = np.concatenate([
    _sweep(600, 1400, 0.07, shape="saw"), _silence(15),
    _make_tone([1400], 0.04, fade_ms=8),
])
TONE_THINKING = np.concatenate([
    _make_tone([200, 400], 0.06, shape="saw"), _silence(20),
    _make_tone([300, 600], 0.06, shape="saw"), _silence(20),
    _make_tone([400, 800], 0.06, shape="saw"),
])
TONE_INTERRUPT = np.concatenate([
    _sweep(1800, 300, 0.1, shape="saw"), _silence(10),
    _make_tone([200], 0.06, fade_ms=5),
])
TONE_ERROR = np.concatenate([
    _make_tone([120, 180], 0.15, shape="saw"), _silence(30),
    _make_tone([100], 0.2, shape="saw"),
])
TONE_SLEEP = np.concatenate([
    _sweep(1200, 80, 0.14, shape="saw"), _silence(20),
    _make_tone([80], 0.06, fade_ms=40),
])

_tts_queue   = queue.Queue()
_tts_state   = None
_tts_interrupt = None
_tool_log_cb = [None]

def _tts_thread_main():
    import torch
    silero_model = None
    silero_sample_rate = 48000
    try:
        device = torch.device('cpu')
        import os
        try:
            hub_dir = torch.hub.get_dir()
            local_repo = os.path.join(hub_dir, 'snakers4_silero-models_master')
            if os.path.exists(local_repo):
                silero_model, _ = torch.hub.load(
                    repo_or_dir=local_repo,
                    source='local',
                    model='silero_tts',
                    language='en',
                    speaker='v3_en',
                    verbose=False,
                    trust_repo=True
                )
            else:
                raise FileNotFoundError()
        except Exception:
            os.environ["HF_HUB_OFFLINE"] = "0"
            silero_model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language='en',
                speaker='v3_en',
                verbose=False,
                trust_repo=True
            )
            os.environ["HF_HUB_OFFLINE"] = "1"
        silero_model.to(device)
    except Exception as e:
        if _tool_log_cb[0]:
            _tool_log_cb[0](f"SILERO LOAD ERROR: {e}", "system")

    while True:
        try:
            text = _tts_queue.get(timeout=0.05)
        except queue.Empty:
            continue
        if text is None:
            break
        clean = text.replace("*", "").replace("#", "")
        if _tts_state:
            _tts_state.value = State.SPEAKING
        if _tts_interrupt:
            _tts_interrupt.clear()
        _play_async(TONE_RESPOND)
        time.sleep(0.2)
        interrupted = False
        if silero_model and clean.strip():
            try:
                audio_tensor = silero_model.apply_tts(text=clean, speaker='en_2', sample_rate=silero_sample_rate)
                audio_data = audio_tensor.numpy().reshape(-1, 1)
                chunk = 2048
                with _out_lock:
                    with sd.OutputStream(samplerate=silero_sample_rate, channels=1, dtype="float32") as stream:
                        for i in range(0, len(audio_data), chunk):
                            if _tts_interrupt and _tts_interrupt.is_set():
                                interrupted = True
                                break
                            stream.write(audio_data[i:i+chunk])
            except Exception as e:
                if _tool_log_cb[0]:
                    _tool_log_cb[0](f"TTS ERROR: {e}", "system")
        else:
            time.sleep(1)
        if interrupted:
            _play_async(TONE_INTERRUPT)
        if _tts_state and not interrupted:
            _tts_state.value = State.SLEEP

threading.Thread(target=_tts_thread_main, daemon=True).start()

def speak(text: str):
    _tts_queue.put(text)

MEMORY_FILE = "jarvis_long_term_memory.txt"

def terminal_command(command: str) -> str:
    """Run a shell command on Windows CMD."""
    if _tool_log_cb[0]:
        _tool_log_cb[0](f"terminal_command  ›  {command}", "exec")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout if result.stdout else result.stderr
    except Exception as e:
        return str(e)

def manage_file(action: str, filename: str, content: str = "") -> str:
    """Read or write a file. action: 'read' or 'write'. content only for write."""
    if _tool_log_cb[0]:
        _tool_log_cb[0](f"manage_file  ›  {action} {filename}", "exec")
    try:
        if action == "write":
            with open(filename, "w") as f:
                f.write(content)
            return f"Wrote to {filename}."
        elif action == "read":
            with open(filename, "r") as f:
                return f.read()
    except Exception as e:
        return str(e)

def save_memory(fact: str) -> str:
    """Save a fact to long-term memory."""
    if _tool_log_cb[0]:
        _tool_log_cb[0](f"save_memory  ›  {fact[:60]}", "exec")
    try:
        with open(MEMORY_FILE, "a") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] {fact}\n")
        return f"Committed to long-term memory: {fact}"
    except Exception as e:
        return str(e)

SYSTEM_APPS = {
    "notepad":         "notepad.exe",
    "calculator":      "calc.exe",
    "paint":           "mspaint.exe",
    "cmd":             "cmd.exe",
    "explorer":        "explorer.exe",
    "taskmgr":         "taskmgr.exe",
    "task manager":    "taskmgr.exe",
    "control panel":   "control.exe",
    "wordpad":         "wordpad.exe",
    "snipping tool":   "snippingtool.exe",
    "regedit":         "regedit.exe",
}

def launch_app(app: str) -> str:
    """Launch an application by name."""
    if _tool_log_cb[0]:
        _tool_log_cb[0](f"launch_app  ›  {app}", "exec")
    app_clean = app.lower().replace(".exe", "").replace(".lnk", "").strip()
    if app_clean in SYSTEM_APPS:
        try:
            subprocess.Popen(SYSTEM_APPS[app_clean], shell=True)
            return f"Launched {app_clean}."
        except Exception as e:
            return str(e)
    start_paths = [
        os.path.expandvars(r"%ProgramData%\Microsoft\Windows\Start Menu\Programs"),
        os.path.expandvars(r"%AppData%\Microsoft\Windows\Start Menu\Programs"),
    ]
    try:
        for base in start_paths:
            for root, dirs, files in os.walk(base):
                for f in files:
                    if app_clean in f.lower() and f.lower().endswith(".lnk"):
                        full_path = os.path.join(root, f)
                        subprocess.Popen(["cmd", "/c", "start", "", full_path], shell=True)
                        return f"Launched {f}."
        try:
            subprocess.Popen(f"start {app_clean}", shell=True)
            return f"Attempted to launch {app_clean}."
        except Exception:
            pass
        return f"Application '{app_clean}' not found."
    except Exception as e:
        return str(e)



def set_volume(level: int) -> str:
    """Set system volume (0-100)."""
    if _tool_log_cb[0]:
        _tool_log_cb[0](f"set_volume  ›  {level}%", "exec")
    try:
        level = max(0, min(100, int(level)))
        from ctypes import cast, POINTER
        import comtypes
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volume.SetMasterVolumeLevelScalar(level / 100.0, None)
        return f"Volume set to {level}%."
    except ImportError:
        
        try:
            val = int(65535 * level / 100)
            subprocess.run(f"nircmd.exe setsysvolume {val}", shell=True, timeout=5)
            return f"Volume set to {level}%."
        except Exception as e:
            return f"Could not set volume: {e}"
    except Exception as e:
        return f"Error setting volume: {e}"


def web_search(query: str) -> str:
    """Search Google in the default browser."""
    if _tool_log_cb[0]:
        _tool_log_cb[0](f"web_search  ›  {query}", "exec")
    try:
        import urllib.parse
        url = f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}"
        webbrowser.open(url)
        return f"Searching the web for '{query}'."
    except Exception as e:
        return str(e)


def open_website(url: str) -> str:
    """Open a URL in the default browser."""
    if _tool_log_cb[0]:
        _tool_log_cb[0](f"open_website  ›  {url}", "exec")
    try:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        webbrowser.open(url)
        return f"Opened {url}."
    except Exception as e:
        return str(e)


def get_datetime() -> str:
    """Get current date, time, and day of week."""
    if _tool_log_cb[0]:
        _tool_log_cb[0]("get_datetime  ›  querying", "exec")
    now = datetime.now()
    return now.strftime("Date: %A, %B %d, %Y | Time: %I:%M:%S %p")


def set_timer(seconds: int, label: str = "Timer") -> str:
    """Set a countdown timer that announces when done."""
    if _tool_log_cb[0]:
        _tool_log_cb[0](f"set_timer  ›  {seconds}s ({label})", "exec")
    def _timer_thread():
        time.sleep(int(seconds))
        msg = f"Your timer '{label}' has finished, sir."
        if _tool_log_cb[0]:
            _tool_log_cb[0](f"TIMER COMPLETE  ›  {label}", "exec")
        _play_async(TONE_WAKE)
        speak(msg)
    threading.Thread(target=_timer_thread, daemon=True).start()
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    if mins > 0:
        return f"Timer '{label}' set for {mins} minute(s) and {secs} second(s)."
    return f"Timer '{label}' set for {secs} second(s)."


def screenshot(filename: str = "screenshot.png") -> str:
    """Take a screenshot and save it."""
    if _tool_log_cb[0]:
        _tool_log_cb[0](f"screenshot  ›  {filename}", "exec")
    try:
        from PIL import ImageGrab
    except ImportError:
        return "Pillow (PIL) package not installed. Run: pip install Pillow"
    try:
        img = ImageGrab.grab()
        img.save(filename)
        full_path = str(Path(filename).resolve())
        return f"Screenshot saved to {full_path}."
    except Exception as e:
        return str(e)


def play_music(query: str = "") -> str:
    """Play music. With query: search YouTube. Without: play random local file."""
    if _tool_log_cb[0]:
        _tool_log_cb[0](f"play_music  ›  {query or '(shuffle local)'}", "exec")
    try:
        if query:
            import urllib.parse, urllib.request, re
            try:
                search_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(query)}"
                req = urllib.request.Request(search_url, headers={"User-Agent": "Mozilla/5.0"})
                html = urllib.request.urlopen(req, timeout=8).read().decode("utf-8", errors="ignore")
                match = re.search(r'"videoId":"([a-zA-Z0-9_-]{11})"', html)
                if match:
                    video_url = f"https://www.youtube.com/watch?v={match.group(1)}"
                    webbrowser.open(video_url)
                    return f"Now playing '{query}' on YouTube."
            except Exception:
                pass
            
            webbrowser.open(f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(query)}")
            return f"Opened YouTube search for '{query}'."
        import random
        music_dirs = [
            os.path.expanduser("~/Music"),
            os.path.expanduser("~/Desktop"),
            os.path.expanduser("~/Downloads"),
        ]
        audio_exts = ('.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma')
        found = []
        for music_dir in music_dirs:
            if not os.path.exists(music_dir):
                continue
            for root, dirs, files in os.walk(music_dir):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                for f in files:
                    if f.lower().endswith(audio_exts):
                        found.append(os.path.join(root, f))
                if len(found) >= 100:
                    break
        if not found:
            webbrowser.open("https://www.youtube.com")
            return "No local music found. Opened YouTube instead."
        pick = random.choice(found)
        os.startfile(pick)
        return f"Now playing: {os.path.basename(pick)}"
    except Exception as e:
        return str(e)


ALL_TOOLS = [
    terminal_command, manage_file, save_memory, launch_app,
    set_volume, web_search, open_website,
    get_datetime, set_timer, play_music, screenshot,
]

TOOLS_MAPPING = {fn.__name__: fn for fn in ALL_TOOLS}


FALSE_SUCCESS = (
    "opened", "launched", "started", "created", "written", "executed",
    "done", "completed", "running", "saved", "deleted", "installed",
)

def _is_false_success(reply: str) -> bool:
    r = reply.lower()
    return any(w in r for w in FALSE_SUCCESS)

FALSE_SUCCESS = (
    "opened", "launched", "started", "created", "written", "executed",
    "done", "completed", "running", "saved", "deleted", "installed",
)

RETRY_PHRASES = (
    "let me try again", "let me retry", "let me correct", "let me fix",
    "trying again", "retrying", "i will try", "i'll try", "attempting again",
)

def _is_false_success(reply: str) -> bool:
    r = reply.lower()
    return any(w in r for w in FALSE_SUCCESS)

def _wants_retry(reply: str) -> bool:
    r = reply.lower()
    return any(p in r for p in RETRY_PHRASES)

def smart_respond(prompt, memory_context=""):
    global conversation_history
    sys_msg = SYSTEM_PROMPT
    if memory_context:
        sys_msg += f"\n\nLong-Term Memory Facts:\n{memory_context}"
    messages = [{"role": "system", "content": sys_msg}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": prompt})
    tool_calls_made = False
    false_success_retried = False
    retry_count = 0
    MAX_RETRIES = 3
    while True:
        response = ollama.chat(
            model="qwen2.5:7b",
            messages=messages,
            tools=ALL_TOOLS,
        )
        messages.append(response.message)
        if response.message.tool_calls:
            tool_calls_made = True
            for tc in response.message.tool_calls:
                name = tc.function.name
                args = tc.function.arguments or {}
                if name in TOOLS_MAPPING:
                    try:
                        result = TOOLS_MAPPING[name](**args)
                    except Exception as e:
                        result = str(e)
                    messages.append({"role": "tool", "name": name, "content": str(result)})
        else:
            reply = response.message.content
            if not tool_calls_made and not false_success_retried and _is_false_success(reply):
                false_success_retried = True
                messages.append({"role": "user", "content": "You said you completed that but you did not call any tool. Call the appropriate tool now."})
                continue
            if _wants_retry(reply) and retry_count < MAX_RETRIES:
                retry_count += 1
                messages.append({"role": "user", "content": "Go ahead and try again now. Call the tool — do not just say you will."})
                continue
            conversation_history.append({"role": "user",      "content": prompt})
            conversation_history.append({"role": "assistant", "content": reply})
            conversation_history[:] = conversation_history[-MAX_HISTORY:]
            return reply

try:
    whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")
except Exception:
    if _tool_log_cb[0]:
        _tool_log_cb[0]("Downloading Whisper model...", "system")
    os.environ["HF_HUB_OFFLINE"] = "0"
    whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8", download_root=os.path.join(os.environ.get("USERPROFILE", ""), ".cache", "huggingface", "hub"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    
oww_model     = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")


def _warmup_ollama():
    try:
        ollama.chat(model="qwen2.5:7b", messages=[{"role": "user", "content": "hi"}])
    except: pass
threading.Thread(target=_warmup_ollama, daemon=True).start()

def process_audio_and_respond(audio_data, state, interrupt_flag, log_callback):
    state.value = State.THINKING
    _play_async(TONE_THINKING)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        wav.write(tmp.name, 16000, audio_data)
        segments, _ = whisper_model.transcribe(tmp.name, beam_size=1)
        command_text = "".join(s.text for s in segments).strip()
    finally:
        tmp.close()
        Path(tmp.name).unlink(missing_ok=True)
    if len(command_text) < 2:
        state.value = State.SLEEP
        return
    log_callback(f"YOU  ›  {command_text}", "user")
    memory_context = ""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            memory_context = f.read()
    try:
        import requests
        requests.get("http://127.0.0.1:11434", timeout=10)
    except Exception as e:
        log_callback(f"Ollama unreachable: {e}", "system")
        speak("I cannot reach Ollama, sir. Please ensure the server is running.")
        state.value = State.SLEEP
        return
    try:
        response = smart_respond(command_text, memory_context)
    except Exception as e:
        log_callback(f"Agent error: {e}", "system")
        _play_async(TONE_ERROR)
        response = "I encountered an error executing that task, sir."
    log_callback(f"JARVIS  ›  {response}", "jarvis")
    speak(str(response))

_audio_lock = threading.Lock()

def make_audio_callback(state, interrupt_flag, log_callback):
    audio_buffer       = []
    silence_frames     = [0]
    last_wake_time     = [0.0]
    WAKE_COOLDOWN      = 2.0
    SILENCE_THRESHOLD  = 0.01
    MAX_SILENCE_FRAMES = 20

    def cb(indata, frames, time_info, status):
        audio_int16 = (indata.flatten() * 32768).astype(np.int16)
        preds = oww_model.predict(audio_int16)
        if any(s > WAKE_WORD_SENSITIVITY for s in preds.values()):
            now = time.time()
            if now - last_wake_time[0] < WAKE_COOLDOWN:
                return
            last_wake_time[0] = now
            if state.is_one_of(State.SPEAKING, State.THINKING):
                interrupt_flag.set()
            state.value = State.LISTENING
            with _audio_lock:
                audio_buffer.clear()
            silence_frames[0] = 0
            _play_async(TONE_WAKE)
            return
        cur = state.value
        if cur in (State.PROCESSING, State.THINKING, State.SPEAKING, State.SLEEP):
            return
        if cur is State.LISTENING:
            with _audio_lock:
                audio_buffer.append(indata.copy())
            rms = float(np.sqrt(np.mean(indata ** 2)))
            silence_frames[0] = silence_frames[0] + 1 if rms < SILENCE_THRESHOLD else 0
            if silence_frames[0] > MAX_SILENCE_FRAMES:
                state.value = State.PROCESSING
                with _audio_lock:
                    recording = np.concatenate(audio_buffer) if audio_buffer else None
                    audio_buffer.clear()
                if recording is not None:
                    threading.Thread(
                        target=process_audio_and_respond,
                        args=(recording, state, interrupt_flag, log_callback),
                        daemon=True,
                    ).start()
                else:
                    state.value = State.SLEEP
    return cb


BG         = "#03080f"
BG_PANEL   = "#060d17"
BG_CARD    = "#070e1a"
CYAN       = "#00d4ff"
CYAN2      = "#00aacc"
CYAN_DIM   = "#003a50"
CYAN_GHOST = "#001828"
GOLD       = "#f0c040"
GOLD_DIM   = "#3a2e00"
GREEN      = "#00ff9f"
ORANGE     = "#ff8c00"
WHITE      = "#b8dff0"
GREY       = "#1a3a55"
GREY2      = "#254a6a"


class CoreRing(tk.Canvas):
    def __init__(self, parent, size=300, **kw):
        super().__init__(parent, width=size, height=size,
                         bg=BG, highlightthickness=0, **kw)
        self.size = size
        self.cx   = size / 2
        self.cy   = size / 2
        self.R    = size / 2 - 8

        self._angle    = 0.0
        self._angle2   = 180.0
        self._sweep    = 0.0
        self._pulse    = 0.0
        self._pulse_dir = 1
        self._state    = State.SLEEP
        self._bars     = [0.0] * 32
        self._tick_n   = 0
        self._hex_dots = self._make_hex_dots()
        self._draw()

    def _make_hex_dots(self):
        dots = []
        step = 16
        cx, cy = self.cx, self.cy
        for row in range(-14, 15):
            for col in range(-14, 15):
                x = cx + col * step * 1.732
                y = cy + row * step + (col % 2) * step * 0.5
                if math.hypot(x - cx, y - cy) < self.R - 4:
                    dots.append((x, y))
        return dots

    def set_state(self, s):
        self._state = s

    def update_bars(self, rms):
        self._bars.pop(0)
        self._bars.append(min(rms * 70, 1.0))

    def _color(self):
        return STATE_COLORS.get(self._state, CYAN)

    def _draw(self):
        self.delete("all")
        cx, cy, R = self.cx, self.cy, self.R
        col = self._color()

        self._draw_hex_grid(cx, cy, col)
        self._draw_scanner(cx, cy, R, col)
        self._draw_outer_ring(cx, cy, R, col)
        self._draw_dashes(cx, cy, R, col)
        self._draw_mid_ring(cx, cy, R, col)
        self._draw_audio_bars(cx, cy, R, col)
        self._draw_inner_ring(cx, cy, R, col)
        self._draw_core(cx, cy, R, col)
        self._draw_label(cx, cy, col)
        self._draw_corner_data(col)

        self._pulse += 0.035 * self._pulse_dir
        if self._pulse > 1 or self._pulse < 0:
            self._pulse_dir *= -1
        speed = {
            State.SLEEP: 0.3, State.LISTENING: 2.8,
            State.PROCESSING: 3.5, State.THINKING: 4.0, State.SPEAKING: 2.2
        }.get(self._state, 1.0)
        self._angle  = (self._angle  + speed)      % 360
        self._angle2 = (self._angle2 - speed * 0.7) % 360
        self._sweep  = (self._sweep  + speed * 2.2) % 360
        self._tick_n += 1

        self.after(33, self._draw)

    def _draw_hex_grid(self, cx, cy, col):
        r, g, b = self.winfo_rgb(col)
        dim = "#{:02x}{:02x}{:02x}".format(r >> 10, g >> 10, b >> 10)
        for (x, y) in self._hex_dots:
            self.create_rectangle(x, y, x+1, y+1, fill=dim, outline="")

    def _draw_scanner(self, cx, cy, R, col):
        sweep_rad = math.radians(self._sweep)
        spread    = math.radians(60)
        for i in range(30):
            frac  = i / 30
            alpha = sweep_rad - spread * frac
            r, g, b = self.winfo_rgb(col)
            intensity = int((1 - frac) * 28)
            fade = "#{:02x}{:02x}{:02x}".format(
                min(255, (r >> 8) * intensity // 28),
                min(255, (g >> 8) * intensity // 28),
                min(255, (b >> 8) * intensity // 28),
            )
            ex = cx + R * 0.88 * math.cos(alpha)
            ey = cy + R * 0.88 * math.sin(alpha)
            self.create_line(cx, cy, ex, ey, fill=fade, width=1)

    def _draw_outer_ring(self, cx, cy, R, col):
        r0 = R
        self.create_oval(cx-r0, cy-r0, cx+r0, cy+r0,
                         outline=GREY2, fill="", width=1)
        self.create_oval(cx-r0, cy-r0, cx+r0, cy+r0,
                         outline=col, fill="", width=2)

    def _draw_dashes(self, cx, cy, R, col):
        for i in range(72):
            a = math.radians(i * 5 + self._angle * 0.25)
            if i % 9 == 0:
                r1, r2, w = R - 2, R - 12, 2
                c = col
            elif i % 3 == 0:
                r1, r2, w = R - 2, R - 7, 1
                c = GREY2
            else:
                r1, r2, w = R - 2, R - 4, 1
                c = GREY
            self.create_line(
                cx + r1 * math.cos(a), cy + r1 * math.sin(a),
                cx + r2 * math.cos(a), cy + r2 * math.sin(a),
                fill=c, width=w,
            )

    def _draw_mid_ring(self, cx, cy, R, col):
        r_mid = R * 0.72
        self.create_oval(cx-r_mid, cy-r_mid, cx+r_mid, cy+r_mid,
                         outline=GREY2, fill="", width=1)
        for i in range(6):
            a0 = math.radians(self._angle2 + i * 60)
            a1 = math.radians(self._angle2 + i * 60 + 38)
            self.create_arc(
                cx - r_mid, cy - r_mid, cx + r_mid, cy + r_mid,
                start=-math.degrees(a0), extent=-38,
                outline=col, style="arc", width=2,
            )
        for i in range(3):
            a0 = math.radians(self._angle + i * 120 + 30)
            a1 = math.radians(self._angle + i * 120 + 55)
            self.create_arc(
                cx - r_mid, cy - r_mid, cx + r_mid, cy + r_mid,
                start=-math.degrees(a0), extent=-25,
                outline=GOLD, style="arc", width=1,
            )

    def _draw_audio_bars(self, cx, cy, R, col):
        r_inner = R * 0.78
        r_outer = R * 0.96
        n = len(self._bars)
        for i, bar in enumerate(self._bars):
            a = math.radians(i * (360 / n) + self._angle * 0.1)
            bar_h = bar * (r_outer - r_inner)
            r1 = r_inner
            r2 = r_inner + max(bar_h, 1)
            if bar > 0.01:
                self.create_line(
                    cx + r1 * math.cos(a), cy + r1 * math.sin(a),
                    cx + r2 * math.cos(a), cy + r2 * math.sin(a),
                    fill=col, width=2 if bar > 0.3 else 1,
                )

    def _draw_inner_ring(self, cx, cy, R, col):
        r_in = R * 0.48
        pulse_r = r_in * (0.88 + 0.12 * self._pulse)
        self.create_oval(
            cx - pulse_r, cy - pulse_r, cx + pulse_r, cy + pulse_r,
            outline=col, fill="", width=2,
        )
        self.create_oval(
            cx - r_in * 0.72, cy - r_in * 0.72,
            cx + r_in * 0.72, cy + r_in * 0.72,
            outline=GREY2, fill=BG, width=1,
        )
        for i in range(4):
            a = math.radians(self._angle + i * 90)
            r1, r2 = r_in * 0.74, r_in * 0.86
            self.create_line(
                cx + r1 * math.cos(a), cy + r1 * math.sin(a),
                cx + r2 * math.cos(a), cy + r2 * math.sin(a),
                fill=col, width=2,
            )

    def _draw_core(self, cx, cy, R, col):
        rc = R * 0.30
        glow = R * 0.36 * (0.9 + 0.1 * self._pulse)
        r, g, b = self.winfo_rgb(col)
        glow_col = "#{:02x}{:02x}{:02x}".format(r >> 9, g >> 9, b >> 9)
        self.create_oval(cx - glow, cy - glow, cx + glow, cy + glow,
                         fill=glow_col, outline="")
        self.create_oval(cx - rc, cy - rc, cx + rc, cy + rc,
                         fill=BG_CARD, outline=col, width=2)

    def _draw_label(self, cx, cy, col):
        label = STATE_LABELS.get(self._state, "")
        r, g, b = self.winfo_rgb(col)
        glow_col = "#{:02x}{:02x}{:02x}".format(r >> 9, g >> 9, b >> 9)
        for ox, oy in [(-1,-1),(1,-1),(-1,1),(1,1),(0,-2),(0,2),(-2,0),(2,0)]:
            self.create_text(cx+ox, cy-10+oy, text="J·A·R·V·I·S",
                             fill=glow_col, font=("Courier", 13, "bold"))
        self.create_text(cx, cy - 10, text="J·A·R·V·I·S",
                         fill=col, font=("Courier", 13, "bold"))
        self.create_text(cx, cy + 10, text=label,
                         fill=GREY2, font=("Courier", 7, "bold"))

    def _draw_corner_data(self, col):
        t = self._tick_n
        snippets = [
            (4,  4,  f"AZ {self._angle:05.1f}°",     "nw"),
            (self.size-4, 4,  f"SWP {self._sweep:05.1f}°", "ne"),
            (4,  self.size-4, f"PLX {self._pulse:.2f}",     "sw"),
            (self.size-4, self.size-4, f"T+{t:05d}",        "se"),
        ]
        for x, y, txt, anchor in snippets:
            self.create_text(x, y, text=txt, fill=GREY2,
                             font=("Courier", 6), anchor=anchor)

    @staticmethod
    def _hex_pts(cx, cy, r, off=0):
        pts = []
        for i in range(6):
            a = math.radians(60 * i + off)
            pts += [cx + r * math.cos(a), cy + r * math.sin(a)]
        return pts


class JarvisGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("J.A.R.V.I.S.")
        self.root.configure(bg=BG)
        self.root.minsize(960, 640)
        self.root.geometry("1100x740")

        self.state          = JarvisState()
        self.interrupt_flag = threading.Event()
        self._log_queue     = queue.Queue()

        self.state.on_change(self._on_state_change)
        self._build_ui()
        self._start_audio()
        self._poll()

        global _tts_state, _tts_interrupt
        _tts_state     = self.state
        _tts_interrupt = self.interrupt_flag
        _tool_log_cb[0] = lambda msg, tag: self._log_queue.put(("log", msg, tag))

        threading.Thread(target=lambda: (time.sleep(0.5), _play(TONE_STARTUP)),
                         daemon=True).start()
        self.after_log("J.A.R.V.I.S. SYSTEMS ONLINE — ALL MODULES NOMINAL", "system")
        self.after_log('Awaiting wake word: "Hey Jarvis"', "system")

    def _hline(self, parent, color=None, pady=3):
        c = color or CYAN_DIM
        cv = tk.Canvas(parent, bg=BG, height=1, highlightthickness=0)
        cv.pack(fill="x", pady=pady)
        cv.bind("<Configure>", lambda e: (
            cv.delete("all"),
            cv.create_line(0, 0, e.width, 0, fill=c),
        ))

    def _section(self, parent, text, color=None):
        c = color or GREY2
        f = tk.Frame(parent, bg=BG)
        f.pack(fill="x", padx=6, pady=(8, 2))
        tk.Label(f, text=text.upper(), bg=BG, fg=c,
                 font=("Courier", 7, "bold")).pack(side="left")
        cv = tk.Canvas(f, bg=BG, height=1, highlightthickness=0)
        cv.pack(side="left", fill="x", expand=True, padx=(6, 0))
        cv.bind("<Configure>", lambda e: (
            cv.delete("all"),
            cv.create_line(0, 0, e.width, 0, fill=CYAN_DIM),
        ))

    def _build_ui(self):
        topbar = tk.Frame(self.root, bg="#020b14", height=48)
        topbar.pack(fill="x")
        topbar.pack_propagate(False)

        tk.Label(topbar, text="J·A·R·V·I·S", bg="#020b14", fg=CYAN,
                 font=("Courier", 18, "bold")).pack(side="left", padx=16, pady=6)

        meta = tk.Frame(topbar, bg="#020b14")
        meta.pack(side="left", pady=10)
        tk.Label(meta, text="MARK VII  ·  NEURAL INTERFACE  ·  v7.3.1",
                 bg="#020b14", fg=GREY2, font=("Courier", 8)).pack(anchor="w")
        tk.Label(meta, text="STARK INDUSTRIES  ·  CLEARANCE: ALPHA  ·  RESTRICTED",
                 bg="#020b14", fg=GREY, font=("Courier", 7)).pack(anchor="w")

        right_top = tk.Frame(topbar, bg="#020b14")
        right_top.pack(side="right", padx=14, pady=6)
        self._clock_var = tk.StringVar()
        tk.Label(right_top, textvariable=self._clock_var,
                 bg="#020b14", fg=GOLD, font=("Courier", 10, "bold")).pack(anchor="e")
        self._uptime_var = tk.StringVar(value="UP  00:00:00")
        tk.Label(right_top, textvariable=self._uptime_var,
                 bg="#020b14", fg=GREY2, font=("Courier", 7)).pack(anchor="e")
        self._start_time = time.time()
        self._tick_clock()

        scan = tk.Canvas(self.root, bg="#020b14", height=2, highlightthickness=0)
        scan.pack(fill="x")
        scan.bind("<Configure>", lambda e: (
            scan.delete("all"),
            scan.create_line(0, 0, e.width, 0, fill=CYAN, width=1),
            scan.create_line(0, 1, e.width, 1, fill=CYAN_DIM, width=1),
        ))

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True)

        left = tk.Frame(body, bg=BG, width=280)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        tk.Frame(left, bg=CYAN_DIM, height=1).pack(fill="x")

        ring_frame = tk.Frame(left, bg=BG)
        ring_frame.pack(pady=(12, 4))
        self.core_ring = CoreRing(ring_frame, size=240)
        self.core_ring.pack()

        self._status_var = tk.StringVar(value="STANDBY")
        tk.Label(left, textvariable=self._status_var,
                 bg=BG, fg=CYAN, font=("Courier", 11, "bold")).pack(pady=(2, 0))

        self._hline(left)
        self._section(left, "▸ system metrics")

        metrics = tk.Frame(left, bg=BG)
        metrics.pack(fill="x", padx=10, pady=2)
        self._mem_var     = tk.StringVar(value="0")
        self._session_var = tk.StringVar(value="0")
        self._cmd_var     = tk.StringVar(value="0")
        self._cmd_count   = 0
        self._session_count = 0
        for label, var, color in [
            ("MEM RECORDS",   self._mem_var,     GOLD),
            ("SESSION OPS",   self._session_var,  CYAN),
            ("CMDS EXEC",     self._cmd_var,      GREEN),
        ]:
            row = tk.Frame(metrics, bg=BG)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=label, bg=BG, fg=GREY2,
                     font=("Courier", 7)).pack(side="left")
            tk.Label(row, textvariable=var, bg=BG, fg=color,
                     font=("Courier", 8, "bold")).pack(side="right")

        self._hline(left)
        self._section(left, "▸ neural status")

        status_grid = tk.Frame(left, bg=BG)
        status_grid.pack(fill="x", padx=10, pady=2)
        self._status_dots = {}
        for name in ("WHISPER ASR", "WAKE DETECT", "LLM CORE", "SILERO TTS"):
            row = tk.Frame(status_grid, bg=BG)
            row.pack(fill="x", pady=2)
            dot = tk.Canvas(row, width=8, height=8, bg=BG, highlightthickness=0)
            dot.pack(side="left", padx=(0, 6))
            d = dot.create_oval(1, 1, 7, 7, fill=GREEN, outline="")
            self._status_dots[name] = (dot, d)
            tk.Label(row, text=name, bg=BG, fg=GREY2,
                     font=("Courier", 7)).pack(side="left")

        self._hline(left)
        self._section(left, "▸ wake sensitivity")
        sens_frame = tk.Frame(left, bg=BG)
        sens_frame.pack(fill="x", padx=10, pady=2)
        self._sens_var = tk.DoubleVar(value=WAKE_WORD_SENSITIVITY)
        tk.Scale(
            sens_frame, variable=self._sens_var, from_=0.1, to=0.9,
            resolution=0.05, orient="horizontal",
            bg=BG, fg=CYAN, troughcolor=CYAN_GHOST,
            highlightthickness=0, bd=0,
            font=("Courier", 7), command=self._update_sensitivity,
        ).pack(fill="x")

        tk.Frame(body, bg=CYAN_DIM, width=1).pack(side="left", fill="y")

        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        pane = tk.PanedWindow(right, orient="vertical", bg=BG,
                              sashwidth=5, sashrelief="flat", sashpad=0)
        pane.pack(fill="both", expand=True)

        log_frame = tk.Frame(pane, bg=BG)
        pane.add(log_frame, stretch="always")

        lh = tk.Frame(log_frame, bg=BG_PANEL, height=24)
        lh.pack(fill="x")
        lh.pack_propagate(False)
        tk.Label(lh, text="▸ COMMUNICATION LOG", bg=BG_PANEL, fg=GREY2,
                 font=("Courier", 7, "bold")).pack(side="left", padx=10, pady=5)
        self._log_count_var = tk.StringVar(value="0 ENTRIES")
        tk.Label(lh, textvariable=self._log_count_var, bg=BG_PANEL, fg=GREY2,
                 font=("Courier", 7)).pack(side="right", padx=10)

        lt = tk.Frame(log_frame, bg=BG_CARD)
        lt.pack(fill="both", expand=True)
        self.log_text = tk.Text(
            lt, bg=BG_CARD, fg=WHITE, font=("Courier", 10),
            relief="flat", state="disabled", wrap="word",
            insertbackground=CYAN, selectbackground=CYAN_DIM,
            padx=12, pady=8, spacing1=4,
        )
        sb = tk.Scrollbar(lt, command=self.log_text.yview,
                          bg=BG_PANEL, troughcolor=BG_CARD)
        self.log_text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)
        self.log_text.tag_config("system", foreground=GOLD)
        self.log_text.tag_config("user",   foreground=CYAN)
        self.log_text.tag_config("jarvis", foreground=GREEN)
        self.log_text.tag_config("ts",     foreground=GREY2)
        self.log_text.tag_config("exec",   foreground=ORANGE)

        exec_frame = tk.Frame(pane, bg=BG)
        pane.add(exec_frame, stretch="never", height=150)

        eh = tk.Frame(exec_frame, bg="#020a10", height=24)
        eh.pack(fill="x")
        eh.pack_propagate(False)
        tk.Label(eh, text="▸ EXECUTION MONITOR", bg="#020a10", fg=ORANGE,
                 font=("Courier", 7, "bold")).pack(side="left", padx=10, pady=5)
        self._exec_count_var = tk.StringVar(value="IDLE")
        tk.Label(eh, textvariable=self._exec_count_var, bg="#020a10", fg=ORANGE,
                 font=("Courier", 7)).pack(side="right", padx=10)

        et = tk.Frame(exec_frame, bg="#040d08")
        et.pack(fill="both", expand=True)
        self.exec_text = tk.Text(
            et, bg="#040d08", fg=ORANGE, font=("Courier", 9),
            relief="flat", state="disabled", wrap="word",
            insertbackground=ORANGE, padx=12, pady=6, spacing1=2,
        )
        esb = tk.Scrollbar(et, command=self.exec_text.yview,
                           bg="#020a10", troughcolor="#040d08")
        self.exec_text.configure(yscrollcommand=esb.set)
        esb.pack(side="right", fill="y")
        self.exec_text.pack(side="left", fill="both", expand=True)
        self.exec_text.tag_config("cmd", foreground=ORANGE)
        self.exec_text.tag_config("mem", foreground=GOLD)

        bot = tk.Frame(self.root, bg="#020b14", height=26)
        bot.pack(fill="x", side="bottom")
        bot.pack_propagate(False)
        tk.Frame(bot, bg=CYAN_DIM, height=1).pack(fill="x", side="top")

        self._indicator = tk.Canvas(bot, width=8, height=8,
                                    bg="#020b14", highlightthickness=0)
        self._indicator.pack(side="left", padx=(12, 5), pady=9)
        self._dot = self._indicator.create_oval(1, 1, 7, 7, fill=CYAN_DIM, outline="")

        self._footer_var = tk.StringVar(value='AWAITING WAKE WORD')
        tk.Label(bot, textvariable=self._footer_var,
                 bg="#020b14", fg=GREY2, font=("Courier", 7, "bold")).pack(side="left")
        tk.Label(bot, text="OLLAMA · WHISPER · OPENWAKEWORD · SILERO-TTS",
                 bg="#020b14", fg=GREY, font=("Courier", 7)).pack(side="right", padx=12)

    def _tick_clock(self):
        self._clock_var.set(datetime.now().strftime("%Y.%m.%d  %H:%M:%S"))
        elapsed = int(time.time() - self._start_time)
        h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
        self._uptime_var.set(f"UP  {h:02d}:{m:02d}:{s:02d}")
        self.root.after(1000, self._tick_clock)

    def _update_sensitivity(self, val):
        global WAKE_WORD_SENSITIVITY
        WAKE_WORD_SENSITIVITY = float(val)

    def _on_state_change(self, new_state):
        self._log_queue.put(("_state_ui", new_state))

    def _apply_state_ui(self, new_state):
        color = STATE_COLORS.get(new_state, CYAN)
        self._status_var.set(STATE_LABELS.get(new_state, ""))
        self.core_ring.set_state(new_state)
        self._indicator.itemconfig(self._dot, fill=color)
        self._footer_var.set({
            State.SLEEP:      'AWAITING WAKE WORD  ·  SAY "HEY JARVIS"',
            State.LISTENING:  "AUDIO INPUT ACTIVE  ·  SPEAK NOW",
            State.PROCESSING: "TRANSCRIBING AUDIO  ·  STAND BY",
            State.THINKING:   "LLM INFERENCE  ·  STAND BY",
            State.SPEAKING:   "TTS OUTPUT  ·  RENDERING RESPONSE",
        }.get(new_state, ""))
        if new_state is State.THINKING:
            self._exec_count_var.set("PROCESSING")
        elif new_state is State.SLEEP:
            self._exec_count_var.set("IDLE")

    def after_log(self, msg, tag="system"):
        self._log_queue.put(("log", msg, tag))

    def _write_log(self, msg, tag):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:12]
        if tag == "exec":
            self.exec_text.config(state="normal")
            self.exec_text.insert("end", f"[{ts}]  {msg}\n", "cmd")
            self.exec_text.config(state="disabled")
            self.exec_text.see("end")
            self._cmd_count += 1
            self._cmd_var.set(str(self._cmd_count))
            self._exec_count_var.set(f"{self._cmd_count} OPS")
        else:
            self.log_text.config(state="normal")
            self.log_text.insert("end", f"[{ts}] ", "ts")
            self.log_text.insert("end", f"{msg}\n", tag)
            self.log_text.config(state="disabled")
            self.log_text.see("end")
            if tag == "user":
                self._session_count += 1
                self._session_var.set(str(self._session_count))
            entries = int(self.log_text.index("end-1c").split(".")[0])
            self._log_count_var.set(f"{entries} ENTRIES")
        self._refresh_memory_count()

    def _refresh_memory_count(self):
        try:
            if os.path.exists(MEMORY_FILE):
                with open(MEMORY_FILE) as f:
                    n = sum(1 for line in f if line.strip())
                self._mem_var.set(str(n))
        except:
            pass

    def _poll(self):
        try:
            while True:
                item = self._log_queue.get_nowait()
                if item[0] == "log":
                    self._write_log(item[1], item[2])
                elif item[0] == "_state_ui":
                    self._apply_state_ui(item[1])
        except queue.Empty:
            pass
        self.root.after(50, self._poll)

    def _start_audio(self):
        def log_cb(msg, tag):
            self._log_queue.put(("log", msg, tag))
        cb = make_audio_callback(self.state, self.interrupt_flag, log_cb)
        self._stream = sd.InputStream(
            samplerate=16000, channels=1, dtype="float32",
            blocksize=1280, callback=cb,
        )
        self._stream.start()


def main():
    root = tk.Tk()
    root.resizable(True, True)
    try:
        root.iconbitmap("jarvis.ico")
    except:
        pass
    app = JarvisGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app._stream.stop(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
