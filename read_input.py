import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api
from mss import mss, tools, screenshot
from queue import Queue
import os, time
import threading
from dataclasses import dataclass
from typing import Union, List, Tuple


CSGO_HWIN = "Counter-Strike: Global Offensive - Direct3D 9"

active = False
fps = 30

output_dir = r"E:\test_screenshot"
# output_dir = r"C:\Users\Raphael\Desktop\screenshot_test"

threads = []
QUEUE_SIZE = 2048
Q = Queue(maxsize=QUEUE_SIZE)


@dataclass(frozen=True)
class Snapshot:
    iteration: int
    image: Union[np.ndarray, screenshot.ScreenShot]
    pressed_keys: List[str]
    mouse_position: Tuple[int, int]
    timestamp: int  # in nanoseconds

    @property
    def img(self):
        return self.image

    @property
    def formatted_timestamp(self):
        secs = self.timestamp / 1e9
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(secs))


def get_keys(use_key_list=False):
    keys = []
    if use_key_list:
        # mouse1 = 0x01
        # mouse2 = 0x02
        # mouse3 = 0x04
        KEY_LIST = ["\x01", "\x02", "W", "A", "S", "D", "\x11", "\x20", "R"]
        for key in KEY_LIST:
            if win32api.GetAsyncKeyState(ord(key)):
                keys.append(key)
    else:
        for i in range(1, 256):
            if win32api.GetAsyncKeyState(i):
                keys.append(i)
        keys = [chr(p) for p in keys]

    return keys  # return the list of keys pressed


def grabber():
    WARMUP_SECONDS = 5

    n_screenshots = -fps * WARMUP_SECONDS  #  2 seconds warmup

    start_time = time.time()

    while True:
        if active:
            current_time = time.time()

            behind = (current_time - start_time) * fps - n_screenshots

            if behind >= 1:
                print(f"Queue size: {Q.qsize()}")

                if behind > 5 and n_screenshots >= 0:
                    print(f"Behind by {behind} screenshots")
                if Q.full():
                    raise RuntimeWarning("Queue is full!")

                timestamp = time.time_ns()
                pressed_keys = get_keys(use_key_list=True)
                img = grab_csgo()
                mouse_position = (0, 0)  # win32api.GetCursorPos()

                s = Snapshot(
                    n_screenshots,
                    img,
                    pressed_keys,
                    mouse_position,
                    timestamp,
                )
                valid = n_screenshots >= 0
                Q.put((valid, s))
                n_screenshots += 1
                time.sleep(0.005)  # 200 fps
            else:
                time.sleep(0.01)  # 100 fps
        else:
            break


def saver():
    while True:
        if active or not Q.empty():
            if not Q.empty():
                valid, s = Q.get()
                if valid:
                    file_name = f"{s.timestamp}.png"
                    full_path = os.path.join(output_dir, file_name)
                    save(s.image, full_path)
                    print(f"Saved {full_path}")
                time.sleep(0.01)  # 100 fps
            else:
                time.sleep(0.025)  # 40 fps
        else:
            break


def start_threads():
    global active
    global Q
    global threads
    active = True
    Q = Queue(maxsize=QUEUE_SIZE)

    threads = []

    g = threading.Thread(target=grabber)
    threads.append(g)

    s = threading.Thread(target=saver)
    threads.append(s)

    s.start()
    g.start()


def stop_threads():
    global active
    global threads

    active = False

    for t in threads:
        t.join()


def grab_window(hwin):
    try:
        (left, top, right, bottom) = win32gui.GetWindowRect(hwin)
        width = right - left
        bar_height = 23  # height of header bar
        height = bottom - top - bar_height

        width -= 6
        height -= 6

        hwindc = win32gui.GetWindowDC(hwin)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, width, height)
        memdc.SelectObject(bmp)
        memdc.BitBlt(
            (0, 0), (width, height), srcdc, (3, 3 + bar_height), win32con.SRCCOPY
        )

        signedIntsArray = bmp.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype="uint8")
        img.shape = (height, width, 4)

        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(hwin, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())

        screenshot = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        return screenshot
    except Exception:
        # print(traceback.format_exc())
        raise RuntimeWarning("Could not grab window")


def get_hwin(guessed_name):
    top_windows = []
    windowEnumerationHandler = lambda x, y: y.append(x)
    win32gui.EnumWindows(windowEnumerationHandler, top_windows)
    count = 0
    for current_hwin in top_windows:
        win_name = win32gui.GetWindowText(current_hwin)
        if win_name.lower().find(guessed_name.lower()) != -1:
            count += 1
            hwin = current_hwin
    if count != 1:
        print("hwin couldnt be identified")
        print("CSGO may be not running")
        exit(-1)
    return hwin


def grab(hwin=None):
    if hwin:
        try:
            return grab_window(get_hwin(hwin))
        except:
            # fallback to mss
            pass
    with mss() as sct:
        sct.compression_level = 0
        return sct.grab(sct.monitors[1])


def save(img, out_path):
    if isinstance(
        img,
        screenshot.ScreenShot,
    ):
        tools.to_png(img.rgb, img.size, output=out_path)
    else:
        try:
            cv2.imwrite(out_path, img)
        except:
            raise RuntimeError("Unknown type of img")


def grab_csgo():
    global CSGO_HWIN
    return grab(hwin=CSGO_HWIN)


if __name__ == "__main__":
    start = time.perf_counter()
    start_threads()
    time.sleep(30)
    stop_threads()
    end = time.perf_counter()
    print(f"Time elapsed: {end - start} seconds")
