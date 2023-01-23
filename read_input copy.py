import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api
from mss import mss, tools, screenshot
from queue import Queue
import os, time
import threading
from dataclasses import dataclass
from typing import Union, List, Tuple
from datetime import datetime
from pandas import DataFrame
from config import config
import logging


@dataclass(frozen=True)
class Snapshot:
    """
    Dataclass for storing a snapshot of the game
    """

    image: Union[np.ndarray, screenshot.ScreenShot]
    pressed_keys: List[str]
    mouse_position: Tuple[int, int]
    timestamp_ns: int

    @property
    def img(self):
        return self.image

    @property
    def timestamp(self):
        d = datetime.fromtimestamp(self.timestamp_ns / 1e9)
        return d.strftime("%Y-%m-%d_%H-%M-%S.%f")


def create_snapshot():
    time_ns = time.time_ns()
    pressed_keys = get_keys()
    img = grab()  # grab_csgo()
    mouse_position = (0, 0)  # win32api.GetCursorPos()
    return Snapshot(img, pressed_keys, mouse_position, time_ns)


class GrabbingThread(threading.Thread):
    def __init__(self, queue, fps, warmup_seconds=5):
        super().__init__()
        self._queue = queue
        self._fps = fps
        self._warmup_seconds = warmup_seconds
        self._active = True

    def run(self):
        n_snapshots = 0
        start_time = time.time()
        fst = True

        while True:
            if self._active:
                current_time = time.time()
                time_since_start = current_time - start_time
                frames_behind = time_since_start * self._fps - n_snapshots
                if frames_behind > 0:
                    if self._queue.full():
                        raise RuntimeError("Queue is full")
                    if frames_behind > 1:
                        logging.warning(f"{frames_behind} frames behind")
                    snapshot = create_snapshot()
                    n_snapshots += 1
                    if time_since_start > self._warmup_seconds:
                        if fst:
                            logging.info("Started putting snapshots in queue")
                            fst = False
                        self._queue.put(snapshot)
                else:
                    time.sleep(0.01)  # 100 fps
            else:
                break
        logging.info("GrabbingThread stopped")

    def stop(self):
        self._active = False


class SavingThread(threading.Thread):
    def __init__(self, queue, output_dir):
        super().__init__()
        self._queue = queue
        self._output_dir = output_dir
        self._active = True
        self._data = []

    def run(self):
        while True:
            if self._active or not self._queue.empty():
                s = self._queue.get()
                file_name = f"{s.timestamp}.png"
                full_path = os.path.join(self._output_dir, file_name)
                save(s.image, full_path)
                self._data.append(
                    {
                        "timestamp": s.timestamp,
                        "keys": s.pressed_keys,
                        "mouse_position": s.mouse_position,
                    }
                )

                logging.debug(f"Saved {full_path}")
                time.sleep(0.02)  # 50 fps
            else:
                break
        logging.info("Saving data to csv")
        self.df.to_csv(os.path.join(self._output_dir, "data.csv"))
        logging.info("SavingThread stopped")

    def stop(self):
        self._active = False

    @property
    def df(self):
        return DataFrame(self._data)


class ThreadManager:
    def __init__(self, output_dir, fps=30, queue_size=1024, warmup_seconds=5):
        self._threads = []
        self._fps = fps
        self._queue = Queue(maxsize=queue_size)
        self._output_dir = output_dir
        self._warmup_seconds = warmup_seconds

    def start_threads(self):
        self._threads.append(
            GrabbingThread(self._queue, self._fps, self._warmup_seconds)
        )
        self._threads.append(SavingThread(self._queue, self._output_dir))
        for t in self._threads:
            t.start()

    def stop_threads(self):
        for t in self._threads:
            t.stop()
        for t in self._threads:
            t.join()


class DataManager:
    def __init__(self, root_path):
        self._root_path = root_path
        self._current_dir = None

    def next_directory(self) -> os.PathLike:
        time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dir_path = os.path.join(self._root_path, time_stamp)

        if os.path.isdir(dir_path):
            offset = 0
            while os.path.isdir(dir_path + f"__{offset}") and offset < 5:
                offset += 1
            if offset == 5:
                raise RuntimeError("Could not create new directory")
            else:
                dir_path += f"_{offset}"
        try:
            os.makedirs(dir_path)
        except OSError as e:
            raise RuntimeError(f"Could not create directory {dir_path}") from e

        self._current_dir = dir_path

        # create new logging handler for this directory
        handler = logging.FileHandler(os.path.join(dir_path, "log.txt"))
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        # remove old handlers except for the console handler
        for h in logging.getLogger().handlers:
            if not isinstance(h, logging.StreamHandler):
                logging.getLogger().removeHandler(h)
        logging.getLogger().addHandler(handler)

        logging.info(f"Created directory {dir_path}")

        return dir_path


def get_keys(use_key_list=False):
    """Returns a list of currently pressed keys.

    Args:
        use_key_list (bool, optional): Only check predefined keys like 'W','A','S','D' etc. . Defaults to False.

    Returns:
        List[str]]: List of currently pressed keys
    """
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


def grab_window(hwin):
    """
    Takes a screenshot of the window with the given handle. Returns a numpy array.

    Args:
        hwin (_type_): Handle of the window to take a screenshot of.

    Raises:
        RuntimeWarning: Window is not found

    Returns:
        np.ndarray: Numpy array of the screenshot of the window.
    """
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
    """
    Finds the handle of the window with the given name.

    Args:
        guessed_name (_type_): Name of the window to find.

    Returns:
        int: Handle of the window with the given name.
    """
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
    """Grabs a screenshot of the given window or the primary monitor.
    If the window is not found, it falls back to mss.
    Mss is a bit slower, but it works on all platforms.

    Args:
        hwin (str, optional): Name of the window to look for. Defaults to None.

    Returns:
        Union(np.ndarray, screenshot.ScreenShot): Screenshot of the given window or the primary monitor. Type depends on the method used.
    """
    if hwin:
        try:
            return grab_window(get_hwin(hwin))
        except:
            # fallback to mss
            pass
    with mss() as sct:
        # sct.compression_level = 0
        return sct.grab(sct.monitors[1])


def save(img: Union[np.ndarray, screenshot.ScreenShot], out_path: os.PathLike):
    """Saves the given image to disk.

    Args:
        img (np.ndarray, screenshot.ScreenShot): Image to save.
        out_path (os.PathLike): Path to save the image to.

    Raises:
        RuntimeError: Unknown type of the image.
    """
    if isinstance(
        img,
        screenshot.ScreenShot,
    ):
        tools.to_png(img.rgb, img.size, output=out_path)
    else:
        try:
            print("saving image")
            cv2.imwrite(out_path, img)
        except:
            raise RuntimeError("Unknown type of img")


def grab_csgo():
    """
    Shortcut for grabbing a screenshot of CSGO.

    Returns:
        Union(np.ndarray, screenshot.ScreenShot): Screenshot of CSGO.
    """
    global hwin
    return grab(hwin)


def main():
    fps = 25
    root_path = config["Output_Dir"]
    q_size = config["Buffer_Size"]

    data_manager = DataManager(root_path)
    dir = data_manager.next_directory()

    logging.info(f"FPS: {fps} | Root path: {root_path} | Queue size: {q_size}")

    thread_manager = ThreadManager(dir, fps, q_size)

    print("Starting threads")
    start = time.perf_counter()
    thread_manager.start_threads()
    time.sleep(30)
    end = time.perf_counter()
    print(f"Time elapsed: {end - start} seconds")
    thread_manager.stop_threads()
    print("Done")


if __name__ == "__main__":
    # example usage
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
    main()
