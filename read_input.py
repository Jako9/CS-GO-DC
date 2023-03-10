import numpy as np
from mss import mss, tools, screenshot
import win32api
import os, time
from dataclasses import dataclass
from typing import Union, List, Tuple
from datetime import datetime
from pandas import DataFrame
from config import config
import logging
from multiprocessing import Process, Queue, Value, Pipe
from threading import Thread
from queue import Queue as ThreadQueue
import cv2


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


def producer(queue: Queue, active: Value, warmup_seconds=5, fps=30):
    n_snapshots = 0
    start_time = time.time()
    sct = mss()

    while active.value:
        current_time = time.time()
        time_since_start = current_time - start_time
        frames_behind = time_since_start * fps - n_snapshots
        if frames_behind > 0:
            if frames_behind > 1:
                logging.warning(f"{frames_behind} frames behind")

            # create snapshot
            time_ns = time.time_ns()
            pressed_keys = get_keys(use_key_list=True)
            img = sct.grab(sct.monitors[1])
            mouse_position = (0, 0)  # win32api.GetCursorPos()
            snapshot = Snapshot(img, pressed_keys, mouse_position, time_ns)

            n_snapshots += 1
            if time_since_start > warmup_seconds:
                # save snapshot if warmup is over
                queue.put(snapshot)
                pass
        else:
            time.sleep(0.001)  # 100 FPS

    queue.put(None)  # signal the consumer to stop


def save(image: screenshot.ScreenShot, output_path: os.PathLike):
    # tools.to_png(image.rgb, image.size, output=output_path)

    arr = np.array(image)
    # print(arr.shape)
    # print(arr.dtype)

    # save the image
    cv2.imwrite(output_path, arr)


def consumer(queue, output_dir):
    data = []

    while True:
        s = queue.get()
        if s is None:
            # stop producer
            break
        file_name = f"{s.timestamp}.png"
        full_path = os.path.join(output_dir, file_name)

        save(s.image, full_path)

        data.append(
            {
                "timestamp": s.timestamp,
                "keys": s.pressed_keys,
                "mouse_position": s.mouse_position,
            }
        )
    data = DataFrame(data)
    data.to_csv(os.path.join(output_dir, "data.csv"))


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


def main():
    fps = 35
    root_path = config["Output_Dir"]
    q_size = config["Buffer_Size"]

    data_manager = DataManager(root_path)
    dir = data_manager.next_directory()

    logging.info(f"FPS: {fps} | Root path: {root_path} | Queue size: {q_size}")

    # create a queue to store the images
    Q = Queue(maxsize=q_size)

    active = Value("b", True)

    # start processes
    producer_proc = Process(target=producer, args=(Q, active, 5, fps))
    consumer_proc = Process(target=consumer, args=(Q, dir))

    # start the processes
    producer_proc.start()
    consumer_proc.start()

    time.sleep(20)

    # terminate the processes
    active.value = False

    # wait for the processes to terminate
    producer_proc.join()
    logging.debug("Producer joined")
    consumer_proc.join()
    logging.debug("Consumer joined")

    logging.info("Done.")


if __name__ == "__main__":
    # example usage
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
