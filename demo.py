import argparse
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import time
from collections import deque, Counter
from multiprocessing import Manager, Process, Value
from typing import Optional, Tuple

import onnxruntime as ort
from loguru import logger

ort.set_default_logger_severity(4)  # NOQA
logger.add(sys.stdout, format="{level} | {message}")  # NOQA
logger.remove(0)  # NOQA
import cv2
import numpy as np
from omegaconf import OmegaConf

from constants import classes

# ===================== ТОХИРГОО =====================
HOLD_TIME = 1.2         # Үсэг барих хугацаа (секунд)
COOLDOWN_TIME = 1.5     # Дараагийн үсэг хүлээх хугацаа
CAMERA_INDEX = 0        # Камер индекс (0, 1, 2...)
PRED_BUFFER = 10        # Тогтвортой үсэг тодорхойлох буфер
# ====================================================


class BaseRecognition:
    def __init__(self, model_path: str, tensors_list, prediction_list, verbose):
        self.verbose = verbose
        self.started = None
        self.output_names = None
        self.input_shape = None
        self.input_name = None
        self.session = None
        self.model_path = model_path
        self.window_size = None
        self.tensors_list = tensors_list
        self.prediction_list = prediction_list

    def clear_tensors(self):
        for _ in range(self.window_size):
            self.tensors_list.pop(0)

    def run(self):
        if self.session is None:
            self.session = ort.InferenceSession(self.model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.window_size = self.input_shape[1]
            self.output_names = [output.name for output in self.session.get_outputs()]

        if len(self.tensors_list) >= self.input_shape[1]:
            input_tensor = np.stack(self.tensors_list[: self.window_size], axis=0)[None]
            st = time.time()
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor.astype(np.float32)})[0]
            et = round(time.time() - st, 3)
            gloss = str(classes[outputs.argmax()])
            if gloss != self.prediction_list[-1] and len(self.prediction_list):
                if gloss != "---":
                    self.prediction_list.append(gloss)
            self.clear_tensors()
            if self.verbose:
                logger.info(f"- Prediction time {et}, new gloss: {gloss}")

    def kill(self):
        pass


class Recognition(BaseRecognition):
    def __init__(self, model_path, tensors_list, prediction_list, verbose):
        super().__init__(model_path=model_path, tensors_list=tensors_list,
                         prediction_list=prediction_list, verbose=verbose)
        self.started = True

    def start(self):
        self.run()


class RecognitionMP(Process, BaseRecognition):
    def __init__(self, model_path, tensors_list, prediction_list, verbose):
        super().__init__()
        BaseRecognition.__init__(self, model_path=model_path, tensors_list=tensors_list,
                                  prediction_list=prediction_list, verbose=verbose)
        self.started = Value("i", False)

    def run(self):
        while True:
            BaseRecognition.run(self)
            self.started = True


class Runner:
    STACK_SIZE = 6

    def __init__(self, model_path, config=None, mp=False, verbose=False, length=STACK_SIZE):
        self.multiprocess = mp
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.manager = Manager() if self.multiprocess else None
        self.tensors_list = self.manager.list() if self.multiprocess else []
        self.prediction_list = self.manager.list() if self.multiprocess else []
        self.prediction_list.append("---")
        self.frame_counter = 0
        self.frame_interval = config.frame_interval
        self.length = length
        self.prediction_classes = deque(maxlen=length)
        self.mean = config.mean
        self.std = config.std

        # Текст бичих төлөв
        self.final_text = ""
        self.pred_buffer = deque(maxlen=PRED_BUFFER)
        self.stable_letter = "---"
        self.current_candidate = None
        self.candidate_start = None
        self.last_committed = None
        self.cooldown = False
        self.cooldown_start = 0.0

        if self.multiprocess:
            self.recognizer = RecognitionMP(model_path, self.tensors_list, self.prediction_list, verbose)
        else:
            self.recognizer = Recognition(model_path, self.tensors_list, self.prediction_list, verbose)

    def add_frame(self, image):
        self.frame_counter += 1
        if self.frame_counter == self.frame_interval:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.resize(image, (224, 224))
            image = (image - self.mean) / self.std
            image = np.transpose(image, [2, 0, 1])
            self.tensors_list.append(image)
            self.frame_counter = 0

    @staticmethod
    def resize(im, new_shape=(224, 224)):
        shape = im.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return im

    def update_text_logic(self):
        """Танисан үсгийг текст рүү нэмэх логик"""
        # Сүүлийн таниlt-аас тогтвортой үсэг олох
        current = self.prediction_list[-1] if self.prediction_list else "---"
        self.pred_buffer.append(current)
        self.stable_letter = Counter(self.pred_buffer).most_common(1)[0][0]

        now = time.time()

        # Cooldown шалгах
        if self.cooldown and now - self.cooldown_start > COOLDOWN_TIME:
            self.cooldown = False
            self.last_committed = None

        hold_text = ""
        if self.stable_letter not in ["---"] and not self.cooldown:
            if self.current_candidate != self.stable_letter:
                self.current_candidate = self.stable_letter
                self.candidate_start = now
            else:
                elapsed = now - self.candidate_start
                hold_text = f"Барьж байна: {elapsed:.1f}с/{HOLD_TIME:.1f}с"
                if elapsed >= HOLD_TIME and self.stable_letter != self.last_committed:
                    self.final_text += self.stable_letter
                    self.last_committed = self.stable_letter
                    self.cooldown = True
                    self.cooldown_start = now
                    print(f"✅ Үсэг нэмэгдлээ: {self.stable_letter} → '{self.final_text}'")
        else:
            self.current_candidate = None
            self.candidate_start = None

        return hold_text

    def draw_ui(self, frame):
        """Дэлгэц дээр мэдээлэл харуулах"""
        h, w = frame.shape[:2]

        # Доод мэдээллийн хэсэг
        info_div = np.zeros((120, w, 3), dtype=np.uint8)

        # Тогтвортой үсэг
        cv2.putText(info_div, f"Танилт: {self.stable_letter}",
                    (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        # Hold мэдээлэл
        hold_text = self.update_text_logic()
        if hold_text:
            cv2.putText(info_div, hold_text,
                        (10, 60), cv2.FONT_HERSHEY_COMPLEX, 0.7, (200, 255, 200), 2)

        # Бичигдсэн текст
        display_text = f"Текст: {self.final_text}"
        cv2.putText(info_div, display_text,
                    (10, 95), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

        # Товчлуурын заавар
        key_div = np.zeros((35, w, 3), dtype=np.uint8)
        cv2.putText(key_div, "Q=Гарах  C=Цэвэрлэх  B=Буцаах  SPACE=Зай  S=Хадгалах",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        frame = np.concatenate((frame, info_div, key_div), axis=0)
        return frame

    def save_text(self):
        if not self.final_text.strip():
            print("⚠️ Хадгалах текст хоосон байна.")
            return
        filename = f"text_{int(time.time())}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.final_text)
        print(f"✅ Хадгалагдлаа: {filename}")

    def run(self):
        if self.multiprocess:
            self.recognizer.start()

        print("🚀 Программ эхэллээ!")
        print("📌 Товчлуурууд: Q=Гарах | C=Цэвэрлэх | B=Буцаах | SPACE=Зай | S=Хадгалах")

        while self.cap.isOpened():
            if self.recognizer.started:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    continue

                frame = cv2.flip(frame, 1)
                self.add_frame(frame)

                if not self.multiprocess:
                    self.recognizer.start()

                if len(self.prediction_list) > self.length:
                    self.prediction_list.pop(0)

                frame = self.draw_ui(frame)
                cv2.imshow("Bukva - Дохионы хэл", frame)

                key = cv2.waitKey(10) & 0xFF
                if key in {ord("q"), ord("Q"), 27}:
                    if self.multiprocess:
                        self.recognizer.kill()
                    self.cap.release()
                    cv2.destroyAllWindows()
                    break
                elif key == ord("c") or key == ord("C"):
                    self.final_text = ""
                    print("🗑️ Текст цэвэрлэгдлээ")
                elif key == ord("b") or key == ord("B"):
                    self.final_text = self.final_text[:-1]
                    print(f"⬅️ Буцаасан: '{self.final_text}'")
                elif key == ord(" "):
                    self.final_text += " "
                    print("➕ Зай нэмэгдлээ")
                elif key == ord("s") or key == ord("S"):
                    self.save_text()


def parse_arguments(params: Optional[Tuple] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo Russian Dactyl Recognition...")
    parser.add_argument("-p", "--config", required=True, type=str, help="Path to config")
    parser.add_argument("--mp", required=False, action="store_true", help="Enable multiprocessing")
    parser.add_argument("-v", "--verbose", required=False, action="store_true", help="Enable logging")
    parser.add_argument("-l", "--length", required=False, type=int, default=4, help="Deque length for predictions")
    known_args, _ = parser.parse_known_args(params)
    return known_args


if __name__ == "__main__":
    args = parse_arguments()
    conf = OmegaConf.load(args.config)
    runner = Runner(conf.model_path, conf, args.mp, args.verbose, args.length)
    runner.run()