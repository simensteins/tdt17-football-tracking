"""Reading jersey numbers from person crops.

Shared by scripts/read_jerseys.py (our tracks), scripts/bench_jersey_readers.py
(SoccerNet-GSR tracks with labelled numbers) and scripts/link_tracks.py.

Two readers:
- 'florence': Florence-2-large OCR (Microsoft, MIT licence). Reads clear
  numbers, but answers '1' for many crops without a number.
- 'qwen': Qwen3.5-4B (Alibaba, Apache 2.0), a vision-language model asked
  for the number, with 'none' as an allowed answer. When it cannot see a
  number it often guesses ('10', '1'), so it also returns the probability
  of its answer, and unsure answers can be dropped.

A reader maps BGR crops to (answer, probability) pairs; Florence-2 gives no
probability (NaN).
"""

from __future__ import annotations

import re
from typing import Callable

import numpy as np
import pandas as pd

from football.data import CLASSES

NUMBER = re.compile(r"\d+")
READERS = {"florence": "florence-community/Florence-2-large", "qwen": "Qwen/Qwen3.5-4B"}
QUESTION = ("Read the jersey number on this football player's shirt. If no number is clearly "
            "visible, answer 'none'. Reply with the number or 'none' only.")


def pick_frames(track: pd.DataFrame, n: int, min_height: float) -> pd.DataFrame:
    """Up to n boxes spread over the track, preferring the larger ones."""
    tall = track[track["h"] >= min_height]
    track = tall if len(tall) else track
    if len(track) <= n:
        return track
    track = track.sort_values("frame")
    stretches = np.array_split(np.arange(len(track)), n)  # one box per stretch of the track
    return pd.concat([track.iloc[s].nlargest(1, "h") for s in stretches])


def upper_body(img: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
    """The shirt below the head (15-60 % of the box height), where back numbers sit."""
    x0, x1 = int(max(x, 0)), int(min(x + w, img.shape[1]))
    y0, y1 = int(max(y + 0.15 * h, 0)), int(min(y + 0.6 * h, img.shape[0]))
    return img[y0:y1, x0:x1]


def shirt_colour(img: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
    """Mean CIELAB colour of the shirt's centre (middle half of the width, 20-50 % of the height)."""
    import cv2

    patch = img[int(y + 0.2 * h):int(y + 0.5 * h), int(x + 0.25 * w):int(x + 0.75 * w)]
    if patch.size == 0:
        return np.full(3, np.nan)
    return cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(0)


def parse_number(text: str) -> str:
    """The first one- or two-digit number in a reader's answer, or ''."""
    numbers = [n for n in NUMBER.findall(text) if len(n) <= 2]
    return numbers[0] if numbers else ""


def make_reader(name: str, model_id: str | None = None) -> Callable[[list[np.ndarray]], list[tuple[str, float]]]:
    """A function from BGR crops to the reader's raw answers and their probabilities."""
    import torch
    from PIL import Image

    model_id = model_id or READERS[name]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if name == "florence":
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        dtype = torch.float16 if device == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(model_id)
        model = Florence2ForConditionalGeneration.from_pretrained(model_id, dtype=dtype).to(device).eval()

        def read(crops: list[np.ndarray]) -> list[tuple[str, float]]:
            images = [Image.fromarray(c[:, :, ::-1]) for c in crops]
            inputs = processor(text=["<OCR>"] * len(images), images=images, return_tensors="pt").to(device, dtype)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=12, num_beams=1, do_sample=False)
            return [(t.strip(), float("nan")) for t in processor.batch_decode(out, skip_special_tokens=True)]

        return read

    if name == "qwen":
        import cv2
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"  # batched generation continues after the prompt
        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype).to(device).eval()
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)

        def read(crops: list[np.ndarray]) -> list[tuple[str, float]]:
            # Crops are small (a 100 px player gives a 45 px shirt); upscale so the
            # number covers several of the model's 32 px vision tokens.
            images = []
            for c in crops:
                scale = max(1.0, 224 / c.shape[0])
                c = cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                images.append(Image.fromarray(c[:, :, ::-1]))
            inputs = processor(text=[prompt] * len(images), images=images, return_tensors="pt",
                               padding=True).to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                                     output_scores=True, return_dict_in_generate=True)
            answer = out.sequences[:, inputs["input_ids"].shape[1]:]
            logp = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
            special = torch.tensor(processor.tokenizer.all_special_ids, device=answer.device)
            logp = torch.where(torch.isin(answer, special), 0.0, logp.float())  # count the answer's own tokens only
            texts = processor.batch_decode(answer, skip_special_tokens=True)
            return [(t.strip(), float(p)) for t, p in zip(texts, logp.sum(1).exp())]

        return read

    raise ValueError(f"unknown reader {name!r}")


def trusted_number(numbers: pd.Series, role: int, min_votes: int, min_share: float,
                   single_digits: bool = True) -> str | None:
    """The track's jersey number, if the readings agree enough.

    Florence-2 often drops a digit of a two-digit number (24 read as 2) and
    reads a front view without a number as '1'. So a two-digit number read at
    least min_votes times wins; single digits need one more vote, and '1'
    only counts for goalkeepers (outfield players do not wear it).
    """
    numbers = numbers.dropna()
    numbers = numbers[numbers != ""]
    two = numbers[numbers.str.len() == 2].value_counts()
    if len(two) and two.iloc[0] >= min_votes and two.iloc[0] / two.sum() >= min_share:
        return two.index[0]
    if not single_digits:
        return None
    one = numbers[numbers.str.len() == 1]
    if role != CLASSES.index("goalkeeper"):
        one = one[one != "1"]
    one = one.value_counts()
    if len(one) and one.iloc[0] >= min_votes + 1 and one.iloc[0] / one.sum() >= min_share:
        return one.index[0]
    return None
