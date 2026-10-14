#!/usr/bin/env python3
"""Sanity-check the official concept tflite on x86: feed a cat photo under
several input-normalization variants and print the top classes. This tells us
the input semantics the RKNN conversion must mirror (mean/std in rknn.config).
Usage: tflite-sanity.py <model.tflite> <labels.txt> <image...>"""
import sys

import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter


def main():
    tflite, labels_path = sys.argv[1], sys.argv[2]
    imgs = sys.argv[3:]
    labels = open(labels_path, encoding="utf-8", errors="replace").read().splitlines()

    interp = Interpreter(model_path=tflite)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    print("input:", inp["shape"], inp["dtype"], "quant:", inp.get("quantization"))
    print("output:", out["shape"], out["dtype"], "quant:", out.get("quantization"))
    h, w = inp["shape"][1], inp["shape"][2]

    img = Image.open(imgs[0]).convert("RGB").resize((w, h), Image.BILINEAR)
    rgb = np.asarray(img, dtype=np.float32)
    bgr = rgb[:, :, ::-1].copy()
    variants = {
        "rgb_raw0-255": rgb,
        "bgr_raw0-255": bgr,
        "rgb_div255": rgb / 255.0,
        "bgr_div255": bgr / 255.0,
        "rgb_pm1": rgb / 127.5 - 1.0,
        "bgr_pm1": bgr / 127.5 - 1.0,
        "rgb_mean127.5": rgb - 127.5,
        "rgb_torch": (rgb / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225],
    }
    for name, x in variants.items():
        data = np.clip(x, -128 if inp["dtype"] == np.int8 else 0, 255).astype(
            np.int8 if inp["dtype"] == np.int8 else (np.uint8 if inp["dtype"] == np.uint8 else np.float32))
        if inp["dtype"] in (np.int8, np.uint8):
            z, s = (inp.get("quantization") or (0.0, 1.0))
            if s:
                data = np.clip(np.round(x / s + z), 0 if inp["dtype"] == np.uint8 else -128,
                               255 if inp["dtype"] == np.uint8 else 127).astype(inp["dtype"])
        interp.set_tensor(inp["index"], data[np.newaxis])
        interp.invoke()
        y = interp.get_tensor(out["index"])[0].astype(np.float32)
        top = np.argsort(-y)[:4]
        print("%-14s -> %s" % (name, " ".join("%d:%s=%.3f" % (k, labels[k] if k < len(labels) else "?", y[k]) for k in top)))


if __name__ == "__main__":
    main()
