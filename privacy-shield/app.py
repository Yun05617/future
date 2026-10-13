# -*- coding: utf-8 -*-
"""
Privacy Shield demo.

Main idea:
    Normal user sees daily computer screens.
    Unauthorized capture gets a random painting.
    The painting is not only a replacement: the real screen is embedded into
    the painting with LSB steganography and a password-derived permutation.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
import time
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk


DEFAULT_BITS = 4
AUTH_PASSWORD = "family123"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")
DAILY_IMAGE_NAMES = ("1", "2", "3", "4", "5")
ART_FILE_NAMES = (
    "1545671748-05e80f2c00710125b1a7bd06a1414c2d.jpg",
    "images.jpg",
    "Picasso_The_Weeping_Woman_Tate_identifier_T05010_10.jpg",
    "Vincent_Willem_van_Gogh_127.jpg",
    "cover_mona_style.png",
)


def key_seed(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def keyed_permutation(length: int, key: str) -> np.ndarray:
    rng = np.random.default_rng(key_seed(key))
    return rng.permutation(length)


def encode_image(
    cover_path: str | Path,
    secret_path: str | Path,
    output_path: str | Path,
    key: str,
    bits: int = DEFAULT_BITS,
) -> None:
    if not 1 <= bits <= 4:
        raise ValueError("bits must be between 1 and 4.")

    cover = load_rgb(cover_path)
    secret = load_rgb(secret_path).resize(cover.size, Image.Resampling.LANCZOS)

    cover_array = np.array(cover, dtype=np.uint8)
    secret_array = np.array(secret, dtype=np.uint8)

    payload = (secret_array >> (8 - bits)).reshape(-1)
    if key:
        payload = payload[keyed_permutation(payload.size, key)]

    high_bit_mask = 0xFF ^ ((1 << bits) - 1)
    clean_cover = cover_array.reshape(-1) & high_bit_mask
    stego_flat = clean_cover | payload
    stego_array = stego_flat.reshape(cover_array.shape).astype(np.uint8)

    Image.fromarray(stego_array, "RGB").save(output_path)


def decode_image(
    stego_path: str | Path,
    output_path: str | Path,
    key: str,
    bits: int = DEFAULT_BITS,
) -> None:
    if not 1 <= bits <= 4:
        raise ValueError("bits must be between 1 and 4.")

    stego = load_rgb(stego_path)
    stego_array = np.array(stego, dtype=np.uint8)

    payload = (stego_array.reshape(-1) & ((1 << bits) - 1)).astype(np.uint8)
    if key:
        order = keyed_permutation(payload.size, key)
        restored = np.empty_like(payload)
        restored[order] = payload
        payload = restored

    recovered_flat = payload << (8 - bits)
    recovered = recovered_flat.reshape(stego_array.shape).astype(np.uint8)
    Image.fromarray(recovered, "RGB").save(output_path)


def ssim_score(path_a: str | Path, path_b: str | Path) -> float:
    img_a = np.asarray(load_rgb(path_a), dtype=np.float64)
    img_b = np.asarray(load_rgb(path_b).resize((img_a.shape[1], img_a.shape[0])), dtype=np.float64)

    scores: list[float] = []
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    for channel in range(3):
        x = img_a[:, :, channel]
        y = img_b[:, :, channel]
        mean_x = x.mean()
        mean_y = y.mean()
        var_x = x.var()
        var_y = y.var()
        covariance = ((x - mean_x) * (y - mean_y)).mean()
        numerator = (2 * mean_x * mean_y + c1) * (2 * covariance + c2)
        denominator = (mean_x**2 + mean_y**2 + c1) * (var_x + var_y + c2)
        scores.append(float(numerator / denominator))

    return float(np.mean(scores))


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def output_dir() -> Path:
    folder = script_dir() / "outputs"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def find_image_by_stem(folder: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = folder / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def fit_photo(path: Path, size: tuple[int, int], bg: str = "#111827") -> ImageTk.PhotoImage:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, bg)
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return ImageTk.PhotoImage(canvas)


def save_display_copy(source: Path, output: Path) -> None:
    image = Image.open(source).convert("RGB")
    image.save(output)


def make_recording_strip(frames: list[Path], output: Path, title: str) -> None:
    tile_w, tile_h = 220, 140
    margin = 18
    title_h = 46
    gap = 12
    width = margin * 2 + tile_w * len(frames) + gap * max(0, len(frames) - 1)
    height = title_h + tile_h + margin * 2 + 24
    canvas = Image.new("RGB", (width, height), "#111827")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((margin, 16), title, fill="#f9fafb", font=font)

    for index, frame_path in enumerate(frames):
        image = Image.open(frame_path).convert("RGB")
        image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_w, tile_h), "#0b1220")
        x = (tile_w - image.width) // 2
        y = (tile_h - image.height) // 2
        tile.paste(image, (x, y))

        px = margin + index * (tile_w + gap)
        py = title_h + margin
        canvas.paste(tile, (px, py))
        draw.rectangle((px, py, px + tile_w, py + tile_h), outline="#334155", width=2)
        draw.text((px + 8, py + tile_h + 6), f"frame {index + 1}", fill="#cbd5e1", font=font)

    canvas.save(output)


class ScreenShieldApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.data_dir = output_dir()
        assets = script_dir() / "assets"
        self.daily_images = [
            path for name in DAILY_IMAGE_NAMES if (path := find_image_by_stem(assets / "daily", name))
        ]
        if not self.daily_images:
            self.daily_images = [assets / "samples" / "secret_fall_scene.png"]
        self.art_images = [assets / "paintings" / name for name in ART_FILE_NAMES if (assets / "paintings" / name).exists()]
        if not self.art_images:
            self.art_images = [assets / "samples" / "cover_mona_style.png"]

        self.current_index = 0
        self.protection_enabled = True
        self.authorized = False
        self.auto_playing = False
        self.last_stego_capture: Path | None = None
        self.last_decoded_capture: Path | None = None

        self.password_value = tk.StringVar()
        self.mode_value = tk.StringVar()
        self.capture_value = tk.StringVar(value="尚未擷取")
        self.log_value = tk.StringVar(
            value="系統待命：按下「模擬駭客截圖」，會用 LSB 隱寫把真實畫面藏進隨機名畫。"
        )

        self.user_photo: ImageTk.PhotoImage | None = None
        self.capture_photo: ImageTk.PhotoImage | None = None

        self.root.title("Privacy Shield 螢幕保護系統")
        self.root.geometry("1220x780")
        self.root.minsize(1060, 700)
        self.root.configure(bg="#edf2f7")

        self.build_ui()
        self.refresh_state()
        self.show_user_screen()

    def build_ui(self) -> None:
        top = tk.Frame(self.root, bg="#0f172a", height=92)
        top.pack(fill="x")
        top.pack_propagate(False)

        tk.Label(
            top,
            text="Privacy Shield 螢幕保護系統",
            bg="#0f172a",
            fg="#f8fafc",
            font=("Microsoft JhengHei UI", 22, "bold"),
        ).pack(anchor="w", padx=26, pady=(18, 2))
        tk.Label(
            top,
            text="保護開啟時，未授權截圖會得到隨機名畫；輸入密碼後才可還原或正常擷取。",
            bg="#0f172a",
            fg="#cbd5e1",
            font=("Microsoft JhengHei UI", 11),
        ).pack(anchor="w", padx=28)

        body = tk.Frame(self.root, bg="#edf2f7")
        body.pack(fill="both", expand=True, padx=22, pady=20)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        left = tk.Frame(body, bg="#ffffff", highlightbackground="#d0d7de", highlightthickness=1)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=1)

        tk.Label(
            left,
            text="正常使用者畫面",
            bg="#ffffff",
            fg="#111827",
            font=("Microsoft JhengHei UI", 16, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=18, pady=(16, 2))
        tk.Label(
            left,
            text="使用 1-5 圖片模擬日常電腦操作畫面。",
            bg="#ffffff",
            fg="#64748b",
            font=("Microsoft JhengHei UI", 10),
        ).grid(row=1, column=0, sticky="w", padx=18, pady=(0, 10))

        self.user_screen = tk.Label(
            left,
            bg="#111827",
            fg="#e5e7eb",
            text="找不到 1-5 圖片",
            font=("Microsoft JhengHei UI", 14),
        )
        self.user_screen.grid(row=2, column=0, sticky="nsew", padx=18, pady=(0, 12))

        nav = tk.Frame(left, bg="#ffffff")
        nav.grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 18))
        nav.columnconfigure((0, 1, 2), weight=1)
        self.add_button(nav, "上一張", self.previous_screen, "#334155", 0, 0)
        self.add_button(nav, "下一張", self.next_screen, "#334155", 0, 1)
        self.add_button(nav, "日常輪播", self.toggle_auto_play, "#176b87", 0, 2)

        right = tk.Frame(body, bg="#edf2f7")
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        control = tk.Frame(right, bg="#ffffff", highlightbackground="#d0d7de", highlightthickness=1)
        control.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        control.columnconfigure(0, weight=1)

        tk.Label(
            control,
            text="系統控制台",
            bg="#ffffff",
            fg="#111827",
            font=("Microsoft JhengHei UI", 16, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=18, pady=(16, 8))

        self.mode_badge = tk.Label(
            control,
            textvariable=self.mode_value,
            bg="#dcfce7",
            fg="#166534",
            font=("Microsoft JhengHei UI", 11, "bold"),
            padx=12,
            pady=8,
        )
        self.mode_badge.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 10))

        password_row = tk.Frame(control, bg="#ffffff")
        password_row.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 10))
        password_row.columnconfigure(0, weight=1)
        tk.Entry(
            password_row,
            textvariable=self.password_value,
            show="*",
            relief="flat",
            bg="#f8fafc",
            fg="#111827",
            insertbackground="#111827",
            font=("Consolas", 13),
        ).grid(row=0, column=0, sticky="ew", ipady=9)
        tk.Button(
            password_row,
            text="解鎖",
            command=self.unlock,
            relief="flat",
            bg="#dc2626",
            fg="#ffffff",
            activebackground="#991b1b",
            activeforeground="#ffffff",
            font=("Microsoft JhengHei UI", 11, "bold"),
            cursor="hand2",
        ).grid(row=0, column=1, padx=(8, 0), ipady=6)

        tk.Label(
            control,
            text="展示密碼：family123",
            bg="#ffffff",
            fg="#64748b",
            font=("Microsoft JhengHei UI", 9),
        ).grid(row=3, column=0, sticky="w", padx=20, pady=(0, 10))

        actions = tk.Frame(control, bg="#ffffff")
        actions.grid(row=4, column=0, sticky="ew", padx=18, pady=(0, 16))
        actions.columnconfigure((0, 1), weight=1)
        self.add_button(actions, "開啟 / 關閉保護", self.toggle_protection, "#0f766e", 0, 0)
        self.add_button(actions, "重新鎖定", self.lock, "#475569", 0, 1)
        self.add_button(actions, "模擬駭客截圖", self.simulate_capture, "#b45309", 1, 0)
        self.add_button(actions, "模擬駭客錄影", self.simulate_recording, "#7c3aed", 1, 1)

        capture = tk.Frame(right, bg="#ffffff", highlightbackground="#d0d7de", highlightthickness=1)
        capture.grid(row=1, column=0, sticky="nsew")
        capture.columnconfigure(0, weight=1)
        capture.rowconfigure(2, weight=1)

        tk.Label(
            capture,
            text="擷取結果展示",
            bg="#ffffff",
            fg="#111827",
            font=("Microsoft JhengHei UI", 16, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=18, pady=(16, 2))
        tk.Label(
            capture,
            textvariable=self.capture_value,
            bg="#ffffff",
            fg="#64748b",
            font=("Microsoft JhengHei UI", 10),
        ).grid(row=1, column=0, sticky="w", padx=18, pady=(0, 10))

        self.capture_screen = tk.Label(
            capture,
            bg="#111827",
            fg="#e5e7eb",
            text="等待擷取",
            font=("Microsoft JhengHei UI", 14),
        )
        self.capture_screen.grid(row=2, column=0, sticky="nsew", padx=18, pady=(0, 12))

        log = tk.Label(
            capture,
            textvariable=self.log_value,
            wraplength=420,
            justify="left",
            bg="#f8fafc",
            fg="#111827",
            font=("Microsoft JhengHei UI", 10),
            padx=12,
            pady=10,
        )
        log.grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 18))

    def add_button(self, parent: tk.Frame, text: str, command, color: str, row: int, column: int) -> None:
        tk.Button(
            parent,
            text=text,
            command=command,
            relief="flat",
            bg=color,
            fg="#ffffff",
            activebackground="#0f172a",
            activeforeground="#ffffff",
            font=("Microsoft JhengHei UI", 10, "bold"),
            cursor="hand2",
        ).grid(row=row, column=column, sticky="ew", padx=5, pady=5, ipady=8)

    def refresh_state(self) -> None:
        if self.authorized:
            self.mode_value.set("授權模式：可正常截圖 / 錄影，也可解碼前一次名畫藏圖")
            self.mode_badge.config(bg="#dbeafe", fg="#1d4ed8")
        elif self.protection_enabled:
            self.mode_value.set("保護中：駭客截圖會變成 LSB 藏圖名畫")
            self.mode_badge.config(bg="#dcfce7", fg="#166534")
        else:
            self.mode_value.set("保護關閉：擷取會看到真實畫面")
            self.mode_badge.config(bg="#fee2e2", fg="#991b1b")

    def current_daily_image(self) -> Path | None:
        if not self.daily_images:
            return None
        return self.daily_images[self.current_index % len(self.daily_images)]

    def show_user_screen(self) -> None:
        path = self.current_daily_image()
        if path is None:
            return
        self.user_photo = fit_photo(path, (700, 430))
        self.user_screen.config(image=self.user_photo, text="")

    def show_capture(self, path: Path) -> None:
        self.capture_photo = fit_photo(path, (470, 300))
        self.capture_screen.config(image=self.capture_photo, text="")
        self.capture_value.set(path.name)

    def previous_screen(self) -> None:
        if not self.daily_images:
            return
        self.current_index = (self.current_index - 1) % len(self.daily_images)
        self.show_user_screen()

    def next_screen(self) -> None:
        if not self.daily_images:
            return
        self.current_index = (self.current_index + 1) % len(self.daily_images)
        self.show_user_screen()

    def toggle_auto_play(self) -> None:
        self.auto_playing = not self.auto_playing
        self.log_value.set("日常畫面輪播已啟動。" if self.auto_playing else "日常畫面輪播已停止。")
        if self.auto_playing:
            self.auto_step()

    def auto_step(self) -> None:
        if not self.auto_playing:
            return
        self.next_screen()
        self.root.after(1400, self.auto_step)

    def toggle_protection(self) -> None:
        self.protection_enabled = not self.protection_enabled
        if not self.protection_enabled:
            self.authorized = False
        self.refresh_state()

    def unlock(self) -> None:
        if self.password_value.get().strip() != AUTH_PASSWORD:
            self.authorized = False
            self.refresh_state()
            messagebox.showwarning("密碼錯誤", "密碼不正確，系統仍維持保護狀態。")
            return

        self.authorized = True
        self.refresh_state()

        if self.last_stego_capture and self.last_stego_capture.exists():
            decoded = self.data_dir / "authorized_decoded_from_hacker_capture.png"
            decode_image(self.last_stego_capture, decoded, AUTH_PASSWORD)
            self.last_decoded_capture = decoded
            self.show_capture(decoded)
            self.log_value.set(
                "密碼正確：已用同一組金鑰從駭客拿到的名畫藏圖中解出真實畫面。\n"
                f"解碼輸出：{decoded.name}"
            )
        else:
            self.log_value.set("密碼正確：現在截圖與錄影會顯示真實畫面。")

    def lock(self) -> None:
        self.authorized = False
        self.password_value.set("")
        self.refresh_state()
        self.log_value.set("系統已重新鎖定，下一次未授權截圖會再次得到名畫藏圖。")

    def simulate_capture(self) -> None:
        real_screen = self.current_daily_image()
        if real_screen is None:
            messagebox.showerror("缺少圖片", "找不到 1-5 日常畫面。")
            return

        if self.protection_enabled and not self.authorized:
            if not self.art_images:
                messagebox.showerror("缺少名畫", "找不到名畫素材，請至少保留一張名畫圖片。")
                return

            art = random.choice(self.art_images)
            output = self.data_dir / "hacker_capture_decoy.png"
            proof = self.data_dir / "authorized_decoded_from_hacker_capture.png"

            start = time.perf_counter()
            encode_image(art, real_screen, output, AUTH_PASSWORD, DEFAULT_BITS)
            decode_image(output, proof, AUTH_PASSWORD, DEFAULT_BITS)
            elapsed = time.perf_counter() - start

            self.last_stego_capture = output
            self.last_decoded_capture = proof
            self.show_capture(output)
            self.log_value.set(
                "技術：LSB 最低有效位元隱寫 + SHA-256 金鑰亂序。\n"
                "流程：把目前真實螢幕藏進隨機名畫的 RGB 低位元，所以駭客表面上只看到名畫。\n"
                f"名畫外觀：{art.name}\n"
                f"真實畫面：{real_screen.name}\n"
                f"駭客取得：{output.name}\n"
                f"授權解碼證明：{proof.name}\n"
                f"處理時間：{elapsed:.4f} 秒"
            )
            return

        output = self.data_dir / "authorized_real_capture.png"
        save_display_copy(real_screen, output)
        self.show_capture(output)
        self.log_value.set(
            "目前是授權模式或保護關閉，因此截圖結果是正常真實畫面。\n"
            f"輸出檔案：{output.name}"
        )

    def simulate_recording(self) -> None:
        if not self.daily_images:
            messagebox.showerror("缺少圖片", "找不到 1-5 日常畫面。")
            return

        if self.protection_enabled and not self.authorized and self.art_images:
            frame_dir = self.data_dir / "hacker_recording_stego_frames"
            frame_dir.mkdir(exist_ok=True)
            frames: list[Path] = []

            start = time.perf_counter()
            for index, real_frame in enumerate(self.daily_images, start=1):
                art = random.choice(self.art_images)
                frame_output = frame_dir / f"stego_frame_{index}.png"
                encode_image(art, real_frame, frame_output, AUTH_PASSWORD, DEFAULT_BITS)
                frames.append(frame_output)
            output = self.data_dir / "hacker_recording_decoy.png"
            make_recording_strip(frames, output, "Hacker recording result: LSB stego paintings")
            elapsed = time.perf_counter() - start

            self.show_capture(output)
            self.log_value.set(
                "錄影模擬也使用 LSB 隱寫：每一格真實畫面都藏進一張名畫影格。\n"
                f"輸出檔案：{output.name}\n"
                f"處理時間：{elapsed:.4f} 秒"
            )
            return

        output = self.data_dir / "authorized_recording_real.png"
        make_recording_strip(self.daily_images, output, "Authorized recording result: real daily screen")
        self.show_capture(output)
        self.log_value.set(
            "授權狀態下錄影成功，分鏡圖顯示真實使用情境。\n"
            f"輸出檔案：{output.name}"
        )


def launch_gui() -> None:
    root = tk.Tk()
    ScreenShieldApp(root)
    root.mainloop()


def run_demo(output_dir: str | Path, key: str, bits: int) -> None:
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    samples = script_dir() / "assets" / "samples"
    cover = samples / "cover_mona_style.png"
    secret = samples / "secret_fall_scene.png"
    stego = folder / "hacker_intercepted_image.png"
    recovered = folder / "authorized_recovered_image.png"

    if not cover.exists() or not secret.exists():
        raise FileNotFoundError("demo needs cover_mona_style.png and secret_fall_scene.png")

    start = time.perf_counter()
    encode_image(cover, secret, stego, key, bits)
    decode_image(stego, recovered, key, bits)
    elapsed = time.perf_counter() - start

    print("Demo files created:")
    print(f"  Cover image:      {cover}")
    print(f"  Secret image:     {secret}")
    print(f"  Intercepted file: {stego}")
    print(f"  Recovered file:   {recovered}")
    print()
    print(f"Cover vs stego SSIM:       {ssim_score(cover, stego):.4f}")
    print(f"Secret vs recovered SSIM:  {ssim_score(secret, recovered):.4f}")
    print(f"Encode + decode time:      {elapsed:.4f} seconds")


def main() -> None:
    if len(sys.argv) == 1:
        launch_gui()
        return

    parser = argparse.ArgumentParser(description="Privacy Shield steganography system")
    sub = parser.add_subparsers(dest="command", required=True)

    encode = sub.add_parser("encode", help="hide a private image inside a cover image")
    encode.add_argument("--cover", required=True)
    encode.add_argument("--secret", required=True)
    encode.add_argument("--output", required=True)
    encode.add_argument("--key", required=True)
    encode.add_argument("--bits", type=int, default=DEFAULT_BITS)

    decode = sub.add_parser("decode", help="recover a hidden private image")
    decode.add_argument("--stego", required=True)
    decode.add_argument("--output", required=True)
    decode.add_argument("--key", required=True)
    decode.add_argument("--bits", type=int, default=DEFAULT_BITS)

    demo = sub.add_parser("demo", help="create a full steganography demonstration")
    demo.add_argument("--output-dir", default=str(script_dir() / "outputs"))
    demo.add_argument("--key", required=True)
    demo.add_argument("--bits", type=int, default=DEFAULT_BITS)

    metrics = sub.add_parser("metrics", help="calculate project evaluation scores")
    metrics.add_argument("--cover", required=True)
    metrics.add_argument("--stego", required=True)
    metrics.add_argument("--secret", required=True)
    metrics.add_argument("--recovered", required=True)

    args = parser.parse_args()

    if args.command == "encode":
        encode_image(args.cover, args.secret, args.output, args.key, args.bits)
        print(f"Saved stego image to {args.output}")
    elif args.command == "decode":
        decode_image(args.stego, args.output, args.key, args.bits)
        print(f"Saved recovered image to {args.output}")
    elif args.command == "demo":
        run_demo(args.output_dir, args.key, args.bits)
    elif args.command == "metrics":
        print(f"Cover vs stego SSIM:       {ssim_score(args.cover, args.stego):.4f}")
        print(f"Secret vs recovered SSIM:  {ssim_score(args.secret, args.recovered):.4f}")


if __name__ == "__main__":
    main()
