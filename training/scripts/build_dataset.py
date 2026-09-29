#!/usr/bin/env python3
# 学習データ: 27書風のフォント(download_fonts.py で取得)で、各フォントが持っている漢字をすべて1字ずつ描いた画像を作る。
# 字の数はフォントによって違う(標準的なフォントで約6,700字、多いもので約14,200字)。
#
# 描く字の範囲(フォントの文字の対応表にある字のうち):
# - 入れる: CJK統合漢字(基本の範囲と拡張A〜J)
#   - 常用漢字・人名用漢字などの普通の漢字、国字(峠、榊など)、旧字体の大部分(國、學など)はここに入っている
# - 入れる: CJK互換漢字の範囲にある、統合漢字と重複しない漢字(「﨑」などの12字)
# - 入れる: CJK互換漢字のうち、そのフォントで描いた形が、対応する統合漢字と違うもの(旧字形の「海」「侮」など)
#   - Unicodeの上では統合漢字と同じ字の重複だが、フォントでは形の違う字として描かれる
# - 入れない: CJK互換漢字のうち、対応する統合漢字と同じ形に描かれるもの(ただの重複)
# - 入れない: 部首の記号(康熙部首・CJK部首補助)、筆画の記号、かな、記号(〇・々・〆を含む)、西夏文字などの別の文字の体系
#   (どれも上の範囲の外にあるので、範囲で選ぶだけで入らない)
# - 入れない: 異体字の選択子(IVS)で呼び出す字形違い(字の後ろに選択子を付けた並びで、1つの符号位置ではないため)
# - 入れない: 中身が空の字と、そのフォントの豆腐(.notdef)と同じ形になる字
#
# 字はフォント本来の大きさで描き(書風ごとの字の大きさの癖を残す)、インクの外接の枠の中心を画像の中心に合わせる。
# 画像はインクの濃さ(0=紙〜255=インク)で保存する。黒地に白の字として見える向きで、学習の増強(位置ずれ・切り抜き)が
# 空いた場所を0で埋めても、背景と同じ色になる
import argparse
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

from download_fonts import FONTS, PinnedFont, font_path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GLYPH_SIZE_RATIO = 0.86  # 字の大きさ(フォントのemの大きさ)を画像の大きさに対してこの割合にする(はみ出す書風がないことを確認した値)
DEFAULT_WEIGHT = 400  # 可変フォントは標準の太さで描く
INK_THRESHOLD = 0.5  # インクの外接の枠を求めるときに、インクとみなす濃さ
CANVAS_SCALE = 2  # 字がはみ出さずに描けるよう、画像のこの倍の大きさのキャンバスに描いてから切り出す
UNIFIED_IDEOGRAPH_RANGES = (  # CJK統合漢字
    (0x4E00, 0x9FFF),  # 基本
    (0x3400, 0x4DBF),  # 拡張A
    (0x20000, 0x2A6DF),  # 拡張B
    (0x2A700, 0x2B73F),  # 拡張C
    (0x2B740, 0x2B81F),  # 拡張D
    (0x2B820, 0x2CEAF),  # 拡張E
    (0x2CEB0, 0x2EBEF),  # 拡張F
    (0x2EBF0, 0x2EE5F),  # 拡張I
    (0x30000, 0x3134F),  # 拡張G
    (0x31350, 0x323AF),  # 拡張H
    (0x323B0, 0x3347F),  # 拡張J
)
# CJK互換漢字。大部分は統合漢字と同じ字の重複(Unicodeの正規化(NFC)で統合漢字に置き換わる)。置き換わらない12字
# (U+FA0E・FA0F・FA11・FA13・FA14・FA1F・FA21・FA23・FA24・FA27・FA28・FA29)は、統合漢字と重複しない漢字
COMPATIBILITY_IDEOGRAPH_RANGES = (
    (0xF900, 0xFAFF),  # CJK互換漢字
    (0x2F800, 0x2FA1F),  # CJK互換漢字補助
)
SKIP_EMPTY, SKIP_TOFU, SKIP_DUPLICATE = "empty", "same as tofu", "compatibility same as unified"
NOT_A_CHARACTER = "\uffff"  # どのフォントも字形を持たない符号位置。描くとそのフォントの豆腐(.notdef)になる


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=64)
    return parser.parse_args()


def _in_ranges(codepoint: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= codepoint <= end for start, end in ranges)


def _kanji_codepoints(path: Path) -> tuple[list[int], list[int]]:
    # 文字の対応表にある字だけを使う(対応表にない字を描くと豆腐になる)
    cmap = TTFont(path).getBestCmap()
    unified = sorted(codepoint for codepoint in cmap if _in_ranges(codepoint, UNIFIED_IDEOGRAPH_RANGES))
    compatibility = sorted(codepoint for codepoint in cmap if _in_ranges(codepoint, COMPATIBILITY_IDEOGRAPH_RANGES))
    return unified, compatibility


def _load_font(path: Path, resolution: int) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(str(path), int(resolution * GLYPH_SIZE_RATIO))
    if "[" in path.name:  # 可変フォント(ファイル名に軸の名前が入る。例: NotoSansJP[wght].ttf)
        font.set_variation_by_axes([DEFAULT_WEIGHT])
    return font


def _render_glyph(font: ImageFont.FreeTypeFont, character: str, resolution: int) -> np.ndarray | None:
    canvas_size = resolution * CANVAS_SCALE
    image = Image.new("L", (canvas_size, canvas_size), 255)
    ImageDraw.Draw(image).text((canvas_size / 2, canvas_size / 2), character, font=font, fill=0, anchor="mm")
    ink = 1 - np.asarray(image, dtype=np.float32) / 255
    ink_rows, ink_cols = np.nonzero(ink > INK_THRESHOLD)
    if len(ink_cols) == 0:
        return None
    center_x, center_y = (ink_cols.min() + ink_cols.max() + 1) / 2, (ink_rows.min() + ink_rows.max() + 1) / 2
    left, top = int(round(center_x - resolution / 2)), int(round(center_y - resolution / 2))
    glyph = np.zeros((resolution, resolution), np.float32)
    source_left, source_top = max(left, 0), max(top, 0)
    source_right, source_bottom = min(left + resolution, canvas_size), min(top + resolution, canvas_size)
    glyph[source_top - top:source_bottom - top, source_left - left:source_right - left] = ink[source_top:source_bottom, source_left:source_right]
    return glyph


def _unified_counterpart(codepoint: int) -> int | None:
    # 互換漢字が正規化で置き換わる統合漢字のコードポイント。置き換わらない(統合漢字と重複しない漢字の)場合はNone
    normalized = unicodedata.normalize("NFC", chr(codepoint))
    return ord(normalized) if normalized != chr(codepoint) else None


def _skip_reason(codepoint: int, glyph: np.ndarray | None, tofu: np.ndarray | None, rendered: dict[int, np.ndarray]) -> str | None:
    if glyph is None:
        return SKIP_EMPTY
    if tofu is not None and np.array_equal(glyph, tofu):
        return SKIP_TOFU
    if _in_ranges(codepoint, COMPATIBILITY_IDEOGRAPH_RANGES):
        counterpart = _unified_counterpart(codepoint)
        if counterpart is not None and counterpart in rendered and np.array_equal(glyph, rendered[counterpart]):
            return SKIP_DUPLICATE
    return None


def _render_font(pinned: PinnedFont, resolution: int) -> dict[int, np.ndarray]:
    # 返り値の画像は0〜1(1=インク)
    path = font_path(pinned)
    unified, compatibility = _kanji_codepoints(path)
    font = _load_font(path, resolution)
    tofu = _render_glyph(font, NOT_A_CHARACTER, resolution)
    rendered: dict[int, np.ndarray] = {}
    skipped: Counter[str] = Counter()
    for codepoint in unified + compatibility:  # 互換漢字は、対応する統合漢字を描いた後に比べる
        glyph = _render_glyph(font, chr(codepoint), resolution)
        reason = _skip_reason(codepoint, glyph, tofu, rendered)
        if reason is not None:
            skipped[reason] += 1
            continue
        rendered[codepoint] = glyph
    kept_compatibility = len(rendered.keys() & set(compatibility))
    skipped_text = ", ".join(f"{reason} {skipped[reason]}" for reason in (SKIP_EMPTY, SKIP_TOFU, SKIP_DUPLICATE))
    print(f"{pinned.family}: {len(rendered)} glyphs (compatibility ideographs kept {kept_compatibility}; skipped: {skipped_text})", flush=True)
    return rendered


def _build_dataset(resolution: int) -> dict[str, np.ndarray]:
    images, codepoints, styles = [], [], []
    for style_index, pinned in enumerate(FONTS):
        for codepoint, glyph in _render_font(pinned, resolution).items():
            images.append((glyph * 255).astype(np.uint8))
            codepoints.append(codepoint)
            styles.append(style_index)

    kanji = sorted(set(codepoints))
    index_of = {codepoint: index for index, codepoint in enumerate(kanji)}
    return {
        "images": np.stack(images),  # (字の数, 解像度, 解像度) uint8。0=紙〜255=インク
        "labels": np.array([index_of[codepoint] for codepoint in codepoints]),  # 何番目の漢字か(kanjiの添字)
        "styles": np.array(styles),  # 何番目の書風か(style_names の添字)
        "kanji": np.array(kanji),  # 漢字のコードポイント
        "style_names": np.array([pinned.family for pinned in FONTS]),
    }


def main() -> None:
    args = _parse_args()
    dataset = _build_dataset(args.resolution)
    output_path = DATA_DIR / f"glyphs_{args.resolution}.npz"
    np.savez_compressed(output_path, **dataset)
    print(f"Saved {len(dataset['images'])} glyphs ({len(dataset['kanji'])} kanji, {len(FONTS)} styles) to {output_path}")


if __name__ == "__main__":
    main()
