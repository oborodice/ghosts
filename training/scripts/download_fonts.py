#!/usr/bin/env python3
# 学習データ(フォントで描いた漢字の画像)に使う27書風のフォントを、Google Fontsのリポジトリ(google/fonts)から取得する。
# 取得元はコミットを固定し、各フォントのファイルはgitのblobの識別子で照合するので、いつ実行しても同じファイルがそろう。
# 各ファミリーのライセンスのファイル(OFLはOFL.txt、ApacheはLICENSE.txt)も同じコミットから取得して、フォントの隣に置く
import hashlib
import urllib.parse
import urllib.request
from pathlib import Path
from typing import NamedTuple


class PinnedFont(NamedTuple):
    family: str
    path: str  # リポジトリの中のパス(先頭のディレクトリがライセンスの種類: ofl / apache)
    git_blob_sha: str


GOOGLE_FONTS_COMMIT = "23e54b51ddffbc7713c583748e3bd86f62b1fa4a"
RAW_URL = "https://raw.githubusercontent.com/google/fonts/{commit}/{path}"
FONT_DIR = Path(__file__).resolve().parent.parent / "data" / "fonts"
LICENSE_FILE_NAMES = {"ofl": "OFL.txt", "apache": "LICENSE.txt"}

# 日本語に対応するGoogle Fontsのファミリーのうち、KanjiVGの漢字(基本のファイル)の95%以上を持つものから、
# 低い解像度(64〜128px)で字の構造が崩れるもの(ドット、二重線、極太など)と、見た目がほぼ同じ組の片方を除いた27書風。
# 各ファミリーで標準の太さのファイル(なければ可変フォント。描くときに太さを400に固定する)を使う
FONTS = (
    PinnedFont("BIZ UDGothic", "ofl/bizudgothic/BIZUDGothic-Regular.ttf", "dd215603417e88b118d9e69fc1296e50bc764d34"),
    PinnedFont("BIZ UDMincho", "ofl/bizudmincho/BIZUDMincho-Regular.ttf", "b5a3f4cc698be74cd12b13219a3a5bddc8eb95cd"),
    PinnedFont("Hachi Maru Pop", "ofl/hachimarupop/HachiMaruPop-Regular.ttf", "1339bf206111860f1d3867146aa53662bdbd1919"),
    PinnedFont("Hina Mincho", "ofl/hinamincho/HinaMincho-Regular.ttf", "839226e23ab32fd5ca2206f32901e0e53d6d2bb6"),
    PinnedFont("IBM Plex Sans JP", "ofl/ibmplexsansjp/IBMPlexSansJP-Regular.ttf", "f7ea368508d862d9c2fca78359d8fa61ada51ffa"),
    PinnedFont("Kaisei Decol", "ofl/kaiseidecol/KaiseiDecol-Regular.ttf", "d764a7456778af76d42dc735637654701d943199"),
    PinnedFont("Kiwi Maru", "ofl/kiwimaru/KiwiMaru-Regular.ttf", "9071c06bb0ccb1496c7bb2fdc8ffc44f947dc0d8"),
    PinnedFont("Klee One", "ofl/kleeone/KleeOne-Regular.ttf", "106bd3dbcfb24f7c971b761d63223b37104d4b6f"),
    PinnedFont("Kosugi", "apache/kosugi/Kosugi-Regular.ttf", "56242e88aaf4aa912f28a93143f277824e356215"),
    PinnedFont("Kosugi Maru", "apache/kosugimaru/KosugiMaru-Regular.ttf", "bc2c935463067d87087791ad22ed10b444fc8ceb"),
    PinnedFont("LINE Seed JP", "ofl/lineseedjp/LINESeedJP-Regular.ttf", "be2b8b991fdb478eb9b76c5dbd2387747831e98c"),
    PinnedFont("Mochiy Pop One", "ofl/mochiypopone/MochiyPopOne-Regular.ttf", "f239f79ceabc9badc25538ab578b81fcc1f116fa"),
    PinnedFont("New Tegomin", "ofl/newtegomin/NewTegomin-Regular.ttf", "adfcf6d90008b8d245f90cf38464bf643133b254"),
    PinnedFont("Noto Sans JP", "ofl/notosansjp/NotoSansJP[wght].ttf", "cdd8f083c1f5928ff3361f8cda4d3fc9462cbe89"),
    PinnedFont("Noto Serif JP", "ofl/notoserifjp/NotoSerifJP[wght].ttf", "bd98d809e02880e32f20f1f2a91ef97854f52f57"),
    PinnedFont("RocknRoll One", "ofl/rocknrollone/RocknRollOne-Regular.ttf", "95f92adda0d50a5255a552e528f4aff05e414b6c"),
    PinnedFont("Shippori Antique", "ofl/shipporiantique/ShipporiAntique-Regular.ttf", "1c2489a2cf1e359e54f6e657842f4be27902c8e5"),
    PinnedFont("Shippori Mincho", "ofl/shipporimincho/ShipporiMincho-Regular.ttf", "cfc25ceb7fd744599d6bde767cb2a01e73b510a4"),
    PinnedFont("Stick", "ofl/stick/Stick-Regular.ttf", "dd521676508b1b9f0457d27bc687ff87d71f8624"),
    PinnedFont("Yomogi", "ofl/yomogi/Yomogi-Regular.ttf", "9ed46e0cdfe749a01ee2bee03852bf39cfcf7c65"),
    PinnedFont("Yuji Syuku", "ofl/yujisyuku/YujiSyuku-Regular.ttf", "8a7db48fbf431ea97f14bedf6a4f5472347ab8a0"),
    PinnedFont("Yusei Magic", "ofl/yuseimagic/YuseiMagic-Regular.ttf", "76e228f8f4cb429e1dc843f461fece77dddb8a49"),
    PinnedFont("Zen Antique", "ofl/zenantique/ZenAntique-Regular.ttf", "6179ef193e65ecb4be3670c3de0559707c2cc46e"),
    PinnedFont("Zen Kaku Gothic New", "ofl/zenkakugothicnew/ZenKakuGothicNew-Regular.ttf", "9940304caecd27df011ef6d15af9e187713967be"),
    PinnedFont("Zen Kurenaido", "ofl/zenkurenaido/ZenKurenaido-Regular.ttf", "306f69461d53039c2595f75891dd8a60be2dd65e"),
    PinnedFont("Zen Maru Gothic", "ofl/zenmarugothic/ZenMaruGothic-Regular.ttf", "c622402a1acf00c3481334de56439b3703cdcbb6"),
    PinnedFont("Zen Old Mincho", "ofl/zenoldmincho/ZenOldMincho-Regular.ttf", "0086880c10a9925338a87b1b1e199e1ef6e9ca39"),
)


def font_path(font: PinnedFont) -> Path:
    return FONT_DIR / Path(font.path).name


def _git_blob_sha(data: bytes) -> str:
    # gitがファイルの中身に付ける識別子(ヘッダー "blob <バイト数>\0" を付けたSHA-1)
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _fetch_repository_file(path: str) -> bytes:
    # 可変フォントのファイル名の角かっこ(例: NotoSansJP[wght].ttf)などを、URLの中で使える形に符号化する
    with urllib.request.urlopen(RAW_URL.format(commit=GOOGLE_FONTS_COMMIT, path=urllib.parse.quote(path))) as response:
        return response.read()


def _download_font(font: PinnedFont) -> bool:
    # 返り値は、取得したかどうか(照合済みのファイルがあれば取得しない)
    destination = font_path(font)
    if destination.exists() and _git_blob_sha(destination.read_bytes()) == font.git_blob_sha:
        return False
    data = _fetch_repository_file(font.path)
    actual_sha = _git_blob_sha(data)
    if actual_sha != font.git_blob_sha:
        raise RuntimeError(f"{font.family}: git blob sha mismatch (expected {font.git_blob_sha}, got {actual_sha})")
    destination.write_bytes(data)
    return True


def _download_license(font: PinnedFont) -> None:
    family_dir = Path(font.path).parent
    license_name = LICENSE_FILE_NAMES[family_dir.parts[0]]
    destination = FONT_DIR / f"{family_dir.name}_{license_name}"
    if not destination.exists():
        destination.write_bytes(_fetch_repository_file(f"{family_dir}/{license_name}"))


def main() -> None:
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for font in FONTS:
        fetched = _download_font(font)
        _download_license(font)
        downloaded += fetched
        print(f"{'downloaded' if fetched else 'verified  '} {font.family}: {font_path(font).name}")
    print(f"Done: {len(FONTS)} fonts in {FONT_DIR} ({downloaded} downloaded, {len(FONTS) - downloaded} already present)")


if __name__ == "__main__":
    main()
