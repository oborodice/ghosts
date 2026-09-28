# 生成器が作ったインクの画像を、GPUのシェーダー(shaders/)でフィルタをかけて画面に描く部品。
# 輪郭のやわらげ → くっきりさせる(境界を溶かすノイズつき) → 色づけ → 発光 → 背景のノイズ。見た目の値はこのファイルの定数。
# フィルタをGPUでかけるのは、CPUを生成器に専念させるため(Raspberry Pi 5 では、解像度を上げるとCPUに余裕がなくなる見込み)
import math
import sys
from pathlib import Path

import numpy as np
import pygame
from OpenGL import GL

IMAGE_SCREEN_FRACTION = 0.7  # 字の画像を、画面の短い辺のこの割合の大きさで中央に描く(周りは余白)
# 字の輪郭のやわらげ: 拡大する前の画像の画素の単位で、ガウスのぼかしをかけてから拡大する(64pxの格子のギザギザを目立たなくする)
SOFTEN_SIGMA = 0.8
GAUSSIAN_RADIUS_IN_SIGMAS = 3  # ぼかしで足し合わせる範囲(σの何倍までか。それより外の重みは無視できるほど小さい)
# 輪郭をくっきりさせる: やわらげたインクの濃さを、INK_EDGE_CENTER の前後 ±INK_EDGE_WIDTH の範囲でなめらかに0から1へ切り替える
# (やわらげでギザギザを消したあとに縁だけを締め、ギザギザもぼんやりもない輪郭にする)
INK_EDGE_CENTER = 0.5
INK_EDGE_WIDTH = 0.15
# ノイズ: 隣どうしをなめらかにつないだ揺らぎの模様を、少しずつ次の模様へ移り変わらせる。同じ1つの模様で、字の境界を背景に
# 溶け込ませ(くっきりさせる前の濃さに足し、境界の近くだけを字の色と背景の間で崩す)、背景に粒を重ねる
NOISE_GRAIN = 1.5  # 模様の大きさ(画面の画素。なめらかにつなぐ格子の間隔)
NOISE_CHANGE_SECONDS = 0.12  # 次の模様へなめらかに移り変わるまでの時間(毎フレーム入れ替えると速くちらつく)
NOISE_OFFSET_RANGE = 1 << 20  # 模様を入れ替えるときに格子をずらす量(整数)の範囲(シェーダーの小数で、整数を正確に表せる大きさ)
EDGE_NOISE_AMOUNT = 0.10  # 境界の揺らぎの振れ幅(濃さの単位。INK_EDGE_WIDTH より大きいほど境界が大きく崩れる)
BACKGROUND_NOISE_AMOUNT = 0.05  # 背景の粒で明るくなる最大の量(色の単位。背景はほぼ黒なので、明るくする方向だけにする)
# 色(R, G, B、0〜1)。くっきりさせたインクの濃さで、背景の色と字の色の間を混ぜる
BACKGROUND_COLOR = (0.02, 0.02, 0.05)  # ほぼ黒で、わずかに青みがある
INK_COLOR = (0.85, 0.92, 1.0)  # 青白い
# 発光: インクを大きくぼかした光を、字の色に足す。ぼかしは64pxの画像の上で縦横に分けてかける(画面の画素ごとに計算すると重いため)
GLOW_SIGMA = 3.0  # 拡大する前の画像の画素の単位
GLOW_COLOR = (0.35, 0.6, 1.0)
GLOW_STRENGTH = 0.8
# 開発機の Mac はデスクトップ向けの OpenGL(4.1 まで)、Raspberry Pi 5 は OpenGL ES 3 に対応する。シェーダーの版の宣言だけが違う
DESKTOP_GL = sys.platform == "darwin"
SHADER_HEADER = "#version 410 core\n" if DESKTOP_GL else "#version 300 es\nprecision highp float;\n"
SHADER_DIR = Path(__file__).resolve().parent / "shaders"  # 版の宣言を除いたGLSL(版の宣言は SHADER_HEADER を先頭に付ける)


def create_window(size: tuple[int, int], fullscreen: bool, hidden: bool) -> tuple[int, int]:
    # OpenGLで描くウィンドウを作る。返り値は、実際に作られた描画先の大きさ(全画面では画面の大きさになる)
    pygame.init()
    if DESKTOP_GL:
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 1)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)
    else:
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 0)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_ES)
    flags = pygame.OPENGL | pygame.DOUBLEBUF
    if fullscreen:
        flags |= pygame.FULLSCREEN
        size = (0, 0)  # 画面の大きさに合わせる
    if hidden:
        flags |= pygame.HIDDEN
    pygame.display.set_mode(size, flags)
    pygame.display.set_caption("ghosts")
    return pygame.display.get_window_size()


def _center_square(width: int, height: int) -> tuple[int, int, int, int]:
    # 画面の中央の、短い辺を一辺とする正方形(x, y, 幅, 高さ)。字はこの中に描く
    side = min(width, height)
    return (width - side) // 2, (height - side) // 2, side, side


def _compile_shader(kind: int, file_name: str) -> int:
    shader = GL.glCreateShader(kind)
    GL.glShaderSource(shader, SHADER_HEADER + (SHADER_DIR / file_name).read_text())
    GL.glCompileShader(shader)
    if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
        raise RuntimeError(f"{file_name}: {GL.glGetShaderInfoLog(shader).decode()}")
    return shader


def _link_program(vertex_file: str, fragment_file: str) -> int:
    program = GL.glCreateProgram()
    for shader in (_compile_shader(GL.GL_VERTEX_SHADER, vertex_file), _compile_shader(GL.GL_FRAGMENT_SHADER, fragment_file)):
        GL.glAttachShader(program, shader)
    GL.glLinkProgram(program)
    if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
        raise RuntimeError(GL.glGetProgramInfoLog(program).decode())
    return program


def _create_texture(size: int) -> int:
    # 1チャンネル(0〜1を8ビットで持つ)の正方形のテクスチャ。拡大は直線の補間、外は端の値で読む
    texture = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_R8, size, size, 0, GL.GL_RED, GL.GL_UNSIGNED_BYTE, None)
    for parameter, value in ((GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR), (GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR),
                             (GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE), (GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)):
        GL.glTexParameteri(GL.GL_TEXTURE_2D, parameter, value)
    return texture


def _create_framebuffer(texture: int) -> int:
    # テクスチャに描くための描画先
    framebuffer = GL.glGenFramebuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, framebuffer)
    GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, texture, 0)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    return framebuffer


def _set_uniforms(program: int, **values: float | int | tuple[float, ...]) -> None:
    for name, value in values.items():
        location = GL.glGetUniformLocation(program, name)
        if isinstance(value, int):
            GL.glUniform1i(location, value)
        elif isinstance(value, float):
            GL.glUniform1f(location, value)
        else:
            {2: GL.glUniform2f, 3: GL.glUniform3f}[len(value)](location, *value)


def _noise_offset(index: int) -> tuple[float, float]:
    # ノイズの格子のずらし量(index ごとにばらばらな整数。格子の点の番号が整数のままになるように)
    return tuple(float(value) for value in np.random.default_rng(index).integers(0, NOISE_OFFSET_RANGE, 2))


def _noise_uniforms(seconds: float) -> dict[str, float | tuple[float, ...]]:
    # NOISE_CHANGE_SECONDS ごとに次の模様を用意し、その間をなめらかに移り変わらせる
    step = seconds / NOISE_CHANGE_SECONDS
    index = math.floor(step)
    return {"noise_offsets[0]": _noise_offset(index), "noise_offsets[1]": _noise_offset(index + 1), "noise_blend": step - index}


class GlyphRenderer:
    # 生成器が出したインクの画像をテクスチャとしてGPUに送り、輪郭のやわらげと発光のぼかしを小さな画像の上でかけてから、
    # シェーダーで画面に描く(ぼかしは画面の画素ごとに計算すると重いため)。create_window でウィンドウを作ってから使う
    def __init__(self, image_size: int, target_size: tuple[int, int]):
        self.image_size = image_size
        self.target_size = target_size
        self.square = _center_square(*target_size)
        self.glow_padding = math.ceil(GAUSSIAN_RADIUS_IN_SIGMAS * GLOW_SIGMA)  # 発光の画像の周りの余白(画素)
        self.glow_size = image_size + 2 * self.glow_padding
        self.glyph_program = _link_program("fullscreen.vert", "glyph.frag")
        self.blur_program = _link_program("fullscreen.vert", "blur.frag")
        self.vertex_array = GL.glGenVertexArrays(1)  # 頂点のデータは使わないが、描くときに結び付けておく必要がある
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)  # 1行の長さが4の倍数でない解像度でも、行の区切りを詰めて読ませる
        self.ink_texture = _create_texture(image_size)
        # ぼかしは横 → 縦の2回に分けるので、それぞれ横にぼかした画像と、さらに縦にぼかした画像の2枚を持つ
        self.soften_textures = [_create_texture(image_size) for _ in range(2)]
        self.soften_framebuffers = [_create_framebuffer(texture) for texture in self.soften_textures]
        self.glow_textures = [_create_texture(self.glow_size) for _ in range(2)]
        self.glow_framebuffers = [_create_framebuffer(texture) for texture in self.glow_textures]
        self._set_constant_uniforms()

    def _set_constant_uniforms(self) -> None:
        width, height = self.target_size
        GL.glUseProgram(self.glyph_program)
        _set_uniforms(self.glyph_program, square_scale=(width / self.square[2], height / self.square[3]),
                      image_fraction=IMAGE_SCREEN_FRACTION,
                      soft_ink=0, edge_center=INK_EDGE_CENTER, edge_width=INK_EDGE_WIDTH,
                      noise_grain=NOISE_GRAIN, edge_noise_amount=EDGE_NOISE_AMOUNT, background_noise_amount=BACKGROUND_NOISE_AMOUNT,
                      background_color=BACKGROUND_COLOR, ink_color=INK_COLOR,
                      glow=1, glow_scale=(self.image_size / self.glow_size,) * 2, glow_offset=(self.glow_padding / self.glow_size,) * 2,
                      glow_color=GLOW_COLOR, glow_strength=GLOW_STRENGTH)

    def _draw_pass(self, program: int, framebuffer: int, viewport: tuple[int, int, int, int], textures: list[int]) -> None:
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, framebuffer)
        GL.glViewport(*viewport)
        GL.glUseProgram(program)
        for unit, texture in enumerate(textures):
            GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
            GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        GL.glBindVertexArray(self.vertex_array)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)

    def _blur_ink(self, sigma: float, padding: int, framebuffers: list[int], textures: list[int]) -> None:
        # インクの画像を横 → 縦にぼかして textures[1] に描く。周りに padding 画素の余白を持たせる
        size = self.image_size + 2 * padding
        viewport = (0, 0, size, size)
        GL.glUseProgram(self.blur_program)
        _set_uniforms(self.blur_program, source=0, sigma=sigma, radius=math.ceil(GAUSSIAN_RADIUS_IN_SIGMAS * sigma),
                      blur_step=(1 / self.image_size, 0.0), source_scale=(size / self.image_size,) * 2,
                      source_offset=(-padding / self.image_size,) * 2)
        self._draw_pass(self.blur_program, framebuffers[0], viewport, [self.ink_texture])
        _set_uniforms(self.blur_program, blur_step=(0.0, 1 / size), source_scale=(1.0, 1.0), source_offset=(0.0, 0.0))
        self._draw_pass(self.blur_program, framebuffers[1], viewport, [textures[0]])

    def draw(self, ink: np.ndarray, seconds: float) -> None:
        # ink: (解像度, 解像度)、0=紙〜1=インク。seconds は表示を始めてからの時間(ノイズを時間とともに動かすため)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.ink_texture)
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, self.image_size, self.image_size, GL.GL_RED, GL.GL_UNSIGNED_BYTE,
                           (ink * 255).astype(np.uint8))
        self._blur_ink(SOFTEN_SIGMA, 0, self.soften_framebuffers, self.soften_textures)
        self._blur_ink(GLOW_SIGMA, self.glow_padding, self.glow_framebuffers, self.glow_textures)
        GL.glUseProgram(self.glyph_program)
        _set_uniforms(self.glyph_program, **_noise_uniforms(seconds))
        # 背景のノイズを画面全体にかけるため、字の正方形の外も含めて画面全体を1回で描く
        self._draw_pass(self.glyph_program, 0, (0, 0, *self.target_size), [self.soften_textures[1], self.glow_textures[1]])

    def read_pixels(self, whole_window: bool) -> np.ndarray:
        # 描いた画像を読み戻す(whole_window なら画面全体、そうでなければ字を描く中央の正方形)。返り値は (高さ, 幅, 3) の uint8
        x, y, width, height = (0, 0, *self.target_size) if whole_window else self.square
        pixels = np.frombuffer(GL.glReadPixels(x, y, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8)
        return pixels.reshape(height, width, 3)[::-1]  # OpenGLは下の行から並ぶので、上下を反転する
