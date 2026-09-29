// 生成器が作ったインクの画像を、GPUのシェーダー(shaders/)でフィルタをかけて画面に描く。
// 輪郭のやわらげ → くっきりさせる(境界を溶かすノイズつき) → 色づけ → 発光 → 背景のノイズ。見た目の値はこのファイルの定数で、
// フィルタをGPUでかけるのは、CPUを生成器に専念させるため
use anyhow::{Result, anyhow};
use glow::{HasContext, NativeFramebuffer, NativeProgram, NativeTexture, NativeVertexArray, PixelUnpackData};

use crate::window::DESKTOP_GL;

const IMAGE_SCREEN_FRACTION: f32 = 0.7; // 字の画像を、画面の短い辺のこの割合の大きさで中央に描く(周りは余白)
// 字の輪郭のやわらげ: 拡大する前の画像の画素の単位で、ガウスのぼかしをかけてから拡大する(64pxの格子のギザギザを目立たなくする)
const SOFTEN_SIGMA: f32 = 0.8;
const GAUSSIAN_RADIUS_IN_SIGMAS: f32 = 3.0; // ぼかしで足し合わせる範囲(σの何倍までか。それより外の重みは無視できるほど小さい)
// 輪郭をくっきりさせる: やわらげたインクの濃さを、INK_EDGE_CENTER の前後 ±INK_EDGE_WIDTH の範囲でなめらかに0から1へ切り替える
const INK_EDGE_CENTER: f32 = 0.5;
const INK_EDGE_WIDTH: f32 = 0.15;
// ノイズ: 隣どうしをなめらかにつないだ揺らぎの模様を、少しずつ次の模様へ移り変わらせる。同じ1つの模様で、字の境界を背景に
// 溶け込ませ、背景に粒を重ねる
const NOISE_GRAIN: f32 = 1.5; // 模様の大きさ(画面の画素。なめらかにつなぐ格子の間隔)
const NOISE_CHANGE_SECONDS: f64 = 0.12; // 次の模様へなめらかに移り変わるまでの時間(毎フレーム入れ替えると速くちらつく)
const NOISE_OFFSET_RANGE: u64 = 1 << 20; // 模様を入れ替えるときに格子をずらす量(整数)の範囲(シェーダーの小数で、整数を正確に表せる大きさ)
const EDGE_NOISE_AMOUNT: f32 = 0.10; // 境界の揺らぎの振れ幅(濃さの単位。INK_EDGE_WIDTH より大きいほど境界が大きく崩れる)
const BACKGROUND_NOISE_AMOUNT: f32 = 0.05; // 背景の粒で明るくなる最大の量(色の単位。背景はほぼ黒なので、明るくする方向だけにする)
// 色(R, G, B、0〜1)。くっきりさせたインクの濃さで、背景の色と字の色の間を混ぜる
const BACKGROUND_COLOR: [f32; 3] = [0.02, 0.02, 0.05]; // ほぼ黒で、わずかに青みがある
const INK_COLOR: [f32; 3] = [0.85, 0.92, 1.0]; // 青白い
// 発光: インクを大きくぼかした光を、字の色に足す。ぼかしは小さな画像の上で縦横に分けてかける(画面の画素ごとに計算すると重いため)
const GLOW_SIGMA: f32 = 3.0; // 拡大する前の画像の画素の単位
const GLOW_COLOR: [f32; 3] = [0.35, 0.6, 1.0];
const GLOW_STRENGTH: f32 = 0.8;
// シェーダーは版の宣言を除いて書き、ここで先頭に付ける(開発機のMacはデスクトップ向けのOpenGL、それ以外(ARMのLinuxなど)はOpenGL ES 3)
const SHADER_HEADER: &str = if DESKTOP_GL {
    "#version 410 core\n"
} else {
    "#version 300 es\nprecision highp float;\n"
};
const FULLSCREEN_VERTEX: &str = include_str!("../shaders/fullscreen.vert");
const BLUR_FRAGMENT: &str = include_str!("../shaders/blur.frag");
const GLYPH_FRAGMENT: &str = include_str!("../shaders/glyph.frag");

enum Uniform {
    Int(i32),
    Float(f32),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
}

struct BlurTargets {
    // 横 → 縦の2回に分けてぼかすので、横にぼかした画像と、さらに縦にぼかした画像の2枚を持つ
    textures: [NativeTexture; 2],
    framebuffers: [NativeFramebuffer; 2],
}

pub struct GlyphRenderer {
    image_size: i32,
    target_size: (i32, i32),
    glow_padding: i32, // 発光の画像の周りの余白(画素)。光が字の画像の外へ広がっても切れないように
    glyph_program: NativeProgram,
    blur_program: NativeProgram,
    vertex_array: NativeVertexArray, // 頂点のデータは使わないが、描くときに結び付けておく必要がある
    ink_texture: NativeTexture,
    soften: BlurTargets,
    glow: BlurTargets,
}

fn compile_shader(gl: &glow::Context, kind: u32, source: &str) -> Result<glow::NativeShader> {
    unsafe {
        let shader = gl.create_shader(kind).map_err(|error| anyhow!(error))?;
        gl.shader_source(shader, &format!("{SHADER_HEADER}{source}"));
        gl.compile_shader(shader);
        if !gl.get_shader_compile_status(shader) {
            return Err(anyhow!("shader compile error: {}", gl.get_shader_info_log(shader)));
        }
        Ok(shader)
    }
}

fn link_program(gl: &glow::Context, vertex: &str, fragment: &str) -> Result<NativeProgram> {
    unsafe {
        let program = gl.create_program().map_err(|error| anyhow!(error))?;
        for shader in [
            compile_shader(gl, glow::VERTEX_SHADER, vertex)?,
            compile_shader(gl, glow::FRAGMENT_SHADER, fragment)?,
        ] {
            gl.attach_shader(program, shader);
        }
        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            return Err(anyhow!("program link error: {}", gl.get_program_info_log(program)));
        }
        Ok(program)
    }
}

fn create_texture(gl: &glow::Context, size: i32) -> Result<NativeTexture> {
    // 1チャンネル(0〜1を8ビットで持つ)の正方形のテクスチャ。拡大は直線の補間、外は端の値で読む
    unsafe {
        let texture = gl.create_texture().map_err(|error| anyhow!(error))?;
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::R8 as i32,
            size,
            size,
            0,
            glow::RED,
            glow::UNSIGNED_BYTE,
            PixelUnpackData::Slice(None),
        );
        for (parameter, value) in [
            (glow::TEXTURE_MIN_FILTER, glow::LINEAR),
            (glow::TEXTURE_MAG_FILTER, glow::LINEAR),
            (glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE),
            (glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE),
        ] {
            gl.tex_parameter_i32(glow::TEXTURE_2D, parameter, value as i32);
        }
        Ok(texture)
    }
}

fn create_framebuffer(gl: &glow::Context, texture: NativeTexture) -> Result<NativeFramebuffer> {
    // テクスチャに描くための描画先
    unsafe {
        let framebuffer = gl.create_framebuffer().map_err(|error| anyhow!(error))?;
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(framebuffer));
        gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::TEXTURE_2D, Some(texture), 0);
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        Ok(framebuffer)
    }
}

fn create_blur_targets(gl: &glow::Context, size: i32) -> Result<BlurTargets> {
    let textures = [create_texture(gl, size)?, create_texture(gl, size)?];
    let framebuffers = [create_framebuffer(gl, textures[0])?, create_framebuffer(gl, textures[1])?];
    Ok(BlurTargets { textures, framebuffers })
}

fn set_uniforms(gl: &glow::Context, program: NativeProgram, values: &[(&str, Uniform)]) {
    unsafe {
        gl.use_program(Some(program));
        for (name, value) in values {
            let location = gl.get_uniform_location(program, name);
            match value {
                Uniform::Int(value) => gl.uniform_1_i32(location.as_ref(), *value),
                Uniform::Float(value) => gl.uniform_1_f32(location.as_ref(), *value),
                Uniform::Vec2([x, y]) => gl.uniform_2_f32(location.as_ref(), *x, *y),
                Uniform::Vec3([x, y, z]) => gl.uniform_3_f32(location.as_ref(), *x, *y, *z),
            }
        }
    }
}

fn noise_offset(index: i64) -> [f32; 2] {
    // ノイズの格子のずらし量(indexごとにばらばらな整数。格子の点の番号が整数のままになるように)。
    // 模様の番号から決まる値であればよいので、整数のハッシュ(SplitMix64)で作る
    let mut state = index as u64;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) % NOISE_OFFSET_RANGE) as f32
    };
    [next(), next()]
}

fn gaussian_radius(sigma: f32) -> i32 {
    (GAUSSIAN_RADIUS_IN_SIGMAS * sigma).ceil() as i32
}

impl GlyphRenderer {
    pub fn new(gl: &glow::Context, image_size: usize, target_size: (u32, u32)) -> Result<Self> {
        let image_size = image_size as i32;
        let glow_padding = gaussian_radius(GLOW_SIGMA);
        let renderer = Self {
            image_size,
            target_size: (target_size.0 as i32, target_size.1 as i32),
            glow_padding,
            glyph_program: link_program(gl, FULLSCREEN_VERTEX, GLYPH_FRAGMENT)?,
            blur_program: link_program(gl, FULLSCREEN_VERTEX, BLUR_FRAGMENT)?,
            vertex_array: unsafe { gl.create_vertex_array().map_err(|error| anyhow!(error))? },
            ink_texture: create_texture(gl, image_size)?,
            soften: create_blur_targets(gl, image_size)?,
            glow: create_blur_targets(gl, image_size + 2 * glow_padding)?,
        };
        unsafe { gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1) }; // 1行の長さが4の倍数でない解像度でも、行の区切りを詰めて読ませる
        renderer.set_constant_uniforms(gl);
        Ok(renderer)
    }

    fn set_constant_uniforms(&self, gl: &glow::Context) {
        let (width, height) = self.target_size;
        let side = width.min(height) as f32; // 字は、画面の中央の、短い辺を一辺とする正方形の中に描く
        let glow_size = (self.image_size + 2 * self.glow_padding) as f32;
        set_uniforms(
            gl,
            self.glyph_program,
            &[
                ("square_scale", Uniform::Vec2([width as f32 / side, height as f32 / side])),
                ("image_fraction", Uniform::Float(IMAGE_SCREEN_FRACTION)),
                ("soft_ink", Uniform::Int(0)),
                ("edge_center", Uniform::Float(INK_EDGE_CENTER)),
                ("edge_width", Uniform::Float(INK_EDGE_WIDTH)),
                ("noise_grain", Uniform::Float(NOISE_GRAIN)),
                ("edge_noise_amount", Uniform::Float(EDGE_NOISE_AMOUNT)),
                ("background_noise_amount", Uniform::Float(BACKGROUND_NOISE_AMOUNT)),
                ("background_color", Uniform::Vec3(BACKGROUND_COLOR)),
                ("ink_color", Uniform::Vec3(INK_COLOR)),
                ("glow", Uniform::Int(1)),
                ("glow_scale", Uniform::Vec2([self.image_size as f32 / glow_size; 2])),
                ("glow_offset", Uniform::Vec2([self.glow_padding as f32 / glow_size; 2])),
                ("glow_color", Uniform::Vec3(GLOW_COLOR)),
                ("glow_strength", Uniform::Float(GLOW_STRENGTH)),
            ],
        );
    }

    fn draw_pass(
        &self,
        gl: &glow::Context,
        program: NativeProgram,
        framebuffer: Option<NativeFramebuffer>,
        size: (i32, i32),
        textures: &[NativeTexture],
    ) {
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, framebuffer);
            gl.viewport(0, 0, size.0, size.1);
            gl.use_program(Some(program));
            for (unit, texture) in textures.iter().enumerate() {
                gl.active_texture(glow::TEXTURE0 + unit as u32);
                gl.bind_texture(glow::TEXTURE_2D, Some(*texture));
            }
            gl.bind_vertex_array(Some(self.vertex_array));
            gl.draw_arrays(glow::TRIANGLES, 0, 3);
        }
    }

    fn upload_ink(&self, gl: &glow::Context, ink: &[f32]) {
        // 0〜1のインクの濃さを8ビットにして、インクのテクスチャに送る
        let bytes: Vec<u8> = ink.iter().map(|value| (value * 255.0) as u8).collect();
        unsafe {
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(self.ink_texture));
            gl.tex_sub_image_2d(
                glow::TEXTURE_2D,
                0,
                0,
                0,
                self.image_size,
                self.image_size,
                glow::RED,
                glow::UNSIGNED_BYTE,
                PixelUnpackData::Slice(Some(&bytes)),
            );
        }
    }

    fn blur_ink(&self, gl: &glow::Context, sigma: f32, padding: i32, targets: &BlurTargets) {
        // インクの画像を横 → 縦にぼかして targets.textures[1] に描く。周りにpadding画素の余白を持たせる
        let size = self.image_size + 2 * padding;
        let image_size = self.image_size as f32;
        set_uniforms(
            gl,
            self.blur_program,
            &[
                ("source", Uniform::Int(0)),
                ("sigma", Uniform::Float(sigma)),
                ("radius", Uniform::Int(gaussian_radius(sigma))),
                ("blur_step", Uniform::Vec2([1.0 / image_size, 0.0])),
                ("source_scale", Uniform::Vec2([size as f32 / image_size; 2])),
                ("source_offset", Uniform::Vec2([-padding as f32 / image_size; 2])),
            ],
        );
        self.draw_pass(gl, self.blur_program, Some(targets.framebuffers[0]), (size, size), &[self.ink_texture]);
        set_uniforms(
            gl,
            self.blur_program,
            &[
                ("blur_step", Uniform::Vec2([0.0, 1.0 / size as f32])),
                ("source_scale", Uniform::Vec2([1.0, 1.0])),
                ("source_offset", Uniform::Vec2([0.0, 0.0])),
            ],
        );
        self.draw_pass(gl, self.blur_program, Some(targets.framebuffers[1]), (size, size), &[targets.textures[0]]);
    }

    pub fn draw(&self, gl: &glow::Context, ink: &[f32], seconds: f64) {
        // ink: 解像度 x 解像度、0=紙〜1=インク。secondsは表示を始めてからの時間(ノイズを時間とともに動かすため)
        self.upload_ink(gl, ink);
        self.blur_ink(gl, SOFTEN_SIGMA, 0, &self.soften);
        self.blur_ink(gl, GLOW_SIGMA, self.glow_padding, &self.glow);
        // NOISE_CHANGE_SECONDS ごとに次の模様を用意し、その間をなめらかに移り変わらせる
        let step = seconds / NOISE_CHANGE_SECONDS;
        let index = step.floor();
        set_uniforms(
            gl,
            self.glyph_program,
            &[
                ("noise_offsets[0]", Uniform::Vec2(noise_offset(index as i64))),
                ("noise_offsets[1]", Uniform::Vec2(noise_offset(index as i64 + 1))),
                ("noise_blend", Uniform::Float((step - index) as f32)),
            ],
        );
        // 背景のノイズを画面全体にかけるため、字の正方形の外も含めて画面全体を1回で描く
        self.draw_pass(
            gl,
            self.glyph_program,
            None,
            self.target_size,
            &[self.soften.textures[1], self.glow.textures[1]],
        );
    }

    #[cfg(feature = "record")]
    pub fn read_pixels(&self, gl: &glow::Context) -> Vec<u8> {
        // 描いた画面全体を読み戻す。返り値は 幅 x 高さ x 3(RGB)で、上の行から
        let (width, height) = self.target_size;
        let mut pixels = vec![0u8; width as usize * height as usize * 3];
        unsafe {
            gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1); // 1行の長さが4の倍数でない幅でも、行の区切りを詰めて書かせる
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.read_pixels(
                0,
                0,
                width,
                height,
                glow::RGB,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(&mut pixels)),
            );
        }
        // OpenGLは下の行から並ぶので、上下を反転する
        pixels.chunks_exact(width as usize * 3).rev().flatten().copied().collect()
    }
}
