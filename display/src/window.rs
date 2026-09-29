// SDL2でOpenGLのウィンドウ(または全画面)を作る。開発機のMacはデスクトップ向けのOpenGL(4.1まで)、
// それ以外(ARMのLinuxなど)はOpenGL ES 3を使うので、作る文脈の種類を切り替える(シェーダーは版の宣言だけが違う。renderer.rs)
use anyhow::{Result, anyhow};
use sdl2::video::{GLContext, GLProfile, Window};
use sdl2::{EventPump, Sdl};

pub const DESKTOP_GL: bool = cfg!(target_os = "macos");

// 項目は書いた順に片付けられるので、使う側(イベント・OpenGL)から、土台(ウィンドウ・SDL)の順に並べる
pub struct GlWindow {
    pub events: EventPump,
    pub gl: glow::Context,
    _context: GLContext,
    pub window: Window,
    _sdl: Sdl,
}

impl GlWindow {
    pub fn open(size: (u32, u32), fullscreen: bool, hidden: bool) -> Result<Self> {
        let sdl = sdl2::init().map_err(|error| anyhow!(error))?;
        let video = sdl.video().map_err(|error| anyhow!(error))?;
        let attributes = video.gl_attr();
        if DESKTOP_GL {
            attributes.set_context_profile(GLProfile::Core);
            attributes.set_context_version(4, 1);
            attributes.set_context_flags().forward_compatible().set();
        } else {
            attributes.set_context_profile(GLProfile::GLES);
            attributes.set_context_version(3, 0);
        }
        attributes.set_double_buffer(true);
        let mut builder = video.window("ghosts", size.0, size.1);
        builder.opengl().position_centered();
        if fullscreen {
            builder.fullscreen_desktop(); // 画面の大きさに合わせる
        }
        if hidden {
            builder.hidden(); // 録画では、ウィンドウを出さずに描いて読み戻す
        }
        let window = builder.build()?;
        let context = window.gl_create_context().map_err(|error| anyhow!(error))?;
        // SAFETY: 作ったばかりの文脈が今の文脈なので、その関数の場所を読んでよい
        let gl = unsafe { glow::Context::from_loader_function(|name| video.gl_get_proc_address(name) as *const _) };
        sdl.mouse().show_cursor(!fullscreen); // 全画面の展示ではカーソルを隠す
        let events = sdl.event_pump().map_err(|error| anyhow!(error))?;
        Ok(Self {
            events,
            gl,
            _context: context,
            window,
            _sdl: sdl,
        })
    }

    pub fn drawable_size(&self) -> (u32, u32) {
        // 実際に描く画素の大きさ(全画面では画面の大きさになる)
        self.window.drawable_size()
    }
}
