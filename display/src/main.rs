// 幽霊文字の表示: 生成器のONNXをONNX Runtime(CPU)で1フレームずつ動かし、形を変え続ける字を、
// GPUでフィルタをかけてウィンドウ(または全画面)に描き続ける。1秒ごとに、FPSと、生成器・描画の1フレームの時間を出す。
// Esc・q・ウィンドウを閉じる操作で終わる。--features record で組み立てると、--record でウィンドウを出さずにGIFかmp4に書き出せる
mod generator;
#[cfg(feature = "record")]
mod record;
mod renderer;
mod walk;
mod window;

use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Parser;
use glow::HasContext;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;

use generator::Generator;
use renderer::GlyphRenderer;
use walk::{DISPLAY_FPS, LatentWalk};
use window::GlWindow;

// 展示に使う画面(720x1280で縦長が本来の向き)を横向きにした大きさ(展示では画面を回転させて使う)
const WINDOW_SIZE: (u32, u32) = (1280, 720);
const REPORT_EVERY: Duration = Duration::from_secs(1);

#[derive(Parser)]
#[command(version)]
struct Args {
    // 埋め込んだ生成器の代わりに使う、生成器のONNX(学習側の training/scripts/export_onnx.py で書き出したもの)
    #[arg(long)]
    model: Option<PathBuf>,
    // simplex noiseの軌跡を選ぶ(変えると、別の字の変わり方になる)
    #[arg(long, default_value_t = 0)]
    seed: u32,
    #[arg(long)]
    fullscreen: bool,
    // ウィンドウの大きさの倍率。既定は、開発機のMacで実物に近い見かけにする値(Macのウィンドウは約110〜130ppiで、展示に使う画面の
    // 約210ppiより大きく見える)。展示に使う画面でウィンドウとして実寸で開くときは1にする(全画面・書き出しでは使わない)
    #[arg(long, default_value_t = 0.6)]
    scale: f64,
    // 指定すると、ウィンドウを出さずに --seconds 秒ぶんを書き出して終わる。拡張子で形式を選ぶ(.mp4 は画面全体を実寸で、.gif は字の正方形を小さく)
    #[cfg(feature = "record")]
    #[arg(long)]
    record: Option<PathBuf>,
    #[cfg(feature = "record")]
    #[arg(long, default_value_t = 10.0)]
    seconds: f64,
}

struct Display {
    generator: Generator,
    walk: LatentWalk,
    window: GlWindow,
    renderer: GlyphRenderer,
}

impl Display {
    fn open(args: &Args, size: (u32, u32), hidden: bool) -> Result<Self> {
        let generator = Generator::load(args.model.as_deref())?;
        let walk = LatentWalk::new(args.seed, generator.latent_dim);
        let window = GlWindow::open(size, args.fullscreen && !hidden, hidden)?;
        let renderer = GlyphRenderer::new(&window.gl, generator.image_size, window.drawable_size())?;
        Ok(Self {
            generator,
            walk,
            window,
            renderer,
        })
    }

    fn draw_frame(&mut self, frame: u64) -> Result<(Duration, Duration)> {
        // 1フレームを生成して描く。返り値は(生成器の時間, 描画の時間)
        let started = Instant::now();
        let ink = self.generator.ink(self.walk.values(frame))?;
        let generated = Instant::now();
        self.renderer.draw(&self.window.gl, &ink, frame as f64 / DISPLAY_FPS);
        unsafe { self.window.gl.finish() }; // GPUの描画が終わるまで待ち、描画の時間を正しく測る
        Ok((generated - started, generated.elapsed()))
    }
}

fn quit_requested(events: &mut sdl2::EventPump) -> bool {
    events.poll_iter().any(|event| {
        matches!(
            event,
            Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape | Keycode::Q),
                    ..
                }
        )
    })
}

// REPORT_EVERY ごとに出す、FPSと1フレームの平均の時間を数える
struct FrameReport {
    started: Instant,
    frames: u32,
    generator_time: Duration,
    render_time: Duration,
}

impl FrameReport {
    fn new() -> Self {
        Self {
            started: Instant::now(),
            frames: 0,
            generator_time: Duration::ZERO,
            render_time: Duration::ZERO,
        }
    }

    fn add(&mut self, (generator, render): (Duration, Duration)) {
        self.frames += 1;
        self.generator_time += generator;
        self.render_time += render;
    }

    fn print_if_due(&mut self) {
        let elapsed = self.started.elapsed();
        if elapsed < REPORT_EVERY {
            return;
        }
        let per_frame = |total: Duration| total.as_secs_f64() * 1000.0 / self.frames as f64;
        println!(
            "{:.1} fps | generator {:.1} ms, render {:.1} ms per frame",
            self.frames as f64 / elapsed.as_secs_f64(),
            per_frame(self.generator_time),
            per_frame(self.render_time)
        );
        *self = Self::new();
    }
}

fn run_window(mut display: Display) -> Result<()> {
    let frame_interval = Duration::from_secs_f64(1.0 / DISPLAY_FPS);
    let mut report = FrameReport::new();
    for frame in 0.. {
        if quit_requested(&mut display.window.events) {
            break;
        }
        let started = Instant::now();
        report.add(display.draw_frame(frame)?);
        display.window.window.gl_swap_window();
        if let Some(wait) = frame_interval.checked_sub(started.elapsed()) {
            thread::sleep(wait); // 表示の速さ(DISPLAY_FPS)を超えないように待つ
        }
        report.print_if_due();
    }
    Ok(())
}

#[cfg(feature = "record")]
fn print_median_timings(timings: &[(Duration, Duration)]) {
    // 最初のフレームは初期化の時間を含むので除き、生成器・描画の1フレームの時間の中央値を出す
    let median_ms = |mut values: Vec<Duration>| {
        values.sort();
        values[values.len() / 2].as_secs_f64() * 1000.0
    };
    let measured = &timings[1.min(timings.len() - 1)..];
    println!(
        "per frame: generator {:.1} ms, render {:.1} ms (median over {} frames)",
        median_ms(measured.iter().map(|timing| timing.0).collect()),
        median_ms(measured.iter().map(|timing| timing.1).collect()),
        measured.len()
    );
}

#[cfg(feature = "record")]
fn record(args: &Args, output: &std::path::Path) -> Result<()> {
    // 1フレームずつ描いて読み戻して書き出し(長い動画でも全フレームをメモリに溜めない)、終わったら1フレームの時間を出す
    let format = record::Format::of(output)?;
    let window_size = match format {
        record::Format::Gif => (record::GIF_SIZE, record::GIF_SIZE),
        record::Format::Mp4 => WINDOW_SIZE, // 動画は実物の画面と同じ画素で書き出す
    };
    let mut display = Display::open(args, window_size, true)?;
    let size = display.window.drawable_size(); // 読み戻す画素の大きさ(高解像度の画面では、ウィンドウの大きさと違うことがある)
    let mut timings = Vec::new();
    let frame_count = (args.seconds * DISPLAY_FPS) as u64;
    anyhow::ensure!(frame_count > 0, "--seconds is shorter than one frame");
    record::write(format, output, size, frame_count, |frame| {
        timings.push(display.draw_frame(frame)?);
        Ok(display.renderer.read_pixels(&display.window.gl))
    })?;
    print_median_timings(&timings);
    println!("Saved {}", output.display());
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    #[cfg(feature = "record")]
    if let Some(output) = &args.record {
        return record(&args, output);
    }
    let window_size = (
        (WINDOW_SIZE.0 as f64 * args.scale).round() as u32,
        (WINDOW_SIZE.1 as f64 * args.scale).round() as u32,
    );
    run_window(Display::open(&args, window_size, false)?)
}
