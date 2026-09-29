// 表示を、ウィンドウを出さずに描いて読み戻し、GIFかmp4に書き出す(Macでの確認・共有のため。--features record のときだけ組み込む)
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use anyhow::{Context, Result, bail};
use ffmpeg::format::Pixel;
use ffmpeg_next as ffmpeg;

use crate::walk::DISPLAY_FPS;

pub const GIF_SIZE: u32 = 350; // GIFはファイルを小さくするため、この大きさ(正方形)で描く
const GIF_TIME_UNIT_MS: u64 = 10; // GIFの表示時間の単位
const GIF_QUANTIZE_SPEED: i32 = 10; // 256色に減らすときの速さ(1〜30。小さいほど色が正確で遅い)
// mp4の画質(x264のCRF。小さいほど高画質で、18は見た目では元とほぼ区別がつかない)。
// Rustだけで書かれた符号化(openh264)は、背景の細かい粒を16画素の四角ごとに平らに潰したので、x264を使う
const MP4_CRF: &str = "18";

pub enum Format {
    Gif,
    Mp4,
}

impl Format {
    pub fn of(path: &Path) -> Result<Self> {
        match path.extension().and_then(|extension| extension.to_str()) {
            Some("gif") => Ok(Self::Gif),
            Some("mp4") => Ok(Self::Mp4),
            _ => bail!("--record must end with .gif or .mp4: {}", path.display()),
        }
    }
}

pub fn write(format: Format, output: &Path, size: (u32, u32), frame_count: u64, next_frame: impl FnMut(u64) -> Result<Vec<u8>>) -> Result<()> {
    // next_frame(フレームの番号) は、そのフレームを描いて読み戻した 幅 x 高さ x 3(RGB、上の行から)
    match format {
        Format::Gif => write_gif(output, size, frame_count, next_frame),
        Format::Mp4 => write_mp4(output, size, frame_count, next_frame),
    }
}

fn gif_frame_start_ms(frame: u64) -> u64 {
    // GIFの表示時間は10ミリ秒単位なので、1/fps 秒ずつに丸めると速さがずれる(30fpsの33ミリ秒は30ミリ秒になる)。
    // 始まりからの時刻を丸めて差をとり、30・40・30ミリ秒…のように配って、全体の長さを合わせる
    ((frame as f64 * 1000.0 / DISPLAY_FPS / GIF_TIME_UNIT_MS as f64).round() as u64) * GIF_TIME_UNIT_MS
}

fn write_gif(output: &Path, size: (u32, u32), frame_count: u64, mut next_frame: impl FnMut(u64) -> Result<Vec<u8>>) -> Result<()> {
    let file = BufWriter::new(File::create(output).with_context(|| format!("creating {}", output.display()))?);
    let (width, height) = (size.0 as u16, size.1 as u16);
    let mut encoder = gif::Encoder::new(file, width, height, &[])?;
    encoder.set_repeat(gif::Repeat::Infinite)?;
    for frame in 0..frame_count {
        let mut image = gif::Frame::from_rgb_speed(width, height, &next_frame(frame)?, GIF_QUANTIZE_SPEED);
        image.delay = ((gif_frame_start_ms(frame + 1) - gif_frame_start_ms(frame)) / GIF_TIME_UNIT_MS) as u16;
        encoder.write_frame(&image)?;
    }
    Ok(())
}

fn open_mp4(output: &Path, size: (u32, u32), time_base: ffmpeg::Rational) -> Result<(ffmpeg::format::context::Output, ffmpeg::encoder::Video)> {
    // H.264(yuv420pはほとんどの再生環境で開ける形式)の符号化を用意し、mp4の先頭の情報まで書き込む
    ffmpeg::init()?;
    ffmpeg::log::set_level(ffmpeg::log::Level::Warning); // x264が毎回出す詳しい集計は出さない
    let mut container = ffmpeg::format::output(output).with_context(|| format!("creating {}", output.display()))?;
    let codec = ffmpeg::encoder::find_by_name("libx264").context("FFmpeg has no libx264 encoder")?;
    let global_header = container.format().flags().contains(ffmpeg::format::Flags::GLOBAL_HEADER);
    let mut stream = container.add_stream(codec)?;
    let mut settings = ffmpeg::codec::context::Context::new_with_codec(codec).encoder().video()?;
    settings.set_width(size.0);
    settings.set_height(size.1);
    settings.set_format(Pixel::YUV420P);
    settings.set_time_base(time_base);
    settings.set_frame_rate(Some(ffmpeg::Rational::new(DISPLAY_FPS as i32, 1)));
    if global_header {
        settings.set_flags(ffmpeg::codec::Flags::GLOBAL_HEADER); // mp4は、画像の設定(SPS・PPS)を動画の情報として別に持つ
    }
    let mut options = ffmpeg::Dictionary::new();
    options.set("crf", MP4_CRF);
    let encoder = settings.open_with(options)?;
    stream.set_parameters(&encoder);
    stream.set_time_base(time_base);
    // 再生側が読む平均のフレームレート(指定しないと、フレームの数と長さから計算され、30からわずかにずれる)
    stream.set_avg_frame_rate(ffmpeg::Rational::new(DISPLAY_FPS as i32, 1));
    container.write_header()?;
    Ok((container, encoder))
}

fn rgb_frame(pixels: &[u8], size: (u32, u32)) -> ffmpeg::frame::Video {
    // FFmpegのフレームは、1行の長さを揃えるため行の終わりに余白を持つことがあるので、1行ずつ写す
    let mut frame = ffmpeg::frame::Video::new(Pixel::RGB24, size.0, size.1);
    let (row_bytes, stride) = (size.0 as usize * 3, frame.stride(0));
    for (row, source) in pixels.chunks_exact(row_bytes).enumerate() {
        frame.data_mut(0)[row * stride..row * stride + row_bytes].copy_from_slice(source);
    }
    frame
}

fn write_packets(encoder: &mut ffmpeg::encoder::Video, container: &mut ffmpeg::format::context::Output, time_base: ffmpeg::Rational) -> Result<()> {
    // 符号化が終わった分を、mp4の時刻の単位に直して書き込む
    let mut packet = ffmpeg::Packet::empty();
    while encoder.receive_packet(&mut packet).is_ok() {
        packet.set_stream(0);
        packet.rescale_ts(time_base, container.stream(0).context("no video stream")?.time_base());
        packet.write_interleaved(container)?;
    }
    Ok(())
}

fn write_mp4(output: &Path, size: (u32, u32), frame_count: u64, mut next_frame: impl FnMut(u64) -> Result<Vec<u8>>) -> Result<()> {
    let time_base = ffmpeg::Rational::new(1, DISPLAY_FPS as i32); // 1フレームを1とする時刻の単位
    let (mut container, mut encoder) = open_mp4(output, size, time_base)?;
    let (width, height) = size;
    let mut converter = ffmpeg::software::scaling::Context::get(
        Pixel::RGB24,
        width,
        height,
        Pixel::YUV420P,
        width,
        height,
        ffmpeg::software::scaling::Flags::BILINEAR,
    )?;
    for frame in 0..frame_count {
        let mut yuv = ffmpeg::frame::Video::empty();
        converter.run(&rgb_frame(&next_frame(frame)?, size), &mut yuv)?;
        yuv.set_pts(Some(frame as i64));
        encoder.send_frame(&yuv)?;
        write_packets(&mut encoder, &mut container, time_base)?;
    }
    encoder.send_eof()?; // 符号化の途中で溜まっている分を出し切る
    write_packets(&mut encoder, &mut container, time_base)?;
    container.write_trailer()?;
    Ok(())
}
