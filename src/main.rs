use muxui::{GssExt, Input, Style, get_virtual_screen_size};
use num::complex::Complex;
use std::cell::UnsafeCell;
use std::f32::consts::PI;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::SystemTime;
use std::{env, fs};

use muxui::raylib::{
    AttachAudioStreamProcessor, AudioCallback, ColorAlphaBlend, DetachAudioStreamProcessor,
    DrawTextCodepoint, GetGlyphIndex, GetMusicTimeLength, GetMusicTimePlayed, GetRandomValue,
    LoadFontFromMemory, SetAudioStreamBufferSizeDefault, SetRandomSeed, prelude::*,
};

const RENDER_FPS: i32 = 60;

fn get_random_value(min: i32, max: i32) -> i32 {
    unsafe { GetRandomValue(min, max) }
}

fn set_random_seed(seed: u32) {
    unsafe { SetRandomSeed(seed) }
}

/// DARK_THEME: false
/// LIGHT_THEME: true
static THEME: AtomicBool = AtomicBool::new(false);

const FFT_SIZE: usize = 1 << 13;
const RING_BUFFER_SIZE: usize = FFT_SIZE * 4;

struct AudioRingBuffer {
    buffer: UnsafeCell<[f32; RING_BUFFER_SIZE]>,
    write_idx: AtomicUsize,
    read_idx: AtomicUsize,
}

unsafe impl Sync for AudioRingBuffer {}
unsafe impl Send for AudioRingBuffer {}

impl AudioRingBuffer {
    fn new() -> Self {
        Self {
            buffer: UnsafeCell::new([0.0; RING_BUFFER_SIZE]),
            write_idx: AtomicUsize::new(0),
            read_idx: AtomicUsize::new(0),
        }
    }

    fn push(&self, sample: f32) {
        let w = self.write_idx.load(Ordering::Relaxed);
        let next = (w + 1) % RING_BUFFER_SIZE;
        // SAFETY: Only one writer thread (audio callback) modifies buffer at `write_idx`
        unsafe {
            (*self.buffer.get())[w] = sample;
        }
        self.write_idx.store(next, Ordering::Release);
    }

    fn available(&self) -> usize {
        let w = self.write_idx.load(Ordering::Acquire);
        let r = self.read_idx.load(Ordering::Relaxed);
        if w >= r {
            w - r
        } else {
            RING_BUFFER_SIZE - (r - w)
        }
    }

    fn pop(&self) -> f32 {
        let r = self.read_idx.load(Ordering::Relaxed);
        // SAFETY: Only one reader thread (main loop) reads from `read_idx`
        let val = unsafe { (*self.buffer.get())[r] };
        self.read_idx
            .store((r + 1) % RING_BUFFER_SIZE, Ordering::Relaxed);
        val
    }

    fn discard(&self, count: usize) {
        let r = self.read_idx.load(Ordering::Relaxed);
        self.read_idx
            .store((r + count) % RING_BUFFER_SIZE, Ordering::Relaxed);
    }
}

impl Default for AudioRingBuffer {
    fn default() -> Self {
        Self::new()
    }
}

struct Visualizer {
    in_raw: Box<[f32; FFT_SIZE]>,
    in_win: Box<[f32; FFT_SIZE]>,
    out_raw: Box<[Complex<f32>; FFT_SIZE]>,
    out_log: Box<[f32; FFT_SIZE]>,
    out_smooth: Box<[f32; FFT_SIZE]>,
    out_smear: Box<[f32; FFT_SIZE]>,
    bass_pulse: f32,
}

impl Visualizer {
    fn new() -> Self {
        Self {
            in_raw: Box::new([0.0; FFT_SIZE]),
            in_win: Box::new([0.0; FFT_SIZE]),
            out_raw: Box::new([Complex::new(0.0, 0.0); FFT_SIZE]),
            out_log: Box::new([0.0; FFT_SIZE]),
            out_smooth: Box::new([0.0; FFT_SIZE]),
            out_smear: Box::new([0.0; FFT_SIZE]),
            bass_pulse: 0.0,
        }
    }

    fn drain_ring(&mut self, ring: &AudioRingBuffer) {
        let available = ring.available();
        if available == 0 {
            return;
        }

        if available >= FFT_SIZE {
            ring.discard(available - FFT_SIZE);
            for sample in self.in_raw.iter_mut() {
                *sample = ring.pop();
            }
        } else {
            self.in_raw.copy_within(available..FFT_SIZE, 0);
            for sample in &mut self.in_raw[FFT_SIZE - available..] {
                *sample = ring.pop();
            }
        }
    }

    fn analyze(&mut self, dt: f32) -> usize {
        // Hann window
        for (i, (win, &raw)) in self.in_win.iter_mut().zip(self.in_raw.iter()).enumerate() {
            let t = i as f32 / (FFT_SIZE - 1) as f32;
            let hann = 0.5 - 0.5 * (2.0 * PI * t).cos();
            *win = raw * hann;
        }

        fft(&self.in_win[..], &mut self.out_raw[..], FFT_SIZE);

        let step = 1.06f32;
        let mut m = 0;
        let mut max_amp = 1.0f32;
        let mut f = 1.0f32;

        while (f as usize) < FFT_SIZE / 2 {
            let f1 = (f * step).ceil();
            let mut a = 0.0f32;
            let q_end = (FFT_SIZE / 2).min(f1 as usize);

            for q in (f as usize)..q_end {
                let b = amp(self.out_raw[q]);
                if b > a {
                    a = b;
                }
            }

            if max_amp < a {
                max_amp = a;
            }
            self.out_log[m] = a;
            m += 1;
            f = f1;
        }

        // Normalize
        for val in &mut self.out_log[..m] {
            *val /= max_amp;
        }

        // Smooth and smear
        let smoothness = 8.0;
        let smearness = 3.0;
        for i in 0..m {
            self.out_smooth[i] += (self.out_log[i] - self.out_smooth[i]) * smoothness * dt;
            self.out_smear[i] += (self.out_smooth[i] - self.out_smear[i]) * smearness * dt;
        }

        m
    }

    fn calculate_bass_energy(&mut self, m: usize, dt: f32) -> f32 {
        if m == 0 {
            return 0.0;
        }

        let bass_bins = m.min(8);
        let current_bass: f32 = self.out_smooth[..bass_bins].iter().sum::<f32>() / bass_bins as f32;

        let attack_speed = 30.0;
        let decay_speed = 10.0;

        if current_bass > self.bass_pulse {
            self.bass_pulse += (current_bass - self.bass_pulse) * attack_speed * dt;
        } else {
            self.bass_pulse += (current_bass - self.bass_pulse) * decay_speed * dt;
        }

        self.bass_pulse
    }

    fn render_radial_pulsing(
        &mut self,
        center: Vector2,
        base_radius: f32,
        max_pulse_expansion: f32,
        max_height: f32,
        m: usize,
        dt: f32,
        foreground_color: Color,
        background_color: Color,
    ) {
        let bass = self.calculate_bass_energy(m, dt);
        let dynamic_inner_radius = base_radius + max_pulse_expansion * bass;
        let angle_step = (2.0 * PI) / m as f32;
        let arc_width = (2.0 * PI * dynamic_inner_radius) / m as f32;
        let min_thick = 5.0;
        let max_thick = arc_width * 1.5;

        let saturation = 0.75;
        let value = 1.0;
        let color = Color::from_hsv(bass * 360.0, saturation, value);

        let foreground_color =
            unsafe { ColorAlphaBlend(background_color, foreground_color, color) };

        for (i, &t) in self.out_smooth[..m].iter().enumerate() {
            let angle = i as f32 * angle_step - PI * 0.5;
            let (sin_a, cos_a) = angle.sin_cos();

            let start_r = dynamic_inner_radius;
            let end_r = dynamic_inner_radius + max_height * t;
            let thick = min_thick + (max_thick - min_thick) * t.sqrt() * (1.0 + 0.5 * bass);

            let color = background_color.lerp(foreground_color.fade(0.8), t);

            let start_pos = center + Vector2::new(cos_a, sin_a) * start_r;
            let end_pos = center + Vector2::new(cos_a, sin_a) * end_r;
            draw_line_ex(start_pos, end_pos, thick, color);

            let opp_start_pos = center - Vector2::new(cos_a, sin_a) * start_r;
            let opp_end_pos = center - Vector2::new(cos_a, sin_a) * end_r;
            draw_line_ex(opp_start_pos, opp_end_pos, thick, color);
        }

        draw_circle_v(
            center,
            (dynamic_inner_radius * bass) % dynamic_inner_radius * 0.9,
            foreground_color.fade(0.8),
        );
        draw_circle_lines_v(center, dynamic_inner_radius, Color::from_hex(0x333333FF));
    }
}

impl Default for Visualizer {
    fn default() -> Self {
        Self::new()
    }
}

fn fft(input: &[f32], out: &mut [Complex<f32>], n: usize) {
    for (o, &i) in out[..n].iter_mut().zip(&input[..n]) {
        *o = Complex::new(i, 0.0);
    }

    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while (j & bit) != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            out.swap(i, j);
        }
    }

    let mut len = 2;
    while len <= n {
        let ang = 2.0 * PI / len as f32;
        let (sin_a, cos_a) = ang.sin_cos();
        let wlen = Complex::new(cos_a, sin_a);
        let mut i = 0;
        while i < n {
            let mut w = Complex::new(1.0, 0.0);
            for j in 0..len / 2 {
                let u = out[i + j];
                let v = out[i + j + len / 2] * w;
                out[i + j] = u + v;
                out[i + j + len / 2] = u - v;
                w *= wlen;
            }
            i += len;
        }
        len <<= 1;
    }
}

#[inline]
fn amp(z: Complex<f32>) -> f32 {
    let mag2 = z.re * z.re + z.im * z.im;
    if mag2 < 1e-6 { 0.0 } else { mag2.ln().max(0.0) }
}

static mut AUDIO_RING_PTR: Option<*const AudioRingBuffer> = None;

extern "C" fn audio_process_callback(buffer_data: *mut std::ffi::c_void, frames: std::ffi::c_uint) {
    let ring = match unsafe { AUDIO_RING_PTR } {
        Some(ptr) if !ptr.is_null() => unsafe { &*ptr },
        _ => return,
    };

    let samples = buffer_data as *const f32;
    let channels = 2usize;
    let slice = unsafe { std::slice::from_raw_parts(samples, (frames as usize) * channels) };

    for chunk in slice.chunks_exact(channels) {
        let mono: f32 = chunk.iter().sum::<f32>() / channels as f32;
        ring.push(mono);
    }
}

struct MusicPlaylist {
    entries: Vec<PathBuf>,
    order: Vec<usize>,
    order_pos: usize,
    current_music: Option<Music>,
    volume: f32,
    processor_callback: AudioCallback,
    paused: bool,
}

impl MusicPlaylist {
    fn new(entries: Vec<PathBuf>, processor_callback: AudioCallback) -> Self {
        let order = (0..entries.len()).collect();
        let mut playlist = Self {
            entries,
            order,
            order_pos: 0,
            current_music: None,
            volume: 0.75,
            processor_callback,
            paused: false,
        };
        playlist.shuffle_order(None);
        playlist
    }

    fn current_idx(&self) -> Option<usize> {
        self.order.get(self.order_pos).copied()
    }

    /// Shuffles playback order with Fisher-Yates, optionally avoiding `avoid_first` at index 0
    fn shuffle_order(&mut self, avoid_first: Option<usize>) {
        let len = self.order.len();
        if len <= 1 {
            return;
        }
        for i in (1..len).rev() {
            let j = get_random_value(0, i as i32) as usize;
            self.order.swap(i, j);
        }
        if let Some(prev) = avoid_first
            && self.order[0] == prev
        {
            let swap_idx = get_random_value(1, (len - 1) as i32) as usize;
            self.order.swap(0, swap_idx);
        }
    }

    /// Loads and starts playing the song at `order_pos`
    fn play_current(&mut self) -> Option<&Music> {
        let idx = self.current_idx()?;

        self.stop_and_unload();

        let path = &self.entries[idx];
        let path_str = path.to_str()?;

        let music = load_music_stream(cstr!(path_str));
        set_music_volume(music, self.volume);
        play_music_stream(music);

        unsafe {
            AttachAudioStreamProcessor(music.stream, self.processor_callback);
        }

        self.current_music = Some(music);
        self.current_music.as_ref()
    }

    /// Advances to the next track in the shuffled order (reshuffles on wrap-around)
    fn play_next(&mut self) -> Option<&Music> {
        if self.entries.is_empty() {
            return None;
        }
        if self.order_pos + 1 < self.order.len() {
            self.order_pos += 1;
        } else {
            let last_played = self.current_idx();
            self.shuffle_order(last_played);
            self.order_pos = 0;
        }
        self.play_current()
    }

    /// Steps back to previous track in the shuffled order
    fn play_previous(&mut self) -> Option<&Music> {
        if self.entries.is_empty() {
            return None;
        }
        self.order_pos = if self.order_pos == 0 {
            self.order.len() - 1
        } else {
            self.order_pos - 1
        };
        self.play_current()
    }

    /// Must be called every frame in the main loop to keep buffers fed and detect song finish
    fn update(&mut self) {
        if self.paused {
            return;
        }
        if let Some(music) = self.current_music {
            update_music_stream(music);

            let time_played = unsafe { GetMusicTimePlayed(music) };
            let time_length = unsafe { GetMusicTimeLength(music) };

            if time_length > 0.0 && time_played >= time_length {
                self.play_next();
            }
        }
    }

    fn set_volume(&mut self, volume: f32) {
        self.volume = volume.clamp(0.0, 1.0);
        if let Some(music) = self.current_music {
            set_music_volume(music, self.volume);
        }
    }

    fn current_track_name(&self) -> Option<&str> {
        self.current_idx()
            .and_then(|idx| self.entries.get(idx))
            .and_then(|p| p.file_name())
            .and_then(|f| f.to_str())
    }

    fn stop_and_unload(&mut self) {
        if let Some(music) = self.current_music.take() {
            unsafe {
                DetachAudioStreamProcessor(music.stream, self.processor_callback);
            }
            stop_music_stream(music);
            unload_music_stream(music);
        }
    }

    fn toggle_pause(&mut self) {
        self.paused = !self.paused;
        if let Some(music) = self.current_music {
            if is_music_stream_playing(music) {
                resume_music_stream(music);
            } else {
                pause_music_stream(music);
            }
        }
    }

    fn restart(&mut self) {
        self.play_current();
    }
}

impl Drop for MusicPlaylist {
    fn drop(&mut self) {
        self.stop_and_unload();
    }
}

pub struct FontRenderer {
    pub text_font: Font,
    pub emoji_font: Font,
}

impl FontRenderer {
    pub fn new(font_size: i32) -> Self {
        let text_font = get_font_default();

        let emoji_font_data = include_bytes!("../resources/fonts/NotoEmoji-Regular.ttf");

        let mut emoji_codepoints: Vec<i32> = Vec::new();

        emoji_codepoints.extend(0x2300..=0x23FF);
        emoji_codepoints.extend(0x2600..=0x27BF);
        emoji_codepoints.extend(0x1F300..=0x1F64F);
        emoji_codepoints.extend(0x1F680..=0x1F6FF);
        emoji_codepoints.extend(0x1F900..=0x1FAFF);

        let emoji_font = unsafe {
            LoadFontFromMemory(
                cstr!(".ttf").as_ptr(),
                emoji_font_data.as_ptr(),
                emoji_font_data.len() as _,
                font_size,
                emoji_codepoints.as_ptr(),
                emoji_codepoints.len() as _,
            )
        };

        set_texture_filter(text_font.texture, TextureFilter::TEXTURE_FILTER_BILINEAR);
        set_texture_filter(emoji_font.texture, TextureFilter::TEXTURE_FILTER_BILINEAR);

        Self {
            text_font,
            emoji_font,
        }
    }

    pub fn unload(self) {
        unload_font(self.text_font);
        unload_font(self.emoji_font);
    }
}

fn find_glyph_index(font: &Font, codepoint: i32) -> Option<i32> {
    let idx = unsafe { GetGlyphIndex(*font, codepoint) };
    if idx >= 0 && (idx as usize) < (font.glyphCount as usize) {
        let glyph_info = unsafe { *font.glyphs.add(idx as usize) };
        if glyph_info.value == codepoint {
            return Some(idx);
        }
    }
    None
}

fn resolve_font_and_glyph(fonts: &FontRenderer, codepoint: i32) -> (&Font, Option<i32>) {
    let prefer_emoji = matches!(
        codepoint,
        0x2300..=0x23FF | 0x2600..=0x27BF | 0x1F000..=0x1FAFF
    );

    if prefer_emoji && let Some(idx) = find_glyph_index(&fonts.emoji_font, codepoint) {
        return (&fonts.emoji_font, Some(idx));
    }
    if let Some(idx) = find_glyph_index(&fonts.text_font, codepoint) {
        return (&fonts.text_font, Some(idx));
    }
    if !prefer_emoji && let Some(idx) = find_glyph_index(&fonts.emoji_font, codepoint) {
        return (&fonts.emoji_font, Some(idx));
    }

    (&fonts.text_font, None)
}

pub fn draw_text_with_supported_emojis(
    fonts: &FontRenderer,
    text: &str,
    mut position: Vector2,
    font_size: f32,
    spacing: f32,
    tint: Color,
) {
    let base_x = position.x;

    for ch in text.chars() {
        if ch == '\n' {
            position.y += font_size + spacing;
            position.x = base_x;
            continue;
        }

        let codepoint = ch as i32;
        let (active_font, glyph_index) = resolve_font_and_glyph(fonts, codepoint);

        unsafe {
            DrawTextCodepoint(*active_font, codepoint, position, font_size, tint);
        }

        let advance = if let Some(idx) = glyph_index {
            let glyph_info = unsafe { *active_font.glyphs.add(idx as usize) };
            if glyph_info.advanceX > 0 {
                (glyph_info.advanceX as f32) * (font_size / active_font.baseSize as f32)
            } else {
                let rect = unsafe { *active_font.recs.add(idx as usize) };
                rect.width * (font_size / active_font.baseSize as f32)
            }
        } else {
            font_size * 0.5
        };

        position.x += advance + spacing;
    }
}

const WIDTH: i32 = 1200;
const HEIGHT: i32 = 800;

struct AppState {
    visualizer: Visualizer,
    fonts: FontRenderer,
    playlist: MusicPlaylist,
    ring_buffer: Arc<AudioRingBuffer>,
}

impl Drop for AppState {
    fn drop(&mut self) {
        unsafe { AUDIO_RING_PTR = None };
    }
}

fn main() {
    use rfd::FileDialog;

    set_random_seed(
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs() as _,
    );

    let music_folder_entries = {
        if let Some(music_folder) = FileDialog::new()
            .add_filter("mp3", &["mp3", "wav", "flac"])
            .set_directory(env::var("HOME").unwrap_or("/".to_string()))
            .pick_folder()
        {
            fs::read_dir(music_folder)
                .unwrap()
                .map(|res| res.map(|e| e.path()))
                .collect::<Result<Vec<_>, std::io::Error>>()
                .unwrap_or_default()
        } else {
            vec![]
        }
    };

    muxui::App::init(WIDTH, HEIGHT, "Visualizer", || {
        let tracks = music_folder_entries
            .into_iter()
            .filter(|p| p.extension().is_some_and(|e| e == "mp3"))
            .collect();
        let mut playlist = MusicPlaylist::new(tracks, Some(audio_process_callback));
        playlist.play_current();

        unsafe { SetAudioStreamBufferSizeDefault(16384) };

        let ring_buffer = Arc::new(AudioRingBuffer::new());
        unsafe {
            AUDIO_RING_PTR = Some(Arc::as_ptr(&ring_buffer));
        }

        let visualizer = Visualizer::new();
        let fonts = FontRenderer::new(28);

        AppState {
            visualizer,
            fonts,
            playlist,
            ring_buffer,
        }
    })
    .set_fps(RENDER_FPS)
    .set_config_flags(&[
        ConfigFlags::FLAG_WINDOW_ALWAYS_RUN,
        ConfigFlags::FLAG_WINDOW_UNFOCUSED,
        ConfigFlags::FLAG_WINDOW_RESIZABLE,
        ConfigFlags::FLAG_MSAA_4X_HINT,
    ])
    .set_audio_device()
    .set_style_file("resources/style.gss")
    .set_exit_key(KeyboardKey::KEY_ESCAPE)
    .register_action("next", [KeyboardKey::KEY_RIGHT])
    .register_action("previous", [KeyboardKey::KEY_LEFT])
    .register_action("volume+", [KeyboardKey::KEY_UP])
    .register_action("volume-", [KeyboardKey::KEY_DOWN])
    .register_action("pause", [KeyboardKey::KEY_SPACE])
    .register_action("restart", [KeyboardKey::KEY_R])
    .register_action("switch-theme", [KeyboardKey::KEY_I])
    .register_action("show-name", [KeyboardKey::KEY_H])
    .register_mouse_axis("mouseY", muxui::MouseAxis::Y, 1.0)
    .register_mouse_axis("mouseX", muxui::MouseAxis::X, 1.0)
    .on_update_with_input(update)
    .run();
}

fn update(ctx: &mut AppState, input: &Input, style: &Style, dt: f32) {
    if input.is_action_pressed("next") {
        ctx.playlist.play_next();
    } else if input.is_action_pressed("previous") {
        ctx.playlist.play_previous();
    } else if input.is_action_pressed("volume+") {
        ctx.playlist.set_volume(ctx.playlist.volume + 0.1);
    } else if input.is_action_pressed("volume-") {
        ctx.playlist.set_volume(ctx.playlist.volume - 0.1);
    } else if input.is_action_pressed("pause") {
        ctx.playlist.toggle_pause();
    } else if input.is_action_pressed("restart") {
        ctx.playlist.restart();
    } else if input.is_action_pressed("switch-theme") {
        let val = THEME.load(Ordering::Relaxed);
        THEME.store(!val, Ordering::Relaxed);
    }

    ctx.playlist.update();

    ctx.visualizer.drain_ring(&ctx.ring_buffer);
    let m = ctx.visualizer.analyze(dt);
    let screen = get_virtual_screen_size();
    let center = screen * style.get_val_or(&["center"], Vector2::new(0.5, 0.5));

    let background_color = style.get_val_or(&["background_color"], Color::from_hex(0x181818FF));
    let foreground_color = style.get_val_or(&["foreground_color"], Color::from_hex(0xAAAAAAFF));

    clear_background(background_color);

    let base_r = 70.0;
    let pulse_max = 50.0;
    let bar_height = 160.0;
    ctx.visualizer.render_radial_pulsing(
        center,
        base_r,
        pulse_max,
        bar_height,
        m,
        dt,
        foreground_color,
        background_color,
    );

    if input.is_action_down("show-name")
        && let Some(name) = ctx.playlist.current_track_name()
    {
        let track_label = name.to_string();
        draw_text_with_supported_emojis(
            &ctx.fonts,
            &track_label,
            Vector2 { x: 30.0, y: 30.0 },
            24.0,
            2.0,
            foreground_color,
        );
    }
}
