use std::{
    ffi::{c_int, c_void, CStr, CString},
    io::prelude::*,
};

use jxl_oxide::{PixelFormat, Render};
use rusty_ffmpeg::ffi as ffmpeg;

use crate::{Error, Result};

pub struct VideoContext<W> {
    writer_ptr: *mut W,
    avio_ctx: *mut ffmpeg::AVIOContext,
    muxer_ctx: *mut ffmpeg::AVFormatContext,
    video_codec: *const ffmpeg::AVCodec,
    video_ctx: *mut ffmpeg::AVCodecContext,
    video_stream: *mut ffmpeg::AVStream,
    packet_ptr: *mut ffmpeg::AVPacket,
    source_frame_ptr: *mut ffmpeg::AVFrame,
    video_frame_ptr: *mut ffmpeg::AVFrame,
    filters: super::filter::VideoFilter,
    pts: usize,
}

impl<W: Write + Seek> VideoContext<W> {
    const BUFFER_SIZE: usize = 4096;

    pub fn new(writer: W) -> Result<Self> {
        super::init_ffmpeg_log();

        let fmt_mp4 = unsafe {
            let mut it = std::ptr::null_mut();
            loop {
                let output_fmt = ffmpeg::av_muxer_iterate(&mut it);
                if output_fmt.is_null() {
                    return Err(Error::WriteVideo("output format mp4 not found"));
                }
                let name = std::ffi::CStr::from_ptr((*output_fmt).name);
                if name == c"mp4" {
                    break output_fmt;
                }
            }
        };

        let mut output = Self {
            writer_ptr: std::ptr::null_mut(),
            avio_ctx: std::ptr::null_mut(),
            muxer_ctx: std::ptr::null_mut(),
            video_codec: std::ptr::null(),
            video_ctx: std::ptr::null_mut(),
            video_stream: std::ptr::null_mut(),
            packet_ptr: std::ptr::null_mut(),
            source_frame_ptr: std::ptr::null_mut(),
            video_frame_ptr: std::ptr::null_mut(),
            filters: super::filter::VideoFilter::new(),
            pts: 0,
        };

        let buffer = unsafe {
            let buffer = ffmpeg::av_malloc(Self::BUFFER_SIZE);
            if buffer.is_null() {
                panic!("cannot allocate memory of 4 KiB");
            }
            buffer
        };

        let writer = Box::new(writer);
        let writer_ptr = Box::into_raw(writer);
        output.writer_ptr = writer_ptr;

        let avio_ctx = unsafe {
            let ctx = ffmpeg::avio_alloc_context(
                buffer as *mut _,
                Self::BUFFER_SIZE as c_int,
                1,
                writer_ptr as *mut _,
                None,
                Some(Self::cb_write_packet),
                Some(Self::cb_seek),
            );
            if ctx.is_null() {
                ffmpeg::av_free(buffer as *mut _);
                return Err(Error::WriteVideo("failed to allocate avio context"));
            }
            ctx
        };
        output.avio_ctx = avio_ctx;

        let muxer_ctx = unsafe {
            let ctx_ptr = ffmpeg::avformat_alloc_context();
            if ctx_ptr.is_null() {
                return Err(Error::WriteVideo("failed to allocate avformat context"));
            }

            let ctx = &mut *ctx_ptr;
            ctx.oformat = fmt_mp4;
            ctx.pb = avio_ctx;

            ctx_ptr
        };
        output.muxer_ctx = muxer_ctx;

        Ok(output)
    }

    unsafe extern "C" fn cb_write_packet(
        opaque: *mut c_void,
        buf: *const u8,
        buf_size: c_int,
    ) -> c_int {
        let result = std::panic::catch_unwind(|| {
            let buf = std::slice::from_raw_parts(buf, buf_size as usize);
            let writer = &mut *(opaque as *mut W);
            writer.write_all(buf)
        });

        match result {
            Ok(Ok(_)) => 0,
            Ok(Err(e)) => {
                tracing::error!(%e, "Failed to write packet");
                e.raw_os_error()
                    .map(|v| ffmpeg::AVERROR(v as u32))
                    .unwrap_or(ffmpeg::AVERROR_UNKNOWN)
            }
            Err(_) => {
                tracing::error!("Panicked while writing packet");
                ffmpeg::AVERROR_UNKNOWN
            }
        }
    }

    unsafe extern "C" fn cb_seek(opaque: *mut c_void, offset: i64, whence: c_int) -> i64 {
        let result = std::panic::catch_unwind(|| {
            let writer = &mut *(opaque as *mut W);
            let pos = match whence as u32 {
                ffmpeg::SEEK_CUR => std::io::SeekFrom::Current(offset),
                ffmpeg::SEEK_SET => std::io::SeekFrom::Start(offset as u64),
                ffmpeg::SEEK_END => std::io::SeekFrom::End(offset),
                _ => return Err(std::io::ErrorKind::InvalidInput.into()),
            };
            writer.seek(pos)
        });

        match result {
            Ok(Ok(pos)) => pos as i64,
            Ok(Err(e)) => {
                tracing::error!(%e, "Failed to seek");
                e.raw_os_error()
                    .map(|v| ffmpeg::AVERROR(v as u32))
                    .unwrap_or(ffmpeg::AVERROR_UNKNOWN) as i64
            }
            Err(_) => {
                tracing::error!("Panicked while seeking");
                ffmpeg::AVERROR_UNKNOWN as i64
            }
        }
    }
}

impl<W> VideoContext<W> {
    pub fn init_video(
        &mut self,
        width: u32,
        height: u32,
        pixel_format: PixelFormat,
        hdr_luminance: Option<(f32, f32)>,
        font: &CStr,
    ) -> Result<()> {
        use std::fmt::Write;

        assert!(self.video_ctx.is_null());

        let video_width = (width.div_ceil(2) * 2) as i32;
        let video_height = (height.div_ceil(2) * 2) as i32;
        let pix_fmt = match pixel_format {
            PixelFormat::Gray | PixelFormat::Graya => ffmpeg::AV_PIX_FMT_GRAY16,
            PixelFormat::Rgb | PixelFormat::Rgba => ffmpeg::AV_PIX_FMT_RGB48,
            PixelFormat::Cmyk | PixelFormat::Cmyka => {
                return Err(Error::WriteVideo("CMYK not supported"))
            }
        };

        let (primaries, colorspace, trc, video_color_range, video_pix_fmt) =
            if hdr_luminance.is_some() {
                // BT.2100 PQ, rgb48 full range -> BT.2100 PQ, yuv420p10 full range
                (
                    ffmpeg::AVCOL_PRI_BT2020,
                    ffmpeg::AVCOL_SPC_BT2020_NCL,
                    ffmpeg::AVCOL_TRC_SMPTEST2084,
                    ffmpeg::AVCOL_RANGE_JPEG,
                    ffmpeg::AV_PIX_FMT_YUV420P10,
                )
            } else {
                // sRGB, rgb48 full range -> BT.709, yuv420p limited range
                (
                    ffmpeg::AVCOL_PRI_BT709,
                    ffmpeg::AVCOL_SPC_BT709,
                    ffmpeg::AVCOL_TRC_IEC61966_2_1,
                    ffmpeg::AVCOL_RANGE_MPEG,
                    ffmpeg::AV_PIX_FMT_YUV420P,
                )
            };

        let video_codec = unsafe {
            if hdr_luminance.is_some() {
                let mut codec = ffmpeg::avcodec_find_encoder_by_name(c"libsvtav1".as_ptr());
                if codec.is_null() {
                    tracing::warn!("codec libsvtav1 not found, trying libx265");
                    codec = ffmpeg::avcodec_find_encoder_by_name(c"libx265".as_ptr());
                }
                if codec.is_null() {
                    return Err(Error::WriteVideo("codec for HDR not found"));
                }
                codec
            } else {
                let codec = ffmpeg::avcodec_find_encoder_by_name(c"libx264".as_ptr());
                if codec.is_null() {
                    return Err(Error::WriteVideo("codec libx264 not found"));
                }
                codec
            }
        };
        self.video_codec = video_codec;

        let video_ctx = unsafe {
            let ctx = ffmpeg::avcodec_alloc_context3(self.video_codec);
            if ctx.is_null() {
                return Err(Error::WriteVideo("failed to allocate avcodec context"));
            }
            ctx
        };
        self.video_ctx = video_ctx;

        let video_stream = unsafe {
            let stream = ffmpeg::avformat_new_stream(self.muxer_ctx, self.video_codec);
            if stream.is_null() {
                return Err(Error::WriteVideo("failed to add stream to format"));
            }
            stream
        };
        self.video_stream = video_stream;

        let packet_ptr = unsafe {
            let packet = ffmpeg::av_packet_alloc();
            if packet.is_null() {
                return Err(Error::WriteVideo("failed to allocate packet"));
            }
            packet
        };
        self.packet_ptr = packet_ptr;

        self.filters.init(
            video_width,
            video_height,
            ffmpeg::AVCOL_SPC_RGB,
            pix_fmt,
            colorspace,
            video_pix_fmt,
            video_color_range,
            hdr_luminance.is_some(),
            font,
        )?;

        unsafe {
            let video_stream = &mut *video_stream;
            let video = &mut *video_ctx;

            video_stream.id = 0;

            video.width = video_width;
            video.height = video_height;
            video.pix_fmt = video_pix_fmt;
            video.colorspace = colorspace;
            video.color_trc = trc;
            video.color_primaries = primaries;
            video.color_range = video_color_range;

            video.time_base = self.filters.time_base();
            video_stream.time_base = video.time_base;

            if (*video_codec).id == ffmpeg::AV_CODEC_ID_AV1 {
                ffmpeg::av_opt_set(video.priv_data, c"preset".as_ptr(), c"10".as_ptr(), 0);
                ffmpeg::av_opt_set(video.priv_data, c"crf".as_ptr(), c"24".as_ptr(), 0);
                if let Some((min, max)) = hdr_luminance {
                    let mut libsvtav1_opts = String::from("enable-hdr=1:");

                    let bt2100_chrm = jxl_oxide::color::Primaries::Bt2100.as_chromaticity();
                    let d65_wtpt = jxl_oxide::color::WhitePoint::D65.as_chromaticity();
                    let min_luminance = min as i32;
                    let max_luminance = max as i32;
                    write!(
                        &mut libsvtav1_opts,
                        "mastering-display=G({gx},{gy})B({bx},{by})R({rx},{ry})WP({wpx},{wpy})L({lmax},{lmin})",
                        gx = bt2100_chrm[1][0], gy = bt2100_chrm[1][1],
                        bx = bt2100_chrm[2][0], by = bt2100_chrm[2][1],
                        rx = bt2100_chrm[0][0], ry = bt2100_chrm[0][1],
                        wpx = d65_wtpt[0], wpy = d65_wtpt[1],
                        lmax = max_luminance, lmin = min_luminance,
                    ).ok();

                    libsvtav1_opts.push('\0');
                    let c_opts = CString::from_vec_with_nul(libsvtav1_opts.into_bytes()).unwrap();
                    ffmpeg::av_opt_set(
                        video.priv_data,
                        c"svtav1-params".as_ptr(),
                        c_opts.as_ptr(),
                        0,
                    );
                }
            } else if (*video_codec).id == ffmpeg::AV_CODEC_ID_HEVC {
                ffmpeg::av_opt_set(video.priv_data, c"crf".as_ptr(), c"24".as_ptr(), 0);
                if let Some((min, max)) = hdr_luminance {
                    let mut libx265_opts = String::from("hdr-opt=1:repeat-headers=1:");

                    let bt2100_chrm = jxl_oxide::color::Primaries::Bt2100
                        .as_chromaticity()
                        .map(|xy| xy.map(|v| (v * 50000.0 + 0.5) as i32));
                    let d65_wtpt = jxl_oxide::color::WhitePoint::D65
                        .as_chromaticity()
                        .map(|v| (v * 50000.0 + 0.5) as i32);
                    let min_luminance = (min * 10000.0 + 0.5) as i32;
                    let max_luminance = (max * 10000.0 + 0.5) as i32;
                    write!(
                        &mut libx265_opts,
                        "master-display=G({gx},{gy})B({bx},{by})R({rx},{ry})WP({wpx},{wpy})L({lmax},{lmin})",
                        gx = bt2100_chrm[1][0], gy = bt2100_chrm[1][1],
                        bx = bt2100_chrm[2][0], by = bt2100_chrm[2][1],
                        rx = bt2100_chrm[0][0], ry = bt2100_chrm[0][1],
                        wpx = d65_wtpt[0], wpy = d65_wtpt[1],
                        lmax = max_luminance, lmin = min_luminance,
                    ).ok();

                    libx265_opts.push('\0');
                    let c_opts = CString::from_vec_with_nul(libx265_opts.into_bytes()).unwrap();
                    ffmpeg::av_opt_set(
                        video.priv_data,
                        c"x265-params".as_ptr(),
                        c_opts.as_ptr(),
                        0,
                    );
                }
            } else {
                ffmpeg::av_opt_set(video.priv_data, c"preset".as_ptr(), c"slow".as_ptr(), 0);
                ffmpeg::av_opt_set(video.priv_data, c"crf".as_ptr(), c"18".as_ptr(), 0);
            }

            let ret = ffmpeg::avcodec_open2(video_ctx, self.video_codec, std::ptr::null_mut());
            if ret < 0 {
                return Err(Error::Ffmpeg(ret));
            }

            let ret = ffmpeg::avcodec_parameters_from_context(video_stream.codecpar, video_ctx);
            if ret < 0 {
                return Err(Error::Ffmpeg(ret));
            }
        }

        let source_frame_ptr = Self::init_frame(|frame| {
            frame.width = video_width;
            frame.height = video_height;
            frame.format = pix_fmt;
            frame.colorspace = ffmpeg::AVCOL_SPC_RGB;
            frame.color_range = ffmpeg::AVCOL_RANGE_JPEG;
            frame.color_primaries = primaries;
            frame.color_trc = trc;
        })?;
        self.source_frame_ptr = source_frame_ptr;

        let video_frame_ptr = Self::init_frame(|frame| {
            frame.width = video_width;
            frame.height = video_height;
            frame.format = video_pix_fmt;
            frame.colorspace = colorspace;
            frame.color_range = video_color_range;
            frame.color_primaries = primaries;
            frame.color_trc = trc;
        })?;
        self.video_frame_ptr = video_frame_ptr;

        unsafe {
            let ret = ffmpeg::avformat_write_header(self.muxer_ctx, std::ptr::null_mut());
            if ret < 0 {
                return Err(Error::Ffmpeg(ret));
            }
        }

        Ok(())
    }

    fn init_frame(config_fn: impl FnOnce(&mut ffmpeg::AVFrame)) -> Result<*mut ffmpeg::AVFrame> {
        let mut frame_ptr = unsafe {
            let frame_ptr = ffmpeg::av_frame_alloc();
            if frame_ptr.is_null() {
                return Err(Error::WriteVideo("failed to allocate frame"));
            }
            frame_ptr
        };

        unsafe {
            let frame = &mut *frame_ptr;
            config_fn(frame);

            let ret = ffmpeg::av_frame_get_buffer(frame_ptr, 0);
            if ret < 0 {
                ffmpeg::av_frame_free(&mut frame_ptr);
                return Err(Error::Ffmpeg(ret));
            }
        }

        Ok(frame_ptr)
    }

    fn push_frame(&mut self) -> Result<()> {
        unsafe {
            let ret = ffmpeg::av_frame_make_writable(self.video_frame_ptr);
            if ret < 0 {
                return Err(Error::Ffmpeg(ret));
            }

            self.filters
                .filter(self.source_frame_ptr as *const _, self.video_frame_ptr)?;

            (*self.video_frame_ptr).pts = (*self.source_frame_ptr).pts;
            self.send_frame(self.video_frame_ptr)?;
        }

        Ok(())
    }

    fn repeat_frame(&mut self) -> Result<()> {
        unsafe {
            (*self.video_frame_ptr).pts = self.pts as i64;
            self.send_frame(self.video_frame_ptr)?;
        }

        Ok(())
    }

    unsafe fn send_frame(&mut self, frame_ptr: *mut ffmpeg::AVFrame) -> Result<()> {
        let start = std::time::Instant::now();

        let ret = ffmpeg::avcodec_send_frame(self.video_ctx, frame_ptr);
        if ret < 0 {
            return Err(Error::Ffmpeg(ret));
        }

        loop {
            let ret = ffmpeg::avcodec_receive_packet(self.video_ctx, self.packet_ptr);
            if ret == ffmpeg::AVERROR(ffmpeg::EAGAIN) || ret == ffmpeg::AVERROR_EOF {
                break;
            }
            if ret < 0 {
                return Err(Error::Ffmpeg(ret));
            }

            ffmpeg::av_packet_rescale_ts(
                self.packet_ptr,
                (*self.video_ctx).time_base,
                (*self.video_stream).time_base,
            );
            (*self.packet_ptr).stream_index = (*self.video_stream).index;

            let ret = ffmpeg::av_write_frame(self.muxer_ctx, self.packet_ptr);
            if ret < 0 {
                return Err(Error::Ffmpeg(ret));
            }
        }

        let elapsed = start.elapsed();
        let elapsed = elapsed.as_secs_f64();
        if elapsed > 1.0 {
            tracing::info!("Frame processing took {elapsed:.2} seconds");
        }

        Ok(())
    }

    pub fn write_frame(
        &mut self,
        render: &Render,
        description: impl std::fmt::Display,
    ) -> Result<()> {
        if self.video_ctx.is_null() {
            return Err(Error::WriteVideo("video context not initialized"));
        }

        let description = format!("{}\0", description);
        let c_description = CStr::from_bytes_with_nul(description.as_bytes())
            .map_err(|_| Error::WriteVideo("invalid description"))?;

        let frame_ptr = self.source_frame_ptr;
        let channels = unsafe {
            match (*frame_ptr).format {
                ffmpeg::AV_PIX_FMT_GRAY16 => 1,
                _ => 3,
            }
        };
        let video_width = unsafe { (*frame_ptr).width } as usize;

        let mut stream = render.stream();
        let width = stream.width() as usize;
        let height = stream.height() as usize;

        let (data, linesize) = unsafe {
            let ret = ffmpeg::av_frame_make_writable(frame_ptr);
            if ret < 0 {
                return Err(Error::Ffmpeg(ret));
            }

            (*frame_ptr).pts = self.pts as i64;
            ((*frame_ptr).data, (*frame_ptr).linesize)
        };

        let data_ptr = data[0];
        let stride = linesize[0];
        let mut tmp = vec![0f32; width * channels];
        for y in 0..height {
            let output_row = unsafe {
                let base_ptr = if stride < 0 {
                    data_ptr.offset((y + 1) as isize * stride as isize)
                } else {
                    data_ptr.offset(y as isize * stride as isize)
                };
                // SAFETY: pointer is aligned by FFmpeg.
                std::slice::from_raw_parts_mut(base_ptr as *mut u16, video_width * channels)
            };
            let (output_row, output_trailing) = output_row.split_at_mut(width * channels);
            stream.write_to_buffer(&mut tmp);
            for (o, i) in output_row.iter_mut().zip(&tmp) {
                *o = (*i * 65535.0 + 0.5).max(0.0) as u16;
            }
            output_trailing.fill(0);
        }

        self.filters.update_text(c_description)?;
        self.push_frame()?;
        self.pts += 1;
        Ok(())
    }

    pub fn write_empty_frame(&mut self, description: impl std::fmt::Display) -> Result<()> {
        if self.video_ctx.is_null() {
            return Err(Error::WriteVideo("video context not initialized"));
        }

        let description = format!("{}\0", description);
        let c_description = CStr::from_bytes_with_nul(description.as_bytes())
            .map_err(|_| Error::WriteVideo("invalid description"))?;

        let frame_ptr = self.source_frame_ptr;
        let channels = unsafe {
            match (*frame_ptr).format {
                ffmpeg::AV_PIX_FMT_GRAY16 => 1,
                _ => 3,
            }
        };
        let video_width = unsafe { (*frame_ptr).width } as usize;
        let video_height = unsafe { (*frame_ptr).height } as usize;

        let (data, linesize) = unsafe {
            let ret = ffmpeg::av_frame_make_writable(frame_ptr);
            if ret < 0 {
                return Err(Error::Ffmpeg(ret));
            }

            (*frame_ptr).pts = self.pts as i64;
            ((*frame_ptr).data, (*frame_ptr).linesize)
        };

        let data_ptr = data[0];
        let stride = linesize[0];
        for y in 0..video_height {
            unsafe {
                let base_ptr = if stride < 0 {
                    data_ptr.offset((y + 1) as isize * stride as isize)
                } else {
                    data_ptr.offset(y as isize * stride as isize)
                };
                (base_ptr as *mut u16).write_bytes(0, video_width * channels);
            }
        }

        self.filters.update_text(c_description)?;
        self.push_frame()?;
        self.pts += 1;
        Ok(())
    }

    pub fn finalize(&mut self) -> Result<()> {
        tracing::info!("Adding 2 seconds of the last image");
        for _ in 0..60 {
            self.repeat_frame()?;
            self.pts += 1;
        }

        tracing::info!("Flushing video");
        unsafe {
            self.send_frame(std::ptr::null_mut())?;
            let ret = ffmpeg::av_write_trailer(self.muxer_ctx);
            if ret < 0 {
                return Err(Error::Ffmpeg(ret));
            }
        }
        Ok(())
    }
}

impl<W> Drop for VideoContext<W> {
    fn drop(&mut self) {
        unsafe {
            ffmpeg::avcodec_free_context(&mut self.video_ctx);

            if !self.muxer_ctx.is_null() {
                ffmpeg::avformat_free_context(self.muxer_ctx);
                self.muxer_ctx = std::ptr::null_mut();
            }

            if !self.avio_ctx.is_null() {
                let buffer = (*self.avio_ctx).buffer;
                ffmpeg::av_free(buffer as *mut _);
                ffmpeg::avio_context_free(&mut self.avio_ctx);
            }

            if !self.writer_ptr.is_null() {
                let writer = Box::from_raw(self.writer_ptr);
                self.writer_ptr = std::ptr::null_mut();
                drop(writer);
            }

            ffmpeg::av_frame_free(&mut self.source_frame_ptr);
            ffmpeg::av_frame_free(&mut self.video_frame_ptr);
            ffmpeg::av_packet_free(&mut self.packet_ptr);
        }
    }
}