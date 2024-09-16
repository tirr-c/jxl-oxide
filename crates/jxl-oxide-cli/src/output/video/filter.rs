use std::ffi::{c_int, CStr};

use rusty_ffmpeg::ffi as ffmpeg;

use super::FfmpegErrorExt;
use crate::{Error, Result};

pub(super) struct VideoFilter {
    graph_ptr: *mut ffmpeg::AVFilterGraph,
    input_ctx: *mut ffmpeg::AVFilterContext,
    output_ctx: *mut ffmpeg::AVFilterContext,
}

impl VideoFilter {
    pub const fn new() -> Self {
        Self {
            graph_ptr: std::ptr::null_mut(),
            input_ctx: std::ptr::null_mut(),
            output_ctx: std::ptr::null_mut(),
        }
    }

    unsafe fn alloc_filter(
        &mut self,
        filter: &CStr,
        name: Option<&CStr>,
    ) -> Result<*mut ffmpeg::AVFilterContext> {
        let filter = ffmpeg::avfilter_get_by_name(filter.as_ptr());
        if filter.is_null() {
            tracing::error!(name = ?filter, "filter not found");
            return Err(Error::from_ffmpeg_msg("filter not found"));
        }

        let filter_ctx = ffmpeg::avfilter_graph_alloc_filter(
            self.graph_ptr,
            filter,
            name.map(|v| v.as_ptr()).unwrap_or(std::ptr::null()),
        );
        if filter_ctx.is_null() {
            tracing::error!(filter_name = ?filter, ?name, "cannot allocate filter");
            return Err(Error::from_ffmpeg_msg("cannot allocate filter"));
        }

        Ok(filter_ctx)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn init(
        &mut self,
        video_width: i32,
        video_height: i32,
        source_colorspace: u32,
        source_pix_fmt: i32,
        video_colorspace: u32,
        video_pix_fmt: i32,
        video_color_range: u32,
        is_bt2100: bool,
        font: &CStr,
    ) -> Result<()> {
        assert!(self.graph_ptr.is_null());

        let graph_ptr = unsafe { ffmpeg::avfilter_graph_alloc() };
        if graph_ptr.is_null() {
            return Err(Error::from_ffmpeg_msg("failed to allocate filter graph"));
        }
        self.graph_ptr = graph_ptr;

        unsafe {
            ffmpeg::avfilter_graph_set_auto_convert(
                graph_ptr,
                ffmpeg::AVFILTER_AUTO_CONVERT_ALL as u32,
            );
        }

        let input_ctx = unsafe {
            let filter_ctx = self.alloc_filter(c"buffer", None)?;

            let mut params = std::ptr::null_mut();
            ffmpeg::av_dict_set_int(&mut params, c"width".as_ptr(), video_width as i64, 0);
            ffmpeg::av_dict_set_int(&mut params, c"height".as_ptr(), video_height as i64, 0);
            ffmpeg::av_dict_set_int(&mut params, c"pix_fmt".as_ptr(), source_pix_fmt as i64, 0);
            ffmpeg::av_dict_set(&mut params, c"range".as_ptr(), c"pc".as_ptr(), 0);
            ffmpeg::av_dict_set_int(
                &mut params,
                c"colorspace".as_ptr(),
                source_colorspace as i64,
                0,
            );
            ffmpeg::av_dict_set(&mut params, c"time_base".as_ptr(), c"1/30".as_ptr(), 0);
            let result = ffmpeg::avfilter_init_dict(filter_ctx, &mut params).into_ffmpeg_result();
            ffmpeg::av_dict_free(&mut params);
            result?;

            filter_ctx
        };

        let drawtext_ctx = unsafe {
            let filter_ctx = self.alloc_filter(c"drawtext", Some(c"text"))?;
            let fontsize = video_height / 40;
            let borderw = (fontsize as f32 / 12.0).ceil() as i64;

            let mut params = std::ptr::null_mut();
            ffmpeg::av_dict_parse_string(
                &mut params,
                c"text=init,text_align=right,x=w-text_w-16,y=h-text_h-16".as_ptr(),
                c"=".as_ptr(),
                c",".as_ptr(),
                0,
            );
            if is_bt2100 {
                // SDR reference white
                ffmpeg::av_dict_set(
                    &mut params,
                    c"fontcolor".as_ptr(),
                    c"white@0.58".as_ptr(),
                    0,
                );
            } else {
                ffmpeg::av_dict_set(&mut params, c"fontcolor".as_ptr(), c"white".as_ptr(), 0);
            }
            ffmpeg::av_dict_set_int(&mut params, c"fontsize".as_ptr(), fontsize as i64, 0);
            ffmpeg::av_dict_set_int(&mut params, c"borderw".as_ptr(), borderw, 0);
            ffmpeg::av_dict_set(&mut params, c"font".as_ptr(), font.as_ptr(), 0);
            let result = ffmpeg::avfilter_init_dict(filter_ctx, &mut params).into_ffmpeg_result();
            ffmpeg::av_dict_free(&mut params);
            result?;

            filter_ctx
        };

        let output_ctx = unsafe {
            let filter_ctx = self.alloc_filter(c"buffersink", None)?;

            ffmpeg::avfilter_init_str(filter_ctx, std::ptr::null()).into_ffmpeg_result()?;

            let pix_fmts = [video_pix_fmt];
            let color_spaces = [video_colorspace];
            let color_ranges = [video_color_range];

            ffmpeg::av_opt_set_bin(
                filter_ctx as *mut _,
                c"pix_fmts".as_ptr(),
                pix_fmts.as_ptr() as *const _,
                size_of_val(&pix_fmts[0]) as c_int,
                ffmpeg::AV_OPT_SEARCH_CHILDREN as c_int,
            )
            .into_ffmpeg_result()?;

            ffmpeg::av_opt_set_bin(
                filter_ctx as *mut _,
                c"color_spaces".as_ptr(),
                color_spaces.as_ptr() as *const _,
                size_of_val(&color_spaces[0]) as c_int,
                ffmpeg::AV_OPT_SEARCH_CHILDREN as c_int,
            )
            .into_ffmpeg_result()?;

            ffmpeg::av_opt_set_bin(
                filter_ctx as *mut _,
                c"color_ranges".as_ptr(),
                color_ranges.as_ptr() as *const _,
                size_of_val(&color_ranges[0]) as c_int,
                ffmpeg::AV_OPT_SEARCH_CHILDREN as c_int,
            )
            .into_ffmpeg_result()?;

            filter_ctx
        };

        unsafe {
            ffmpeg::avfilter_link(input_ctx, 0, drawtext_ctx, 0).into_ffmpeg_result()?;
            ffmpeg::avfilter_link(drawtext_ctx, 0, output_ctx, 0).into_ffmpeg_result()?;
            ffmpeg::avfilter_graph_config(graph_ptr, std::ptr::null_mut()).into_ffmpeg_result()?;
        }

        self.input_ctx = input_ctx;
        self.output_ctx = output_ctx;

        Ok(())
    }

    #[inline]
    pub fn time_base(&self) -> ffmpeg::AVRational {
        if self.output_ctx.is_null() {
            ffmpeg::AVRational { num: 0, den: 1 }
        } else {
            unsafe { ffmpeg::av_buffersink_get_time_base(self.output_ctx as *const _) }
        }
    }

    pub fn filter(
        &mut self,
        source_frame_ptr: *const ffmpeg::AVFrame,
        dest_frame_ptr: *mut ffmpeg::AVFrame,
    ) -> Result<()> {
        unsafe {
            ffmpeg::av_buffersrc_write_frame(self.input_ctx, source_frame_ptr)
                .into_ffmpeg_result()?;
            ffmpeg::av_buffersink_get_frame(self.output_ctx, dest_frame_ptr)
                .into_ffmpeg_result()?;
        }

        Ok(())
    }

    pub fn update_text(&mut self, text: &CStr) -> Result<()> {
        unsafe {
            ffmpeg::avfilter_graph_send_command(
                self.graph_ptr,
                c"text".as_ptr(),
                c"text".as_ptr(),
                text.as_ptr(),
                std::ptr::null_mut(),
                0,
                0,
            )
            .into_ffmpeg_result()?;
        }
        Ok(())
    }
}

impl Drop for VideoFilter {
    fn drop(&mut self) {
        unsafe {
            ffmpeg::avfilter_graph_free(&mut self.graph_ptr);
        }
    }
}
