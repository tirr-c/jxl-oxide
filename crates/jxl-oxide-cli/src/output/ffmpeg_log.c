#include <stdio.h>
#include <stdlib.h>

void jxl_oxide_ffmpeg_log(void *avcl, int level, const char *line);

void jxl_oxide_ffmpeg_log_c(void *avcl, int level, const char *fmt, va_list vl) {
  char *buf = malloc(65536);
  if (buf == NULL) {
    return;
  }

  vsnprintf(buf, 65536, fmt, vl);
  jxl_oxide_ffmpeg_log(avcl, level, buf);
  free(buf);
}
