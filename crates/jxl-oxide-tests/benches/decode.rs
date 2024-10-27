use std::{io::Cursor, path::Path, time::Duration};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use jxl_oxide::{EnumColourEncoding, JxlThreadPool, RenderingIntent};

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn decode(c: &mut Criterion) {
    let mut bench_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    bench_path.push("tests/decode/benchmark-data");

    #[cfg(feature = "rayon")]
    let pool = JxlThreadPool::rayon(None);
    #[cfg(not(feature = "rayon"))]
    let pool = JxlThreadPool::none();

    bench_one(c, &bench_path, "lumine-paimon.d0-e7", &pool);
    bench_one(c, &bench_path, "minecraft.d0-e6", &pool);
    bench_one(c, &bench_path, "srgb.d0-e1", &pool);
    bench_one(c, &bench_path, "nahida-motion.d1-e7", &pool);
    bench_one(c, &bench_path, "starrail.d1-e6", &pool);
    bench_one(c, &bench_path, "genshin-cafe.d2-e6-epf2", &pool);
    bench_one(c, &bench_path, "genshin-cafe.d2-e6-epf3", &pool);
}

fn bench_one(c: &mut Criterion, bench_path: &Path, name: &str, pool: &JxlThreadPool) {
    let mut g = c.benchmark_group(name);
    g.warm_up_time(Duration::from_secs(5));
    g.measurement_time(Duration::from_secs(15));

    let path = bench_path.join(format!("{}.jxl", name));
    let data = std::fs::read(path).unwrap();

    let pixels = {
        let reader = Cursor::new(&data);
        let image = jxl_oxide::JxlImage::builder()
            .pool(JxlThreadPool::none())
            .read(reader)
            .unwrap();
        image.width() as u64 * image.height() as u64
    };
    g.throughput(criterion::Throughput::Elements(pixels));

    g.bench_function("preferred-color", |b| {
        b.iter_with_large_drop(|| {
            let reader = Cursor::new(&data);
            let image = jxl_oxide::JxlImage::builder()
                .pool(pool.clone())
                .read(reader)
                .unwrap();
            image.render_frame(black_box(0))
        })
    });

    g.bench_function("srgb-relative", |b| {
        b.iter_with_large_drop(|| {
            let reader = Cursor::new(&data);
            let mut image = jxl_oxide::JxlImage::builder()
                .pool(pool.clone())
                .read(reader)
                .unwrap();
            image.request_color_encoding(EnumColourEncoding::srgb(RenderingIntent::Relative));
            image.render_frame(black_box(0))
        })
    });

    g.finish();
}

criterion_group!(group, decode);
criterion_main!(group);
