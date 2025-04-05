mod clap {
    use std::path::Path;

    use clap::Parser;

    use super::super::{Args, Subcommands};

    #[test]
    fn basic_decode() {
        let args =
            Args::try_parse_from(["jxl-oxide", "decode", "input.jxl", "-o", "output.png"]).unwrap();
        let Some(Subcommands::Decode(decode_args)) = args.subcommand else {
            panic!();
        };
        assert!(args.decode.is_none());
        assert_eq!(args.globals.verbose, 0);
        assert_eq!(decode_args.input, Path::new("input.jxl"));
        assert_eq!(decode_args.output.as_deref(), Some(Path::new("output.png")));
    }

    #[test]
    fn basic_info() {
        let args = Args::try_parse_from(["jxl-oxide", "info", "input.jxl"]).unwrap();
        let Some(Subcommands::Info(info_args)) = args.subcommand else {
            panic!();
        };
        assert!(args.decode.is_none());
        assert_eq!(args.globals.verbose, 0);
        assert_eq!(info_args.input, Path::new("input.jxl"));
    }

    #[test]
    fn default_decode() {
        let args = Args::try_parse_from(["jxl-oxide", "input.jxl", "-o", "output.png"]).unwrap();
        let Some(decode_args) = args.decode else {
            panic!();
        };
        assert!(args.subcommand.is_none());
        assert_eq!(args.globals.verbose, 0);
        assert_eq!(decode_args.input, Path::new("input.jxl"));
        assert_eq!(decode_args.output.as_deref(), Some(Path::new("output.png")));
    }

    #[test]
    fn flag_style_info() {
        let args = Args::try_parse_from(["jxl-oxide", "-I", "input.jxl"]).unwrap();
        let Some(Subcommands::Info(info_args)) = args.subcommand else {
            panic!();
        };
        assert!(args.decode.is_none());
        assert_eq!(args.globals.verbose, 0);
        assert_eq!(info_args.input, Path::new("input.jxl"));
    }

    #[test]
    fn verbose() {
        let args = Args::try_parse_from(["jxl-oxide", "input.jxl", "-v"]).unwrap();
        assert_eq!(args.globals.verbose, 1);

        let args = Args::try_parse_from(["jxl-oxide", "--verbose", "input.jxl"]).unwrap();
        assert_eq!(args.globals.verbose, 1);

        assert!(Args::try_parse_from(["jxl-oxide", "-v"]).is_err());

        let args = Args::try_parse_from(["jxl-oxide", "-Iv", "input.jxl"]).unwrap();
        assert_eq!(args.globals.verbose, 1);

        let args = Args::try_parse_from(["jxl-oxide", "-vv", "input.jxl"]).unwrap();
        assert_eq!(args.globals.verbose, 2);

        let args = Args::try_parse_from(["jxl-oxide", "-v", "-v", "input.jxl"]).unwrap();
        assert_eq!(args.globals.verbose, 2);
    }

    #[test]
    fn quiet() {
        let args = Args::try_parse_from(["jxl-oxide", "input.jxl", "-q"]).unwrap();
        assert!(args.globals.quiet);

        let args = Args::try_parse_from(["jxl-oxide", "--quiet", "input.jxl"]).unwrap();
        assert!(args.globals.quiet);

        assert!(Args::try_parse_from(["jxl-oxide", "-q"]).is_err());

        let args = Args::try_parse_from(["jxl-oxide", "-Iq", "input.jxl"]).unwrap();
        assert!(args.globals.quiet);
    }

    #[test]
    fn verbose_quiet_conflicts() {
        assert!(Args::try_parse_from(["jxl-oxide", "input.jxl", "-q", "-v"]).is_err());
        assert!(Args::try_parse_from(["jxl-oxide", "-v", "-q", "input.jxl"]).is_err());
        assert!(Args::try_parse_from(["jxl-oxide", "--verbose", "-q", "input.jxl"]).is_err());
        assert!(Args::try_parse_from(["jxl-oxide", "-Ivvq", "input.jxl"]).is_err());
    }
}

mod color_encoding {
    use super::super::*;
    use jxl_oxide::{
        EnumColourEncoding, RenderingIntent,
        color::{ColourSpace, TransferFunction},
    };

    fn test_encoding_eq(a: &EnumColourEncoding, b: &EnumColourEncoding) -> bool {
        if a.colour_space != b.colour_space || a.rendering_intent != b.rendering_intent {
            false
        } else if a.colour_space == ColourSpace::Xyb {
            true
        } else if a.white_point != b.white_point || a.tf != b.tf {
            false
        } else if a.colour_space == ColourSpace::Grey {
            true
        } else {
            a.primaries == b.primaries
        }
    }

    #[test]
    fn with_presets() {
        let actual = parse_color_encoding("srgb").unwrap();
        let expected = EnumColourEncoding::srgb(RenderingIntent::Relative);
        assert!(test_encoding_eq(&actual, &expected));

        let actual = parse_color_encoding("display_p3").unwrap();
        let expected = EnumColourEncoding::display_p3(RenderingIntent::Relative);
        assert!(test_encoding_eq(&actual, &expected));

        let actual = parse_color_encoding("rec2020").unwrap();
        let expected = EnumColourEncoding {
            tf: TransferFunction::Bt709,
            ..EnumColourEncoding::bt2100_pq(RenderingIntent::Relative)
        };
        assert!(test_encoding_eq(&actual, &expected));
    }

    #[test]
    fn rec2100_needs_tf() {
        assert!(parse_color_encoding("rec2100").is_err());

        let actual = parse_color_encoding("rec2100,tf=pq").unwrap();
        let expected = EnumColourEncoding::bt2100_pq(RenderingIntent::Relative);
        assert!(test_encoding_eq(&actual, &expected));
    }

    #[test]
    fn parameters() {
        assert!(parse_color_encoding("type=gray").is_err());

        let actual = parse_color_encoding("type=gray,wp=d65,tf=srgb,intent=rel").unwrap();
        let expected = EnumColourEncoding::gray_srgb(RenderingIntent::Relative);
        assert!(test_encoding_eq(&actual, &expected));

        let actual = parse_color_encoding("type=xyb,intent=per").unwrap();
        let expected = EnumColourEncoding::xyb(RenderingIntent::Perceptual);
        assert!(test_encoding_eq(&actual, &expected));

        assert!(parse_color_encoding("type=xyb,tf=srgb,intent=per").is_err());
    }

    #[test]
    fn tfs() {
        let actual = parse_color_encoding("srgb,tf=2.2").unwrap();
        let expected = EnumColourEncoding::srgb_gamma22(RenderingIntent::Relative);
        assert!(test_encoding_eq(&actual, &expected));

        let actual = parse_color_encoding("srgb,tf=linear").unwrap();
        let expected = EnumColourEncoding {
            tf: TransferFunction::Linear,
            ..EnumColourEncoding::srgb(RenderingIntent::Relative)
        };
        assert!(test_encoding_eq(&actual, &expected));

        let actual = parse_color_encoding("srgb,tf=bt709").unwrap();
        let expected = EnumColourEncoding {
            tf: TransferFunction::Bt709,
            ..EnumColourEncoding::srgb(RenderingIntent::Relative)
        };
        assert!(test_encoding_eq(&actual, &expected));

        let actual = parse_color_encoding("display_p3,tf=hlg").unwrap();
        let expected = EnumColourEncoding {
            tf: TransferFunction::Hlg,
            ..EnumColourEncoding::display_p3(RenderingIntent::Relative)
        };
        assert!(test_encoding_eq(&actual, &expected));
    }
}
