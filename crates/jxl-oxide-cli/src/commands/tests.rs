mod color_encoding {
    use super::super::*;
    use jxl_oxide::{
        color::{ColourSpace, TransferFunction},
        EnumColourEncoding, RenderingIntent,
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
