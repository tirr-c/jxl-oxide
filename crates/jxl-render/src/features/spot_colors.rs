use jxl_grid::AlignedGrid;
use jxl_image::ExtraChannelType;

/// Renders a spot color channel onto color_channels
pub fn render_spot_color(
    mut color_channels: [&mut AlignedGrid<f32>; 3],
    ec_grid: &AlignedGrid<f32>,
    ec_ty: &ExtraChannelType,
) -> crate::Result<()> {
    let ExtraChannelType::SpotColour {
        red,
        green,
        blue,
        solidity,
    } = ec_ty
    else {
        return Err(crate::Error::NotSupported("EC type is not SpotColour"));
    };
    if color_channels.len() != 3 {
        return Ok(());
    }

    let spot_colors = [red, green, blue];
    let s = ec_grid.buf();

    (0..3).for_each(|c| {
        let channel = color_channels[c].buf_mut();
        let color = spot_colors[c];
        assert_eq!(channel.len(), s.len());

        (0..channel.len()).for_each(|i| {
            let mix = s[i] * solidity;
            channel[i] = mix * color + (1.0 - mix) * channel[i];
        });
    });
    Ok(())
}
