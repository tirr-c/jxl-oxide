//! Constants used by the various colorspaces.
//!
//! Values are in xy-chromaticity coordinates.
#![allow(clippy::excessive_precision)]

/// CIE illuminant D65.
///
/// This is the white point mainly used in the sRGB colorspace.
pub const ILLUMINANT_D65: [f32; 2] = [0.3127, 0.329];

/// CIE illuminant E (equal-energy).
pub const ILLUMINANT_E: [f32; 2] = [1.0 / 3.0, 1.0 / 3.0];

/// DCI-P3 illuminant.
pub const ILLUMINANT_DCI: [f32; 2] = [0.314, 0.351];

/// CIE illuminant D50.
///
/// xy-chromaticity value is computed so that the resulting `chad` tag matches that of libjxl.
pub(crate) const ILLUMINANT_D50: [f32; 2] = [0.345669, 0.358496];

/// Primaries used by the sRGB colorspace.
pub const PRIMARIES_SRGB: [[f32; 2]; 3] = [
    [0.639998686, 0.330010138],
    [0.300003784, 0.600003357],
    [0.150002046, 0.059997204],
];

/// Primaries specified in ITU-R BT.2100-2.
pub const PRIMARIES_BT2100: [[f32; 2]; 3] = [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]];

/// Primaries specified in SMPTE ST 428-1.
pub const PRIMARIES_P3: [[f32; 2]; 3] = [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]];
