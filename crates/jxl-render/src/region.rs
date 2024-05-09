use jxl_image::ImageHeader;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default, Hash)]
pub struct Region {
    pub left: i32,
    pub top: i32,
    pub width: u32,
    pub height: u32,
}

impl Region {
    #[inline]
    pub fn empty() -> Self {
        Self::default()
    }

    #[inline]
    pub fn with_size(width: u32, height: u32) -> Self {
        Self {
            left: 0,
            top: 0,
            width,
            height,
        }
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }

    #[inline]
    pub fn right(self) -> i32 {
        self.left.saturating_add_unsigned(self.width)
    }

    #[inline]
    pub fn bottom(self) -> i32 {
        self.top.saturating_add_unsigned(self.height)
    }

    pub fn contains(self, target: Region) -> bool {
        if target.is_empty() {
            return true;
        }

        self.left <= target.left
            && self.top <= target.top
            && self.right() >= target.right()
            && self.bottom() >= target.bottom()
    }

    pub fn translate(self, x: i32, y: i32) -> Self {
        Self {
            left: self.left + x,
            top: self.top + y,
            ..self
        }
    }

    pub fn intersection(self, rhs: Region) -> Self {
        if self.width == 0 || rhs.width == 0 || self.height == 0 || rhs.height == 0 {
            return Self {
                left: 0,
                top: 0,
                width: 0,
                height: 0,
            };
        }

        let mut ax = (self.left, self.right());
        let mut ay = (self.top, self.bottom());
        let mut bx = (rhs.left, rhs.right());
        let mut by = (rhs.top, rhs.bottom());
        if ax.0 > bx.0 {
            std::mem::swap(&mut ax, &mut bx);
        }
        if ay.0 > by.0 {
            std::mem::swap(&mut ay, &mut by);
        }

        if ax.1 <= bx.0 || ay.1 <= by.0 {
            Self {
                left: 0,
                top: 0,
                width: 0,
                height: 0,
            }
        } else {
            Self {
                left: bx.0,
                top: by.0,
                width: std::cmp::min(ax.1, bx.1).abs_diff(bx.0),
                height: std::cmp::min(ay.1, by.1).abs_diff(by.0),
            }
        }
    }

    #[inline]
    pub fn merge(self, other: Self) -> Self {
        if other.is_empty() {
            return self;
        }
        if self.is_empty() {
            return other;
        }

        let left = self.left.min(other.left);
        let top = self.top.min(other.top);
        let right = self
            .left
            .wrapping_add_unsigned(self.width)
            .max(other.left.wrapping_add_unsigned(other.width));
        let bottom = self
            .top
            .wrapping_add_unsigned(self.height)
            .max(other.top.wrapping_add_unsigned(other.height));
        let width = right.abs_diff(left);
        let height = bottom.abs_diff(top);

        Self {
            left,
            top,
            width,
            height,
        }
    }

    #[inline]
    pub fn pad(self, size: u32) -> Self {
        Self {
            left: self.left.saturating_sub_unsigned(size),
            top: self.top.saturating_sub_unsigned(size),
            width: self.width + size * 2,
            height: self.height + size * 2,
        }
    }

    #[inline]
    pub fn downsample(self, factor: u32) -> Self {
        if factor == 0 {
            return self;
        }

        let add = (1u32 << factor) - 1;
        let new_left = self.left >> factor;
        let new_top = self.top >> factor;
        let adj_width = self.width + self.left.abs_diff(new_left << factor);
        let adj_height = self.height + self.top.abs_diff(new_top << factor);
        Self {
            left: new_left,
            top: new_top,
            width: (adj_width + add) >> factor,
            height: (adj_height + add) >> factor,
        }
    }

    #[inline]
    pub fn downsample_separate(self, factor_x: u32, factor_y: u32) -> Self {
        if factor_x == 0 && factor_y == 0 {
            return self;
        }

        let add_x = (1u32 << factor_x) - 1;
        let new_left = self.left >> factor_x;
        let adj_width = self.width + self.left.abs_diff(new_left << factor_x);
        let add_y = (1u32 << factor_y) - 1;
        let new_top = self.top >> factor_y;
        let adj_height = self.height + self.top.abs_diff(new_top << factor_y);
        Self {
            left: new_left,
            top: new_top,
            width: (adj_width + add_x) >> factor_x,
            height: (adj_height + add_y) >> factor_y,
        }
    }

    #[inline]
    pub fn upsample(self, factor: u32) -> Self {
        self.upsample_separate(factor, factor)
    }

    #[inline]
    pub fn upsample_separate(self, factor_x: u32, factor_y: u32) -> Self {
        Self {
            left: self.left << factor_x,
            top: self.top << factor_y,
            width: self.width << factor_x,
            height: self.height << factor_y,
        }
    }

    pub(crate) fn container_aligned(self, grid_dim: u32) -> Self {
        debug_assert!(grid_dim.is_power_of_two());
        let add = grid_dim - 1;
        let mask = !add;
        let new_left = ((self.left as u32) & mask) as i32;
        let new_top = ((self.top as u32) & mask) as i32;
        let x_diff = self.left.abs_diff(new_left);
        let y_diff = self.top.abs_diff(new_top);
        Self {
            left: new_left,
            top: new_top,
            width: (self.width + x_diff + add) & mask,
            height: (self.height + y_diff + add) & mask,
        }
    }

    pub fn apply_orientation(self, image_header: &ImageHeader) -> Self {
        let image_width = image_header.width_with_orientation();
        let image_height = image_header.height_with_orientation();
        let (_, _, mut left, mut top) = image_header.metadata.apply_orientation(
            image_width,
            image_height,
            self.left,
            self.top,
            true,
        );
        let (_, _, mut right, mut bottom) = image_header.metadata.apply_orientation(
            image_width,
            image_height,
            self.left + self.width as i32 - 1,
            self.top + self.height as i32 - 1,
            true,
        );

        if left > right {
            std::mem::swap(&mut left, &mut right);
        }
        if top > bottom {
            std::mem::swap(&mut top, &mut bottom);
        }
        let width = right.abs_diff(left) + 1;
        let height = bottom.abs_diff(top) + 1;
        Self {
            left,
            top,
            width,
            height,
        }
    }
}
