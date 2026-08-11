//! AnyRes tiler for Granite Vision and similar per-tile models.
//!
//! Divides an image into a base tile (downsampled full image) plus
//! N detail tiles from a resolution-aware grid. Each tile is resized
//! to `tile_size x tile_size` and handed to the encoder independently.

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Configuration for AnyRes tiling, derived from the model's
/// `MultiModalProtocol::valid_tile_counts()` and the encoder's
/// expected input size.
#[derive(Debug, Clone)]
pub struct AnyResTileSpec {
    pub tile_size: usize,
    pub valid_tile_counts: Vec<usize>,
}

/// Result of tiling an image. Contains one base tile (downsampled full
/// image) plus zero or more detail tiles from the resolution grid.
#[derive(Debug)]
pub struct TiledImage {
    pub base_tile: Vec<u8>,
    pub detail_tiles: Vec<Vec<u8>>,
}

impl AnyResTileSpec {
    /// Select the optimal grid `(rows, cols)` for an image of given size.
    ///
    /// For each valid tile count N, enumerates factorizations `(r, c)` where
    /// `r * c <= N`, picks the one whose aspect ratio best matches the input.
    pub fn optimal_grid(&self, width: usize, height: usize) -> (usize, usize) {
        let aspect = width as f64 / height as f64;
        let mut best = (1usize, 1usize);
        let mut best_score = f64::MAX;

        for &count in &self.valid_tile_counts {
            for r in 1..=count {
                for c in 1..=count {
                    if r * c > count {
                        continue;
                    }
                    let grid_aspect = c as f64 / r as f64;
                    let score = (grid_aspect - aspect).abs();
                    if score < best_score || (score == best_score && r * c > best.0 * best.1) {
                        best_score = score;
                        best = (r, c);
                    }
                }
            }
        }
        best
    }

    /// Tile an image into base + detail tiles.
    ///
    /// The base tile is the full image downsampled to `tile_size x tile_size`.
    /// Detail tiles are cut from the image resized to fit the optimal grid.
    pub fn tile(&self, rgb: &[u8], width: usize, height: usize) -> TiledImage {
        let (rows, cols) = self.optimal_grid(width, height);
        let ts = self.tile_size;

        let base_tile = resize_rgb(rgb, width, height, ts, ts);

        let grid_w = cols * ts;
        let grid_h = rows * ts;
        let grid_rgb = resize_rgb(rgb, width, height, grid_w, grid_h);

        let mut detail_tiles = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                let mut tile = vec![0u8; ts * ts * 3];
                for y in 0..ts {
                    for x in 0..ts {
                        let src_y = r * ts + y;
                        let src_x = c * ts + x;
                        let src_idx = (src_y * grid_w + src_x) * 3;
                        let dst_idx = (y * ts + x) * 3;
                        tile[dst_idx] = grid_rgb[src_idx];
                        tile[dst_idx + 1] = grid_rgb[src_idx + 1];
                        tile[dst_idx + 2] = grid_rgb[src_idx + 2];
                    }
                }
                detail_tiles.push(tile);
            }
        }

        TiledImage {
            base_tile,
            detail_tiles,
        }
    }
}

/// Nearest-neighbor resize of RGB u8 row-major image.
fn resize_rgb(src: &[u8], src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Vec<u8> {
    let mut out = vec![0u8; dst_w * dst_h * 3];
    for y in 0..dst_h {
        let sy = y * src_h / dst_h;
        for x in 0..dst_w {
            let sx = x * src_w / dst_w;
            let si = (sy * src_w + sx) * 3;
            let di = (y * dst_w + x) * 3;
            out[di] = src[si];
            out[di + 1] = src[si + 1];
            out[di + 2] = src[si + 2];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> AnyResTileSpec {
        AnyResTileSpec {
            tile_size: 4,
            valid_tile_counts: vec![1, 2, 3, 4, 5, 6],
        }
    }

    #[test]
    fn square_image_selects_square_grid() {
        let s = spec();
        let (r, c) = s.optimal_grid(100, 100);
        assert_eq!(r, c, "square image should select a square grid");
    }

    #[test]
    fn wide_image_selects_1x2() {
        let s = spec();
        let (r, c) = s.optimal_grid(200, 100);
        assert_eq!((r, c), (1, 2));
    }

    #[test]
    fn tall_image_selects_2x1() {
        let s = spec();
        let (r, c) = s.optimal_grid(100, 200);
        assert_eq!((r, c), (2, 1));
    }

    #[test]
    fn tile_produces_base_plus_detail() {
        let s = spec();
        let rgb = vec![128u8; 8 * 4 * 3];
        let tiled = s.tile(&rgb, 8, 4);
        assert_eq!(tiled.base_tile.len(), 4 * 4 * 3);
        assert!(!tiled.detail_tiles.is_empty());
    }

    #[test]
    fn all_tiles_are_correct_size() {
        let s = spec();
        let rgb = vec![128u8; 12 * 8 * 3];
        let tiled = s.tile(&rgb, 12, 8);
        assert_eq!(tiled.base_tile.len(), 4 * 4 * 3);
        for tile in &tiled.detail_tiles {
            assert_eq!(tile.len(), 4 * 4 * 3);
        }
    }

    #[test]
    fn tile_count_within_valid_range() {
        let s = spec();
        let rgb = vec![128u8; 20 * 10 * 3];
        let tiled = s.tile(&rgb, 20, 10);
        let detail_count = tiled.detail_tiles.len();
        assert!(
            s.valid_tile_counts.contains(&detail_count),
            "detail tile count {detail_count} not in valid set {:?}",
            s.valid_tile_counts
        );
    }

    #[test]
    fn single_valid_count_works() {
        let s = AnyResTileSpec {
            tile_size: 4,
            valid_tile_counts: vec![1],
        };
        let rgb = vec![128u8; 8 * 8 * 3];
        let tiled = s.tile(&rgb, 8, 8);
        assert_eq!(tiled.detail_tiles.len(), 1); // 1 base + 1 detail = 2 tiles
    }

    #[test]
    fn base_tile_always_present() {
        let s = spec();
        let rgb = vec![128u8; 4 * 4 * 3];
        let tiled = s.tile(&rgb, 4, 4);
        assert!(!tiled.base_tile.is_empty());
    }
}
