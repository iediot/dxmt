#pragma once
/* ============================================================================
 * ml2100 -- halve a BC1/BC3 image in the compressed domain.
 *
 * The mip clamp (ml675/ml2000) saves memory by dropping top mip levels, so it
 * cannot touch a texture that HAS no lower level. A Unity 2D title created ~150
 * single-level BC1/BC3 textures up to 4096x2304 (full-screen art) next to
 * ~5,000 mipmapped ones, and the clamp, even at mipClampBC=1, could not keep the
 * scene load under the jetsam limit.
 *
 * For those textures the PHYSICAL Metal texture is created at half width and
 * half height and filled from this function: decode each 2x2 group of source
 * blocks (8x8 texels), box-filter to 4x4, re-encode one block. Sampling with
 * normalised coordinates is unaffected; the D3D descriptor stays LOGICAL.
 *
 * The encoder is a fast bounding-box one (diagonal chosen from the colour
 * covariance, palette indices picked by exact error against the palette the
 * decoder in dxmt_bcn.hpp will produce), which is plenty for content that is
 * being halved anyway. Averaging happens on the stored values, so sRGB content
 * is filtered in gamma space -- slightly darker edges than a linear filter,
 * not visible at half resolution.
 *
 * Pure C++, no Windows/Metal types, so it can be tested on the host.
 * ==========================================================================*/
#include "dxmt_bcn.hpp"
#include <cstddef>
#include <cstdint>

namespace dxmt {

/* Palette exactly as bcn_bc1_block() decodes it. */
static inline void bcn_enc_bc1_palette(uint16_t c0, uint16_t c1, bool three_colour, int p[4][3]) {
  uint8_t a[3], b[3];
  bcn_rgb565(c0, a);
  bcn_rgb565(c1, b);
  for (int i = 0; i < 3; i++) {
    p[0][i] = a[i];
    p[1][i] = b[i];
    if (three_colour) {
      p[2][i] = (a[i] + b[i] + 1) / 2;
      p[3][i] = 0;
    } else {
      p[2][i] = (2 * a[i] + b[i] + 1) / 3;
      p[3][i] = (a[i] + 2 * b[i] + 1) / 3;
    }
  }
}

static inline uint16_t bcn_enc_565(int r, int g, int b) {
  r = r < 0 ? 0 : r > 255 ? 255 : r;
  g = g < 0 ? 0 : g > 255 ? 255 : g;
  b = b < 0 ? 0 : b > 255 ? 255 : b;
  return (uint16_t)((((r * 31 + 127) / 255) << 11) | (((g * 63 + 127) / 255) << 5) | ((b * 31 + 127) / 255));
}

/* Encode 16 RGBA texels into an 8-byte BC1 colour block. With allow_alpha, a
 * texel whose alpha is below 128 selects 3-colour mode and index 3 (transparent
 * black), matching BC1's punch-through rule. BC3's colour half passes false:
 * D3D decodes it in 4-colour mode whatever the endpoint order. */
static inline void bcn_enc_bc1_block(const uint8_t px[64], uint8_t *blk, bool allow_alpha) {
  bool transparent[16];
  bool any_transparent = false;
  int lo[3] = {255, 255, 255}, hi[3] = {0, 0, 0}, sum[3] = {0, 0, 0}, n = 0;
  for (int t = 0; t < 16; t++) {
    transparent[t] = allow_alpha && px[t * 4 + 3] < 128;
    any_transparent |= transparent[t];
    if (transparent[t]) continue;
    for (int i = 0; i < 3; i++) {
      int v = px[t * 4 + i];
      lo[i] = v < lo[i] ? v : lo[i];
      hi[i] = v > hi[i] ? v : hi[i];
      sum[i] += v;
    }
    n++;
  }
  if (!n) {   /* fully transparent block */
    blk[0] = blk[1] = blk[2] = blk[3] = 0;
    blk[4] = blk[5] = blk[6] = blk[7] = 0xFF;
    return;
  }

  /* Pick the box diagonal that follows the colour trend: flip red and blue
   * against green when they are anti-correlated with it. */
  int cov_rg = 0, cov_bg = 0;
  for (int t = 0; t < 16; t++) {
    if (transparent[t]) continue;
    int dg = px[t * 4 + 1] * n - sum[1];
    cov_rg += (px[t * 4 + 0] * n - sum[0]) / 16 * dg / 16;
    cov_bg += (px[t * 4 + 2] * n - sum[2]) / 16 * dg / 16;
  }
  int e0[3] = {hi[0], hi[1], hi[2]}, e1[3] = {lo[0], lo[1], lo[2]};
  if (cov_rg < 0) { e0[0] = lo[0]; e1[0] = hi[0]; }
  if (cov_bg < 0) { e0[2] = lo[2]; e1[2] = hi[2]; }
  /* Inset by 1/16 of the range, as most fast encoders do, so the endpoints
   * are not wasted on the extremes. */
  for (int i = 0; i < 3; i++) {
    int inset = (e0[i] - e1[i]) / 16;
    e0[i] -= inset;
    e1[i] += inset;
  }
  uint16_t c0 = bcn_enc_565(e0[0], e0[1], e0[2]);
  uint16_t c1 = bcn_enc_565(e1[0], e1[1], e1[2]);

  /* 4-colour mode needs c0 > c1, 3-colour mode c0 <= c1. */
  bool three = any_transparent;
  if (three ? c0 > c1 : c0 < c1) { uint16_t s = c0; c0 = c1; c1 = s; }
  if (!three && c0 == c1) {
    /* Equal endpoints: index 0 is c0 in either mode, so write all zeros. */
    blk[0] = (uint8_t)c0; blk[1] = (uint8_t)(c0 >> 8);
    blk[2] = (uint8_t)c1; blk[3] = (uint8_t)(c1 >> 8);
    blk[4] = blk[5] = blk[6] = blk[7] = 0;
    return;
  }

  int pal[4][3];
  bcn_enc_bc1_palette(c0, c1, three, pal);
  uint32_t idx = 0;
  for (int t = 0; t < 16; t++) {
    uint32_t best = 3;
    if (!transparent[t]) {
      int best_err = 1 << 30;
      for (int k = 0; k < (three ? 3 : 4); k++) {
        int dr = px[t * 4 + 0] - pal[k][0], dg = px[t * 4 + 1] - pal[k][1], db = px[t * 4 + 2] - pal[k][2];
        int err = dr * dr + dg * dg + db * db;
        if (err < best_err) { best_err = err; best = (uint32_t)k; }
      }
    }
    idx |= best << (t * 2);
  }
  blk[0] = (uint8_t)c0; blk[1] = (uint8_t)(c0 >> 8);
  blk[2] = (uint8_t)c1; blk[3] = (uint8_t)(c1 >> 8);
  blk[4] = (uint8_t)idx; blk[5] = (uint8_t)(idx >> 8);
  blk[6] = (uint8_t)(idx >> 16); blk[7] = (uint8_t)(idx >> 24);
}

/* Encode the 16 alpha values of `px` into BC3's 8-byte alpha block, in the
 * 8-interpolant mode (a0 > a1) that bcn_scalar_block() decodes. */
static inline void bcn_enc_bc3_alpha(const uint8_t px[64], uint8_t *blk) {
  int lo = 255, hi = 0;
  for (int t = 0; t < 16; t++) {
    int a = px[t * 4 + 3];
    lo = a < lo ? a : lo;
    hi = a > hi ? a : hi;
  }
  blk[0] = (uint8_t)hi;
  blk[1] = (uint8_t)lo;
  uint64_t bits = 0;
  if (hi != lo) {   /* equal: every index 0 decodes to a0 in either mode */
    int pal[8];
    pal[0] = hi;
    pal[1] = lo;
    for (int k = 1; k <= 6; k++) pal[k + 1] = ((7 - k) * hi + k * lo + 3) / 7;
    for (int t = 0; t < 16; t++) {
      int a = px[t * 4 + 3], best = 0, best_err = 1 << 30;
      for (int k = 0; k < 8; k++) {
        int d = a - pal[k];
        if (d * d < best_err) { best_err = d * d; best = k; }
      }
      bits |= (uint64_t)best << (t * 3);
    }
  }
  for (int i = 0; i < 6; i++) blk[2 + i] = (uint8_t)(bits >> (8 * i));
}

/* Halve a BC1 (kind 1) or BC3 (kind 3) image. width and height are in texels
 * and must be multiples of 8; pitches are bytes per row of BLOCKS. Returns
 * false, writing nothing, for any other kind or size. */
static inline bool bcn_downscale_half(int kind, const uint8_t *src, size_t src_pitch, uint32_t width,
                                      uint32_t height, uint8_t *dst, size_t dst_pitch) {
  if ((kind != 1 && kind != 3) || (width & 7) || (height & 7) || !width || !height)
    return false;
  const uint32_t blk_bytes = kind == 1 ? 8u : 16u;
  const uint32_t out_bw = width / 8, out_bh = height / 8;
  uint8_t in[4][64], out[64];
  for (uint32_t by = 0; by < out_bh; by++) {
    for (uint32_t bx = 0; bx < out_bw; bx++) {
      /* The 2x2 source blocks behind output block (bx, by). */
      for (int q = 0; q < 4; q++) {
        const uint8_t *sblk = src + (size_t)(by * 2 + (q >> 1)) * src_pitch + (size_t)(bx * 2 + (q & 1)) * blk_bytes;
        if (kind == 1)
          bcn_bc1_block(sblk, in[q], true);
        else
          bcn_bc3_block(sblk, in[q]);
      }
      /* Box filter: output texel (ox, oy) averages source texels 2ox..2ox+1,
       * 2oy..2oy+1 of the 8x8 square, which all sit in one source block. */
      for (int oy = 0; oy < 4; oy++) {
        for (int ox = 0; ox < 4; ox++) {
          const uint8_t *s = in[(oy >> 1) * 2 + (ox >> 1)];
          int sx = (ox & 1) * 2, sy = (oy & 1) * 2;
          for (int c = 0; c < 4; c++) {
            int v = s[((sy) * 4 + sx) * 4 + c] + s[((sy) * 4 + sx + 1) * 4 + c] +
                    s[((sy + 1) * 4 + sx) * 4 + c] + s[((sy + 1) * 4 + sx + 1) * 4 + c];
            out[(oy * 4 + ox) * 4 + c] = (uint8_t)((v + 2) / 4);
          }
        }
      }
      uint8_t *dblk = dst + (size_t)by * dst_pitch + (size_t)bx * blk_bytes;
      if (kind == 1) {
        bcn_enc_bc1_block(out, dblk, true);
      } else {
        bcn_enc_bc3_alpha(out, dblk);
        bcn_enc_bc1_block(out, dblk + 8, false);
      }
    }
  }
  return true;
}

/* ml2101: the same halving for uncompressed texels with 8-bit channels, a
 * plain 2x2 box filter per byte, which is channel-order agnostic. bpp is the
 * bytes per texel: 4 for R8G8B8A8 / B8G8R8A8 / B8G8R8X8 (UNORM, SRGB and
 * TYPELESS forms), 1 for R8_UNORM / A8_UNORM (ml2113).
 *
 * ml2113: odd sizes too. The result is ceil(w/2) x ceil(h/2); a last column or
 * row with no partner averages the texels that exist. 547 MB of single-level
 * R8G8B8A8_SRGB art in Stick It to the Stickman was skipped only for an odd
 * width or height. Normalised sampling maps the half-size image onto the same
 * 0..1 range, off by at most half a source texel at the far edge. pitches are
 * bytes per row of texels. */
static inline bool texel8_downscale_half(uint32_t bpp, const uint8_t *src, size_t src_pitch, uint32_t width,
                                         uint32_t height, uint8_t *dst, size_t dst_pitch) {
  if (!width || !height || (bpp != 1 && bpp != 4))
    return false;
  const uint32_t ow = (width + 1) / 2, oh = (height + 1) / 2;
  for (uint32_t y = 0; y < oh; y++) {
    const uint32_t y0 = 2 * y, y1 = (2 * y + 1 < height) ? 2 * y + 1 : 2 * y;
    const uint8_t *r0 = src + (size_t)y0 * src_pitch, *r1 = src + (size_t)y1 * src_pitch;
    uint8_t *d = dst + (size_t)y * dst_pitch;
    for (uint32_t x = 0; x < ow; x++) {
      const uint32_t x0 = 2 * x, x1 = (2 * x + 1 < width) ? 2 * x + 1 : 2 * x;
      for (uint32_t c = 0; c < bpp; c++)
        d[x * bpp + c] = (uint8_t)((r0[x0 * bpp + c] + r0[x1 * bpp + c] + r1[x0 * bpp + c] + r1[x1 * bpp + c] + 2) / 4);
    }
  }
  return true;
}

static inline bool rgba8_downscale_half(const uint8_t *src, size_t src_pitch, uint32_t width, uint32_t height,
                                        uint8_t *dst, size_t dst_pitch) {
  return texel8_downscale_half(4, src, src_pitch, width, height, dst, dst_pitch);
}

} // namespace dxmt
