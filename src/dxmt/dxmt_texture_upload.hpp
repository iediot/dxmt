/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <cstddef>
#include <cstring>
namespace dxmt {
// Writes into a fresh, fenced staging suballocation. Guest padding is not texture data.
inline void copyTextureUpload(void *dst, const void *src, size_t rows, size_t depth,
                              size_t dst_row, size_t src_row, size_t src_image,
                              size_t valid_row, size_t valid_image) {
  auto *output = static_cast<unsigned char *>(dst);
  auto *input = static_cast<const unsigned char *>(src);
  const size_t dst_image = dst_row * rows;
  if (src_row == dst_row && valid_image == dst_image && src_image == dst_image) {
    memcpy(output, input, dst_image * depth);
    return;
  }
  for (size_t z = 0; z < depth; ++z) {
    if (src_row == dst_row) {
      memcpy(output + z * dst_image, input + z * src_image, valid_image);
      if (valid_image < dst_image) memset(output + z * dst_image + valid_image, 0, dst_image - valid_image);
    } else {
      for (size_t y = 0; y < rows; ++y) {
        memcpy(output + z * dst_image + y * dst_row, input + z * src_image + y * src_row, valid_row);
        if (valid_row < dst_row) memset(output + z * dst_image + y * dst_row + valid_row, 0, dst_row - valid_row);
      }
    }
  }
}
} // namespace dxmt
