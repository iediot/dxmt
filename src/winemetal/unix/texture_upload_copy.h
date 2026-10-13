/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Native copy into an already allocated/fenced suballocation. No temporary
 * image, allocation, GPU submission or wait is introduced by this operation. */
static inline int wmt_texture_upload_size(size_t rows, size_t depth, size_t row,
                                         size_t src_row, size_t src_image,
                                         size_t valid_row, size_t valid_image,
                                         size_t *size) {
  if (!rows || !depth || !row || rows > SIZE_MAX / row) return 0;
  const size_t image = rows * row;
  if (depth > SIZE_MAX / image || valid_row > row || valid_image > image) return 0;
  const size_t last = (src_row == row) ? valid_image : valid_row;
  size_t tail = 0;
  if (src_row != row) {
    if (src_row && rows - 1 > SIZE_MAX / src_row) return 0;
    tail = (rows - 1) * src_row;
  }
  if (tail > SIZE_MAX - last) return 0;
  tail += last;
  if (src_image && depth - 1 > (SIZE_MAX - tail) / src_image) return 0;
  *size = image * depth;
  return 1;
}

static inline void wmt_copy_texture_upload(void *dst, const void *src,
                                          size_t rows, size_t depth, size_t dst_row,
                                          size_t src_row, size_t src_image,
                                          size_t valid_row, size_t valid_image) {
  unsigned char *output = (unsigned char *)dst;
  const unsigned char *input = (const unsigned char *)src;
  const size_t dst_image = dst_row * rows;
  if (src_row == dst_row && valid_image == dst_image && src_image == dst_image) {
    memcpy(output, input, dst_image * depth);
    return;
  }
  for (size_t z = 0; z < depth; ++z) {
    if (src_row == dst_row) {
      memcpy(output + z * dst_image, input + z * src_image, valid_image);
      if (valid_image < dst_image)
        memset(output + z * dst_image + valid_image, 0, dst_image - valid_image);
    } else {
      for (size_t y = 0; y < rows; ++y) {
        memcpy(output + z * dst_image + y * dst_row, input + z * src_image + y * src_row, valid_row);
        if (valid_row < dst_row)
          memset(output + z * dst_image + y * dst_row + valid_row, 0, dst_row - valid_row);
      }
    }
  }
}
