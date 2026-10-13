#pragma once
#include <cstdint>

#include "d3d9.h"

// Pure map-mode / lock-flag / dirty-range logic for vertex and index
// buffers, factored out of d3d9_buffer.cpp so it carries no device or Metal
// include surface. References: DXVK
// d3d9_common_buffer.h, DXVK d3d9_device.cpp, wined3d buffer.c.

namespace dxmt {

// Large drawable buffers default to guest-owned Shared Metal generations.
// NOOVERWRITE promises that new writes do not overlap earlier GPU readers;
// DISCARD renames, and a busy plain lock preserves contents in a fresh name.
// Small buffers, remote Metal, and DXMT_D9_BUFFER_ALIAS=0 retain the mirror
// and write-watch path. Lock ranges remain advisory in either mode.
//
// Platform history: translated writes to Metal-wired pages carrying translation
// state have previously faulted repeatedly. Direct mapping makes those pages
// app-visible again; the env switch preserves the staged workaround for such
// hosts. This is distinct from the write-watch protection faults eliminated by
// aliasing. Runtime profiling is needed to distinguish the two.

// Lock-flag sanitisation. The runtime silently drops flags that the
// pool/usage/device combination does not honour before any of them take
// effect.
inline DWORD
sanitize_buffer_lock_flags(DWORD flags, D3DPOOL pool, DWORD usage, bool swvp_only) {
  // DISCARD is honoured only when it rides alone.
  if ((flags & (D3DLOCK_DISCARD | D3DLOCK_NOOVERWRITE | D3DLOCK_READONLY)) != D3DLOCK_DISCARD)
    flags &= ~D3DLOCK_DISCARD;
  // DISCARD and NOOVERWRITE are honoured only in the DEFAULT pool, and never
  // on a software-vertex-processing-only device: native keeps the lock
  // pointer stable and the contents intact there (DXVK d3d9_device.cpp
  // strips both the same way; wine's d3d9 device tests assert the pinned
  // behaviour).
  if (pool != D3DPOOL_DEFAULT || swvp_only)
    flags &= ~(D3DLOCK_DISCARD | D3DLOCK_NOOVERWRITE);
  // DONOTWAIT is honoured only for non-DYNAMIC buffers.
  if (usage & D3DUSAGE_DYNAMIC)
    flags &= ~D3DLOCK_DONOTWAIT;
  // READONLY is honoured only in the MANAGED pool.
  if ((flags & D3DLOCK_READONLY) && pool != D3DPOOL_MANAGED)
    flags &= ~D3DLOCK_READONLY;
  return flags;
}

// SizeToLock IS NOT TRACKED, deliberately. The runtime hands back
// base + OffsetToLock and validates nothing, and shipped titles lock a small
// window then write far past it: one such title lost 233,077 of 524,288 index
// bytes that way, rendering a black world with no error anywhere. Windows
// tolerates it because the system-memory copy the application writes IS the
// buffer there, so a write outside the announced window still reaches the draw.
// Applications also keep writing after Unlock, which no bookkeeping finalised at
// Unlock can survive. So a write lock marks the buffer dirty; its flags govern
// whether a rename is needed, but only write-watch can narrow the copy;
// what a draw reads is decided from the draw, not from what the lock announced.
//
// Whether a Lock dirties the buffer at all. Post-sanitisation flags, so
// READONLY here is already the effective (MANAGED-only) flag.
// D3DLOCK_NO_DIRTY_UPDATE is a texture-only flag: D3D9 buffers always mark a
// write-lock dirty regardless of it (wined3d buffer.c invalidates on any write
// map and never consults the flag; only the texture map path checks it).
inline bool
buffer_lock_updates_dirty(DWORD flags) {
  return !(flags & D3DLOCK_READONLY);
}

// Lock ranges remain untrusted; write-watch observes the pages actually
// stored to. In-place publication additionally needs NOOVERWRITE's promise
// that no changed byte is read by an earlier draw. Remember unsafe locks even
// when a later lock is NOOVERWRITE. With no per-Unlock flags, conservatively
// retain an unsafe nested lock until the outermost Unlock.
struct D3D9BufferRefreshState {
  bool whole = true; // The initial allocation has never received the mirror.
  bool heldWhole = false;

  void
  writeLock(DWORD flags) {
    if (!(flags & D3DLOCK_NOOVERWRITE) || (flags & D3DLOCK_DISCARD))
      whole = heldWhole = true;
  }
  void
  unlockAll() {
    heldWhole = false;
  }
  void
  refreshed() {
    whole = heldWhole;
  }
};

// Nested-lock counter. D3D9 permits nested Lock/Unlock on one buffer, and only
// the outer Unlock means the application has stopped writing, which is what the
// refresh needs to know before it can call the device copy current. Unlock is
// clamped so an unbalanced one cannot underflow.
struct D3D9BufferLockCount {
  uint32_t count = 0;

  uint32_t
  increment() {
    return ++count;
  }

  uint32_t
  decrement() {
    if (count == 0)
      return 0;
    return --count;
  }
};

} // namespace dxmt
