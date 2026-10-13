#include "d3d9_buffer.hpp"
#include "d3d9_guest_alloc.hpp"
#include "d3d9_debug.hpp"

#include <algorithm>

#include "d3d9_device.hpp"
#include "d3d9_private_data.hpp"
#include "d3d9_resource_priority.hpp"
#include "log/log.hpp"
#include "wsi_platform.hpp"

#include "d3d9_census.hpp"

namespace dxmt {

namespace {

// Refresh-time rename shared by the vertex and index buffers: recycle a
// GPU-idle allocation (or mint one) and install it as the current
// name. Metal returns a null buffer on video-memory exhaustion, and a placed
// allocation keeps a non-null CPU pointer even then, so the plain immediate-
// memory null check downstream would not catch it. Reject the fresh
// allocation here, leaving the current name intact, so a null-backed name is
// never installed (a later draw would bind a null MTLBuffer) and Lock reports
// the failure instead of handing back a pointer into a dead allocation.
//
// coherent_seq must mean the GPU has completed through that seq, which is what
// m_completionEvent reports and what the upload rings already recycle their
// host blocks on. The refresh overwrites the recycled allocation with the CPU,
// so a reader draw that is encoded but not yet retired would be torn.
HRESULT
renameDynamicBuffer(MTLD3D9Device *device, DynamicBuffer *dynamic, uint64_t current_seq, uint64_t coherent_seq) {
  bool minted_fresh = false;
  auto fresh = dynamic->allocate(coherent_seq, &minted_fresh);
  if (!fresh.ptr() || !fresh->buffer())
    return D3DERR_OUTOFVIDEOMEMORY;
  // Only a mint adds to the footprint the next commit has to relieve; a recycle
  // reuses memory already accounted for.
  if (minted_fresh)
    device->noteDynamicRenameBytes(fresh->length());
  dynamic->updateImmediateName(current_seq, std::move(fresh), 0, false);
  return D3D_OK;
}

// Count events even in an all-aliased workload, where refresh never runs.
void
recordAlias(bool created, bool minted, bool discard, bool copied) {
  static const bool enabled = d9EnvGateEnabled("DXMT_D9_REFRESH_STATS");
  if (!enabled)
    return;
  struct Stats { uint64_t buffers = 0, generations = 0, discards = 0, copies = 0, events = 0; };
  static thread_local Stats stats;
  stats.buffers += created;
  stats.generations += minted;
  stats.discards += discard;
  stats.copies += copied;
  if (created || ++stats.events % 4000 == 0)
    Logger::info(str::format("d3d9 alias: buffers=", stats.buffers, " generations=", stats.generations,
                            " discard_switches=", stats.discards, " plain_lock_copies=", stats.copies));
}

// Per-thread totals avoid synchronization between devices. Disabled runs pay
// only the cached env gate, and the page array is reused by each buffer.
void
recordRefresh(bool inplace, bool fallback, uint64_t bytes, size_t pages) {
  static const bool enabled = d9EnvGateEnabled("DXMT_D9_REFRESH_STATS");
  if (!enabled)
    return;
  struct Stats {
    uint64_t refreshes = 0, whole = 0, inplace = 0, fallback = 0, bytes = 0, pages = 0;
  };
  static thread_local Stats stats;
  ++stats.refreshes;
  if (fallback)
    ++stats.fallback;
  else if (inplace)
    ++stats.inplace;
  else
    ++stats.whole;
  stats.bytes += bytes;
  stats.pages += pages;
  if (stats.refreshes % 4000 == 0)
    Logger::info(str::format("d3d9 refresh: refreshes=", stats.refreshes, " whole=", stats.whole,
                 " inplace=", stats.inplace, " fallback=", stats.fallback,
                 " bytes=", stats.bytes, " pages/inplace=",
                 stats.inplace ? double(stats.pages) / stats.inplace : 0.0));
}

bool
refreshMirror(MTLD3D9Device *device, DynamicBuffer *dynamic, void *mirror, UINT size,
              bool watched, D3D9BufferRefreshState &state, std::vector<void *> &pages,
              uint64_t current_seq, uint64_t coherent_seq) {
  bool fallback = !watched;
#if defined(_WIN32) && !defined(DXMT_MADEIRA)
  if (watched && !state.whole) {
    // A full-capacity array makes even a completely rewritten mirror fit.
    // Use the Windows page size, not DXMT_PAGE_SIZE or Metal's page size.
    if (pages.empty()) {
      SYSTEM_INFO info;
      GetSystemInfo(&info);
      pages.resize(size / info.dwPageSize + (size % info.dwPageSize != 0));
    }
    ULONG_PTR count = pages.size();
    ULONG granularity = 0;
    auto allocation = dynamic->immediateName();
    const UINT result = GetWriteWatch(WRITE_WATCH_FLAG_RESET, mirror, size, pages.data(), &count, &granularity);
    bool valid = result == 0 && count <= pages.size() && granularity != 0 &&
                 allocation->buffer() && !allocation->flags().test(BufferAllocationFlag::CpuInvisible);
    const uintptr_t base = reinterpret_cast<uintptr_t>(mirror);
    // Do not assume API ordering. Validate the entire result before copying;
    // a failed/partial query must never establish a new incremental baseline.
    if (valid) {
      std::sort(pages.begin(), pages.begin() + count, [](void *a, void *b) {
        return reinterpret_cast<uintptr_t>(a) < reinterpret_cast<uintptr_t>(b);
      });
      for (size_t i = 0; i < count; ++i) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(pages[i]);
        if (address < base || address - base >= size || (address - base) % granularity != 0)
          valid = false;
      }
    }
    if (valid) {
      uint64_t bytes = 0;
      for (size_t i = 0; i < count;) {
        const size_t start = reinterpret_cast<uintptr_t>(pages[i]) - base;
        size_t end = start + std::min<size_t>(granularity, size - start);
        while (++i < count && reinterpret_cast<uintptr_t>(pages[i]) - base == end)
          end += std::min<size_t>(granularity, size - end);
        // The current allocation equals the mirror except for watched pages.
        // NOOVERWRITE protects changed bytes from earlier draws; unchanged
        // bytes sharing their pages retain exactly the values those draws use.
        // updateContents is memcpy for mapped Shared storage and forwards to
        // the host for an allocation this process cannot address. Shared needs
        // no didModifyRange. Draw capture still registers/pins this allocation;
        // updateImmediateName only retires names, it does not publish CPU writes.
        allocation->updateContents(start, static_cast<char *>(mirror) + start, end - start,
                                   dynamic->immediateSuballocation());
        bytes += end - start;
      }
      state.refreshed();
      recordRefresh(true, false, bytes, count);
      return true;
    }
    fallback = true;
    // GetWriteWatch may have reset some pages even on failure. If the rename
    // fails too, the next draw MUST retry a whole copy rather than lose them.
    state.whole = true;
  }
#endif
  // First publication, DISCARD, plain write locks and unsupported/failed watch
  // queries keep the old rename + whole-mirror path. Lock's declared extent
  // cannot replace this: measured games write well beyond it.
  if (FAILED(renameDynamicBuffer(device, dynamic, current_seq, coherent_seq))) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      Logger::warn("d3d9: out of video memory refreshing a dynamic buffer; the draw reads stale contents");
    }
    return false;
  }
  dynamic->immediateName()->updateContents(0, mirror, size, dynamic->immediateSuballocation());
  state.refreshed();
#if defined(_WIN32) && !defined(DXMT_MADEIRA)
  // Initialization and whole copies establish the baseline. If resetting it
  // fails, keep using whole copies until a reset succeeds.
  if (watched && ResetWriteWatch(mirror, size) != 0) {
    state.whole = true;
    fallback = true;
  }
#endif
  recordRefresh(false, fallback, size, 0);
  return true;
}

} // namespace

MTLD3D9VertexBuffer::MTLD3D9VertexBuffer(
    MTLD3D9Device *device, UINT size, DWORD usage, DWORD fvf, D3DPOOL pool, void *host_ptr, bool write_watch,
    Rc<dxmt::Buffer> dxmt_buffer
) :
    m_device(device),
    m_dxmtBuffer(std::move(dxmt_buffer)),
    m_hostPtr(host_ptr),
    m_writeWatch(write_watch),
    m_size(size),
    m_usage(usage),
    m_fvf(fvf),
    m_pool(pool) {
  // External generations use DefaultCache because READONLY and CPU index
  // remapping read them. Fallback allocations retain their existing flags.
  m_aliased = m_dxmtBuffer->hasExternalBacking();
  m_dynamic = new dxmt::DynamicBuffer(m_dxmtBuffer.ptr(),
      m_aliased ? Flags<BufferAllocationFlag>{} : Flags<BufferAllocationFlag>{BufferAllocationFlag::CpuWriteCombined});
  if (m_aliased)
    recordAlias(true, true, false, false);
  // A non-DEFAULT buffer is dirty over its whole extent from creation so its
  // first bind or Unlock uploads the mirror; DEFAULT contents are undefined
  // until the app writes them (DXVK d3d9_common_buffer.cpp).
  if (!m_aliased && m_pool != D3DPOOL_DEFAULT)
    m_dirty = true;
  // Self-pin: same shape as MTLD3D9Surface / MTLD3D9Texture. The
  // override Release path drops the device pin after ComObject::
  // Release has decremented public to 0; the self-pin keeps `this`
  // alive across that window.
  AddRefPrivate();
}

MTLD3D9VertexBuffer::~MTLD3D9VertexBuffer() {
  // Frozen draw captures and encoder ref-tracker pins own BufferAllocations
  // independently of this wrapper. Their destructors release Metal before
  // freeing external guest backing, even if this resource has been released.
  // Only the fallback mirror belongs directly to the D3D9 resource.
  if (!m_aliased && m_hostPtr)
    guest_free_watch(m_hostPtr, m_writeWatch);
  if (m_isLosable)
    m_device->onLosableResourceDestroyed(m_size);
}

void
MTLD3D9VertexBuffer::refreshWholeMirror() {
  if (m_aliased || !m_dirty)
    return;
  if (refreshMirror(m_device, m_dynamic.ptr(), m_hostPtr, m_size, m_writeWatch, m_refreshState, m_writtenPages,
                    m_device->m_currentCmdSeq, m_device->m_cachedSignaled.load(std::memory_order_acquire)))
    // A held lock may write again without another Lock. Watch is reset at each
    // refresh, so the next draw publishes only the pages written since then.
    m_dirty = m_writeLocked;
}

void
MTLD3D9VertexBuffer::markLosable() {
  if (!m_isLosable) {
    m_isLosable = true;
    m_device->onLosableResourceCreated(m_size);
  }
}

ULONG STDMETHODCALLTYPE
MTLD3D9VertexBuffer::AddRef() {
  D3D9_CENSUS(D3D9_CENSUS_MTLD3D9VertexBuffer_AddRef);
  ULONG ref = ComObject::AddRef();
  if (ref == 1)
    m_device->AddRef();
  return ref;
}

ULONG STDMETHODCALLTYPE
MTLD3D9VertexBuffer::Release() {
  D3D9_CENSUS(D3D9_CENSUS_MTLD3D9VertexBuffer_Release);
  // D3D9 Release-at-0 clamp: handed out at public 0 while self-pinned / bound,
  // so guard the underflow before the decrement (DXVK clamps every device
  // child; same shape as the surface/swapchain/texture clamps).
  if (m_refCount.load() == 0)
    return 0;
  ULONG ref = ComObject::Release();
  if (ref == 0) {
    // Losable counter: decrement on pub->0, BEFORE m_device->Release
    // can destruct the device. See MTLD3D9Surface::Release for the
    // full rationale: Reset's counter check fires while bound
    // resources still have device priv refs, so the counter must
    // track app-pub-ref presence, not full destruct.
    if (m_isLosable) {
      m_isLosable = false;
      m_device->onLosableResourceDestroyed(m_size);
    }
    // The destructor releases the DynamicBuffer's Metal allocations (whose
    // dispose calls into the device's Metal device), so the device has to
    // outlive it. Drop the device pin LAST: capture it (ReleasePrivate frees
    // `this`), let the destructor run while the pin still keeps the device
    // alive, then release the pin, which may now free it.
    MTLD3D9Device *device = m_device;
    // Drop the ctor self-pin exactly once: same shape as
    // MTLD3D9Surface / MTLD3D9Texture. Subsequent Get/Release cycles
    // on a slot-bound buffer must not call ReleasePrivate again
    // (m_vertexBuffers[N] holds its own priv ref).
    if (m_self_pinned) {
      m_self_pinned = false;
      ReleasePrivate();
    }
    device->Release();
  }
  return ref;
}

HRESULT STDMETHODCALLTYPE
MTLD3D9VertexBuffer::QueryInterface(REFIID riid, void **ppvObject) {
  D3D9_CENSUS(D3D9_CENSUS_MTLD3D9VertexBuffer_QueryInterface);
  if (!ppvObject)
    return E_POINTER;
  *ppvObject = nullptr;

  if (riid == __uuidof(IUnknown) || riid == __uuidof(IDirect3DResource9) || riid == __uuidof(IDirect3DVertexBuffer9)) {
    *ppvObject = static_cast<IDirect3DVertexBuffer9 *>(this);
    AddRef();
    return S_OK;
  }
  return E_NOINTERFACE;
}

HRESULT STDMETHODCALLTYPE
MTLD3D9VertexBuffer::GetDevice(IDirect3DDevice9 **ppDevice) {
  D3D9_CENSUS(D3D9_CENSUS_MTLD3D9VertexBuffer_GetDevice);
  D9DeviceLock lock = m_device->LockDevice();
  if (!ppDevice)
    return D3DERR_INVALIDCALL;
  *ppDevice = ::dxmt::ref(static_cast<IDirect3DDevice9 *>(m_device));
  return D3D_OK;
}

HRESULT STDMETHODCALLTYPE
MTLD3D9VertexBuffer::SetPrivateData(REFGUID refguid, const void *pData, DWORD SizeOfData, DWORD Flags) {
  D3D9_CENSUS(D3D9_CENSUS_MTLD3D9VertexBuffer_SetPrivateData);
  D9DeviceLock lock = m_device->LockDevice();
  return D3D9SetPrivateData(m_privateData, refguid, pData, SizeOfData, Flags);
}

HRESULT STDMETHODCALLTYPE
MTLD3D9VertexBuffer::GetPrivateData(REFGUID refguid, void *pData, DWORD *pSizeOfData) {
  D3D9_CENSUS(D3D9_CENSUS_MTLD3D9VertexBuffer_GetPrivateData);
  D9DeviceLock lock = m_device->LockDevice();
  return D3D9GetPrivateData(m_privateData, refguid, pData, pSizeOfData);
}

HRESULT STDMETHODCALLTYPE
MTLD3D9VertexBuffer::FreePrivateData(REFGUID refguid) {
  D3D9_CENSUS(D3D9_CENSUS_MTLD3D9VertexBuffer_FreePrivateData);
  D9DeviceLock lock = m_device->LockDevice();
  return D3D9FreePrivateData(m_privateData, refguid);
}

DWORD STDMETHODCALLTYPE
MTLD3D9VertexBuffer::SetPriority(DWORD PriorityNew) {
  D3D9_CENSUS(D3D9_CENSUS_MTLD3D9VertexBuffer_SetPriority);
  D9DeviceLock lock = m_device->LockDevice();
  return D3D9SetResourcePriority(m_pool, m_priority, PriorityNew);
}

DWORD STDMETHODCALLTYPE
MTLD3D9VertexBuffer::GetPriority() {
  D3D9_CENSUS(D3D9_CENSUS_MTLD3D9VertexBuffer_GetPriority);
  D9DeviceLock lock = m_device->LockDevice();
  return m_priority;
}

void STDMETHODCALLTYPE
MTLD3D9VertexBuffer::PreLoad() {
  D3D9_CENSUS(D3D9_CENSUS_MTLD3D9VertexBuffer_PreLoad);
  D9DeviceLock lock = m_device->LockDevice();
  // Apple Silicon's unified memory makes residency hints a no-op.
}

D3DRESOURCETYPE STDMETHODCALLTYPE
MTLD3D9VertexBuffer::GetType() {
  D3D9_CENSUS(D3D9_CENSUS_MTLD3D9VertexBuffer_GetType);
  D9DeviceLock lock = m_device->LockDevice();
  return D3DRTYPE_VERTEXBUFFER;
}

// A nested lock must not strand an outstanding pointer in an old generation.
// Keep that generation, suppress nested DISCARD, and only for an unsafe write
// wait for its readers. The streaming NOOVERWRITE/READONLY path never waits.
HRESULT
MTLD3D9VertexBuffer::prepareAliasLock(DWORD flags) {
  if (flags & (D3DLOCK_NOOVERWRITE | D3DLOCK_READONLY))
    return D3D_OK;
  const uint64_t coherent = m_device->m_cachedSignaled.load(std::memory_order_acquire);
  if (m_lockCount.count) {
    if (m_lastUseSeq > coherent) {
      m_device->forceFlushAndCommit();
      if (!m_device->waitForGpuOrDeviceError(m_lastUseSeq))
        return D3DERR_DEVICELOST;
    }
    return D3D_OK;
  }
  const bool discard = flags & D3DLOCK_DISCARD;
  if (!discard && m_lastUseSeq <= coherent)
    return D3D_OK;
  bool minted = false;
  auto fresh = m_dynamic->allocate(coherent, &minted);
  if (!fresh.ptr() || !fresh->buffer() || !fresh->mappedMemory(0)) {
    // Preserve contents and all frozen draw allocations when direct storage
    // runs out. Install a normal mirror only after both fallback allocations
    // succeed; otherwise the failed Lock leaves this generation untouched.
    Rc<dxmt::Buffer> fallback;
    void *mirror = nullptr;
    bool watched = false;
    if (!allocateD3D9BufferStorage(m_device->m_metalDevice, m_size, m_pool, fallback, mirror, watched, false))
      return D3DERR_OUTOFVIDEOMEMORY;
    std::memcpy(mirror, m_hostPtr, m_size);
    m_dynamic = new dxmt::DynamicBuffer(fallback.ptr(), BufferAllocationFlag::CpuWriteCombined);
    m_dxmtBuffer = std::move(fallback);
    m_hostPtr = mirror;
    m_writeWatch = watched;
    m_aliased = false;
    m_refreshState = {};
    return D3D_OK;
  }
  if (!discard)
    std::memcpy(fresh->mappedMemory(0), m_hostPtr, m_size);
  if (minted)
    m_device->noteDynamicRenameBytes(fresh->length());
  // Retire conservatively at the current submission sequence, as the mirror
  // path does. This keeps DynamicBuffer's FIFO ordered even when DISCARD
  // replaces a generation which has never been drawn (last-use sequence 0).
  m_dynamic->updateImmediateName(m_device->m_currentCmdSeq, std::move(fresh), 0, false);
  // Buffer also holds its initial name. Move that anchor forward so trimming
  // an idle generation can really release its guest address space.
  m_dxmtBuffer->rename(m_dynamic->immediateName());
  m_lastUseSeq = 0;
  m_hostPtr = m_dynamic->immediateMappedMemory();
  recordAlias(false, minted, discard, !discard);
  return D3D_OK;
}

HRESULT STDMETHODCALLTYPE
MTLD3D9VertexBuffer::Lock(UINT OffsetToLock, UINT SizeToLock, void **ppbData, DWORD Flags) {
  D3D9_CENSUS(D3D9_CENSUS_MTLD3D9VertexBuffer_Lock);
  census::lockBytes(SizeToLock);
  D9DeviceLock lock = m_device->LockDevice();
  if (!ppbData)
    return D3DERR_INVALIDCALL;
  *ppbData = nullptr;
  // Flag sanitisation (d3d9_buffer_map.hpp): drop the flags this
  // pool/usage does not honour before any of them take effect. No bounds
  // validation: the runtime neither clamps nor rejects OffsetToLock /
  // SizeToLock, the returned pointer is simply base + offset.
  // CPU readers can use an alias without renaming, even in DEFAULT where
  // the mirror path historically drops READONLY. This also keeps
  // ProcessVertices source locks from copying a busy buffer.
  const bool alias_readonly = m_aliased && (Flags & D3DLOCK_READONLY);
  Flags = sanitize_buffer_lock_flags(Flags, m_pool, m_usage, m_device->canOnlySWVP());
  if (alias_readonly)
    Flags |= D3DLOCK_READONLY;
  if (!m_hostPtr)
    return D3DERR_INVALIDCALL;
  if (m_aliased) {
    if (HRESULT hr = prepareAliasLock(Flags); FAILED(hr))
      return hr;
  }
  // Record overwrite permission, never the declared byte range: only write-
  // watch can tell which pages actually changed (d3d9_buffer_map.hpp).
  if (!m_aliased && buffer_lock_updates_dirty(Flags)) {
    m_dirty = true;
    m_writeLocked = true;
    m_refreshState.writeLock(Flags);
  }
  m_lockCount.increment();
  *ppbData = static_cast<char *>(m_hostPtr) + OffsetToLock;
  return D3D_OK;
}

HRESULT STDMETHODCALLTYPE
MTLD3D9VertexBuffer::Unlock() {
  D3D9_CENSUS(D3D9_CENSUS_MTLD3D9VertexBuffer_Unlock);
  D9DeviceLock lock = m_device->LockDevice();
  // Aliased writes already reach Shared storage; only lock depth changes.
  // For mirrors, nothing uploads here. The draw that reads the buffer
  // refreshes it, because an application may still be writing through the
  // pointer it was handed: a lock is advisory about its extent and about its
  // lifetime both, so anything finalised at Unlock can already be stale.
  if (m_lockCount.decrement() == 0) {
    m_writeLocked = false;
    m_refreshState.unlockAll();
  }
  return D3D_OK;
}

HRESULT STDMETHODCALLTYPE
MTLD3D9VertexBuffer::GetDesc(D3DVERTEXBUFFER_DESC *pDesc) {
  D3D9_CENSUS(D3D9_CENSUS_MTLD3D9VertexBuffer_GetDesc);
  D9DeviceLock lock = m_device->LockDevice();
  if (!pDesc)
    return D3DERR_INVALIDCALL;
  pDesc->Format = D3DFMT_VERTEXDATA;
  pDesc->Type = D3DRTYPE_VERTEXBUFFER;
  pDesc->Usage = m_usage;
  pDesc->Pool = m_pool;
  pDesc->Size = m_size;
  pDesc->FVF = m_fvf;
  return D3D_OK;
}

// ============================================================
// MTLD3D9IndexBuffer
// ============================================================

MTLD3D9IndexBuffer::MTLD3D9IndexBuffer(
    MTLD3D9Device *device, UINT size, DWORD usage, D3DFORMAT format, D3DPOOL pool, void *host_ptr, bool write_watch,
    Rc<dxmt::Buffer> dxmt_buffer
) :
    m_device(device),
    m_dxmtBuffer(std::move(dxmt_buffer)),
    m_hostPtr(host_ptr),
    m_writeWatch(write_watch),
    m_size(size),
    m_usage(usage),
    m_format(format),
    m_pool(pool) {
  m_aliased = m_dxmtBuffer->hasExternalBacking();
  m_dynamic = new dxmt::DynamicBuffer(m_dxmtBuffer.ptr(),
      m_aliased ? Flags<BufferAllocationFlag>{} : Flags<BufferAllocationFlag>{BufferAllocationFlag::CpuWriteCombined});
  if (m_aliased)
    recordAlias(true, true, false, false);
  // See MTLD3D9VertexBuffer's ctor: a non-DEFAULT buffer starts wholly dirty.
  if (!m_aliased && m_pool != D3DPOOL_DEFAULT)
    m_dirty = true;
  AddRefPrivate();
}

MTLD3D9IndexBuffer::~MTLD3D9IndexBuffer() {
  // See MTLD3D9VertexBuffer::~MTLD3D9VertexBuffer: same shape. The
  // DynamicBuffer allocations release via m_dynamic (before m_dxmtBuffer);
  // only the host mirror is freed here.
  if (!m_aliased && m_hostPtr)
    guest_free_watch(m_hostPtr, m_writeWatch);
  if (m_isLosable)
    m_device->onLosableResourceDestroyed(m_size);
}

void
MTLD3D9IndexBuffer::refreshWholeMirror() {
  if (m_aliased || !m_dirty)
    return;
  if (refreshMirror(m_device, m_dynamic.ptr(), m_hostPtr, m_size, m_writeWatch, m_refreshState, m_writtenPages,
                    m_device->m_currentCmdSeq, m_device->m_cachedSignaled.load(std::memory_order_acquire)))
    // A held lock may write again without another Lock. Watch is reset at each
    // refresh, so the next draw publishes only the pages written since then.
    m_dirty = m_writeLocked;
}

void
MTLD3D9IndexBuffer::markLosable() {
  if (!m_isLosable) {
    m_isLosable = true;
    m_device->onLosableResourceCreated(m_size);
  }
}

ULONG STDMETHODCALLTYPE
MTLD3D9IndexBuffer::AddRef() {
  D3D9_CENSUS(D3D9_CENSUS_MTLD3D9IndexBuffer_AddRef);
  ULONG ref = ComObject::AddRef();
  if (ref == 1)
    m_device->AddRef();
  return ref;
}

ULONG STDMETHODCALLTYPE
MTLD3D9IndexBuffer::Release() {
  D3D9_CENSUS(D3D9_CENSUS_MTLD3D9IndexBuffer_Release);
  // D3D9 Release-at-0 clamp (see MTLD3D9VertexBuffer::Release).
  if (m_refCount.load() == 0)
    return 0;
  ULONG ref = ComObject::Release();
  if (ref == 0) {
    if (m_isLosable) {
      m_isLosable = false;
      m_device->onLosableResourceDestroyed(m_size);
    }
    // The destructor releases the DynamicBuffer's Metal allocations, so the
    // device has to outlive it (see MTLD3D9VertexBuffer::Release): drop the
    // device pin last, after the destructor has run.
    MTLD3D9Device *device = m_device;
    if (m_self_pinned) {
      m_self_pinned = false;
      ReleasePrivate();
    }
    device->Release();
  }
  return ref;
}

HRESULT STDMETHODCALLTYPE
MTLD3D9IndexBuffer::QueryInterface(REFIID riid, void **ppvObject) {
  D3D9_CENSUS(D3D9_CENSUS_MTLD3D9IndexBuffer_QueryInterface);
  if (!ppvObject)
    return E_POINTER;
  *ppvObject = nullptr;

  if (riid == __uuidof(IUnknown) || riid == __uuidof(IDirect3DResource9) || riid == __uuidof(IDirect3DIndexBuffer9)) {
    *ppvObject = static_cast<IDirect3DIndexBuffer9 *>(this);
    AddRef();
    return S_OK;
  }
  return E_NOINTERFACE;
}

HRESULT STDMETHODCALLTYPE
MTLD3D9IndexBuffer::GetDevice(IDirect3DDevice9 **ppDevice) {
  D3D9_CENSUS(D3D9_CENSUS_MTLD3D9IndexBuffer_GetDevice);
  D9DeviceLock lock = m_device->LockDevice();
  if (!ppDevice)
    return D3DERR_INVALIDCALL;
  *ppDevice = ::dxmt::ref(static_cast<IDirect3DDevice9 *>(m_device));
  return D3D_OK;
}

HRESULT STDMETHODCALLTYPE
MTLD3D9IndexBuffer::SetPrivateData(REFGUID refguid, const void *pData, DWORD SizeOfData, DWORD Flags) {
  D3D9_CENSUS(D3D9_CENSUS_MTLD3D9IndexBuffer_SetPrivateData);
  D9DeviceLock lock = m_device->LockDevice();
  return D3D9SetPrivateData(m_privateData, refguid, pData, SizeOfData, Flags);
}

HRESULT STDMETHODCALLTYPE
MTLD3D9IndexBuffer::GetPrivateData(REFGUID refguid, void *pData, DWORD *pSizeOfData) {
  D3D9_CENSUS(D3D9_CENSUS_MTLD3D9IndexBuffer_GetPrivateData);
  D9DeviceLock lock = m_device->LockDevice();
  return D3D9GetPrivateData(m_privateData, refguid, pData, pSizeOfData);
}

HRESULT STDMETHODCALLTYPE
MTLD3D9IndexBuffer::FreePrivateData(REFGUID refguid) {
  D3D9_CENSUS(D3D9_CENSUS_MTLD3D9IndexBuffer_FreePrivateData);
  D9DeviceLock lock = m_device->LockDevice();
  return D3D9FreePrivateData(m_privateData, refguid);
}

DWORD STDMETHODCALLTYPE
MTLD3D9IndexBuffer::SetPriority(DWORD PriorityNew) {
  D3D9_CENSUS(D3D9_CENSUS_MTLD3D9IndexBuffer_SetPriority);
  D9DeviceLock lock = m_device->LockDevice();
  return D3D9SetResourcePriority(m_pool, m_priority, PriorityNew);
}

DWORD STDMETHODCALLTYPE
MTLD3D9IndexBuffer::GetPriority() {
  D3D9_CENSUS(D3D9_CENSUS_MTLD3D9IndexBuffer_GetPriority);
  D9DeviceLock lock = m_device->LockDevice();
  return m_priority;
}

void STDMETHODCALLTYPE
MTLD3D9IndexBuffer::PreLoad() {
  D3D9_CENSUS(D3D9_CENSUS_MTLD3D9IndexBuffer_PreLoad);
  D9DeviceLock lock = m_device->LockDevice();
  // Apple Silicon's unified memory makes residency hints a no-op.
}

D3DRESOURCETYPE STDMETHODCALLTYPE
MTLD3D9IndexBuffer::GetType() {
  D3D9_CENSUS(D3D9_CENSUS_MTLD3D9IndexBuffer_GetType);
  D9DeviceLock lock = m_device->LockDevice();
  return D3DRTYPE_INDEXBUFFER;
}

// A nested lock must not strand an outstanding pointer in an old generation.
// Keep that generation, suppress nested DISCARD, and only for an unsafe write
// wait for its readers. The streaming NOOVERWRITE/READONLY path never waits.
HRESULT
MTLD3D9IndexBuffer::prepareAliasLock(DWORD flags) {
  if (flags & (D3DLOCK_NOOVERWRITE | D3DLOCK_READONLY))
    return D3D_OK;
  const uint64_t coherent = m_device->m_cachedSignaled.load(std::memory_order_acquire);
  if (m_lockCount.count) {
    if (m_lastUseSeq > coherent) {
      m_device->forceFlushAndCommit();
      if (!m_device->waitForGpuOrDeviceError(m_lastUseSeq))
        return D3DERR_DEVICELOST;
    }
    return D3D_OK;
  }
  const bool discard = flags & D3DLOCK_DISCARD;
  if (!discard && m_lastUseSeq <= coherent)
    return D3D_OK;
  bool minted = false;
  auto fresh = m_dynamic->allocate(coherent, &minted);
  if (!fresh.ptr() || !fresh->buffer() || !fresh->mappedMemory(0)) {
    // Preserve contents and all frozen draw allocations when direct storage
    // runs out. Install a normal mirror only after both fallback allocations
    // succeed; otherwise the failed Lock leaves this generation untouched.
    Rc<dxmt::Buffer> fallback;
    void *mirror = nullptr;
    bool watched = false;
    if (!allocateD3D9BufferStorage(m_device->m_metalDevice, m_size, m_pool, fallback, mirror, watched, false))
      return D3DERR_OUTOFVIDEOMEMORY;
    std::memcpy(mirror, m_hostPtr, m_size);
    m_dynamic = new dxmt::DynamicBuffer(fallback.ptr(), BufferAllocationFlag::CpuWriteCombined);
    m_dxmtBuffer = std::move(fallback);
    m_hostPtr = mirror;
    m_writeWatch = watched;
    m_aliased = false;
    m_refreshState = {};
    return D3D_OK;
  }
  if (!discard)
    std::memcpy(fresh->mappedMemory(0), m_hostPtr, m_size);
  if (minted)
    m_device->noteDynamicRenameBytes(fresh->length());
  // Retire conservatively at the current submission sequence, as the mirror
  // path does. This keeps DynamicBuffer's FIFO ordered even when DISCARD
  // replaces a generation which has never been drawn (last-use sequence 0).
  m_dynamic->updateImmediateName(m_device->m_currentCmdSeq, std::move(fresh), 0, false);
  // Buffer also holds its initial name. Move that anchor forward so trimming
  // an idle generation can really release its guest address space.
  m_dxmtBuffer->rename(m_dynamic->immediateName());
  m_lastUseSeq = 0;
  m_hostPtr = m_dynamic->immediateMappedMemory();
  recordAlias(false, minted, discard, !discard);
  return D3D_OK;
}

HRESULT STDMETHODCALLTYPE
MTLD3D9IndexBuffer::Lock(UINT OffsetToLock, UINT SizeToLock, void **ppbData, DWORD Flags) {
  D3D9_CENSUS(D3D9_CENSUS_MTLD3D9IndexBuffer_Lock);
  census::lockBytes(SizeToLock);
  D9DeviceLock lock = m_device->LockDevice();
  // Same shape as MTLD3D9VertexBuffer::Lock: see the rationale there
  // for the flag sanitisation, DISCARD / NOOVERWRITE semantics, the
  // plain-map GPU sync, and the mode-aware lockable base.
  if (!ppbData)
    return D3DERR_INVALIDCALL;
  *ppbData = nullptr;
  // CPU readers can use an alias without renaming, even in DEFAULT where
  // the mirror path historically drops READONLY. This also keeps
  // ProcessVertices source locks from copying a busy buffer.
  const bool alias_readonly = m_aliased && (Flags & D3DLOCK_READONLY);
  Flags = sanitize_buffer_lock_flags(Flags, m_pool, m_usage, m_device->canOnlySWVP());
  if (alias_readonly)
    Flags |= D3DLOCK_READONLY;
  if (!m_hostPtr)
    return D3DERR_INVALIDCALL;
  if (m_aliased) {
    if (HRESULT hr = prepareAliasLock(Flags); FAILED(hr))
      return hr;
  }
  // Record overwrite permission, never the declared byte range: only write-
  // watch can tell which pages actually changed (d3d9_buffer_map.hpp).
  if (!m_aliased && buffer_lock_updates_dirty(Flags)) {
    m_dirty = true;
    m_writeLocked = true;
    m_refreshState.writeLock(Flags);
  }
  m_lockCount.increment();
  *ppbData = static_cast<char *>(m_hostPtr) + OffsetToLock;
  return D3D_OK;
}

HRESULT STDMETHODCALLTYPE
MTLD3D9IndexBuffer::Unlock() {
  D3D9_CENSUS(D3D9_CENSUS_MTLD3D9IndexBuffer_Unlock);
  D9DeviceLock lock = m_device->LockDevice();
  // See MTLD3D9VertexBuffer::Unlock: the draw refreshes, not this.
  if (m_lockCount.decrement() == 0) {
    m_writeLocked = false;
    m_refreshState.unlockAll();
  }
  return D3D_OK;
}

HRESULT STDMETHODCALLTYPE
MTLD3D9IndexBuffer::GetDesc(D3DINDEXBUFFER_DESC *pDesc) {
  D3D9_CENSUS(D3D9_CENSUS_MTLD3D9IndexBuffer_GetDesc);
  D9DeviceLock lock = m_device->LockDevice();
  if (!pDesc)
    return D3DERR_INVALIDCALL;
  pDesc->Format = m_format;
  pDesc->Type = D3DRTYPE_INDEXBUFFER;
  pDesc->Usage = m_usage;
  pDesc->Pool = m_pool;
  pDesc->Size = m_size;
  return D3D_OK;
}

} // namespace dxmt
