#pragma once

#include "Metal.hpp"

#include "com/com_object.hpp"
#include "com/com_pointer.hpp"
#include "com/com_private_data.hpp"
#include "d3d9.h"
#include "d3d9_buffer_map.hpp"
#include "dxmt_buffer.hpp"
#include "dxmt_dynamic.hpp"

#include <vector>

namespace dxmt {

class MTLD3D9Device;

bool allocateD3D9BufferStorage(WMT::Device device, UINT length, D3DPOOL pool,
                              Rc<dxmt::Buffer> &buffer, void *&host, bool &watch, bool allow_alias = true);

// Large drawable buffers can expose guest-owned Shared Metal storage directly.
// DynamicBuffer recycles generations after GPU completion; smaller/unsupported
// buffers retain the mirror + write-watch publication path.
class MTLD3D9VertexBuffer final : public ComObject<IDirect3DVertexBuffer9> {
public:
  MTLD3D9VertexBuffer(
      MTLD3D9Device *device, UINT size, DWORD usage, DWORD fvf, D3DPOOL pool, void *host_ptr, bool write_watch,
      Rc<dxmt::Buffer> dxmt_buffer
  );
  ~MTLD3D9VertexBuffer();

  ULONG STDMETHODCALLTYPE AddRef() override;
  ULONG STDMETHODCALLTYPE Release() override;
  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void **ppvObject) override;

  // IDirect3DResource9
  HRESULT STDMETHODCALLTYPE GetDevice(IDirect3DDevice9 **ppDevice) override;
  HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID refguid, const void *pData, DWORD SizeOfData, DWORD Flags) override;
  HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID refguid, void *pData, DWORD *pSizeOfData) override;
  HRESULT STDMETHODCALLTYPE FreePrivateData(REFGUID refguid) override;
  DWORD STDMETHODCALLTYPE SetPriority(DWORD PriorityNew) override;
  DWORD STDMETHODCALLTYPE GetPriority() override;
  void STDMETHODCALLTYPE PreLoad() override;
  D3DRESOURCETYPE STDMETHODCALLTYPE GetType() override;

  // IDirect3DVertexBuffer9
  HRESULT STDMETHODCALLTYPE Lock(UINT OffsetToLock, UINT SizeToLock, void **ppbData, DWORD Flags) override;
  HRESULT STDMETHODCALLTYPE Unlock() override;
  HRESULT STDMETHODCALLTYPE GetDesc(D3DVERTEXBUFFER_DESC *pDesc) override;

  // The GPU buffer. Lock or refresh may rename it; do not cache the handle.
  WMT::Buffer
  metalBuffer() const {
    return m_dynamic->immediateName()->buffer();
  }
  // GPU virtual address for the manual-fetch VS variant, which pulls vertex
  // data through the [[buffer(16)]] vertex_buffers table rather than a
  // [[buffer(N)]] binding.
  uint64_t
  gpuAddress() const {
    return m_dynamic->immediateName()->gpuAddress();
  }
  // Current DynamicBuffer allocation. The draw path freezes the handle, the
  // GPU address and the Rc from ONE read, so the binding and the access that
  // retains it name the same allocation even if a later refresh renames the
  // buffer. Two reads could straddle that rename and bind one allocation while
  // retaining another.
  Rc<dxmt::BufferAllocation>
  immediateAllocation() const {
    return m_dynamic->immediateName();
  }
  // Whether the fallback GPU cache is stale; aliases never need refresh.
  bool
  isDirty() const {
    return m_dirty;
  }
  // Publish observed NOOVERWRITE pages in place, or rename and copy the whole
  // mirror when earlier draws may read changed bytes. The first refresh is
  // always whole so untouched bytes in each watched page have a known baseline.
  // Aliases are already visible in Shared storage, so this is a no-op.
  void refreshWholeMirror();
  // Called when a captured draw is queued, before asynchronous encoding.
  void markUsed(uint64_t seq) { m_lastUseSeq = seq; }
  // Raw access to the owning device: same rationale as
  // MTLD3D9Surface / MTLD3D9Texture: SetStreamSource's cross-device
  // check needs identity, not a public ref, on a hot path.
  MTLD3D9Device *
  deviceRaw() const {
    return m_device;
  }
  UINT
  size() const {
    return m_size;
  }
  // Host-mapped pointer to the authoritative vertex data. Callers must
  // null-check.
  const void *
  hostPointer() const {
    return m_hostPtr;
  }
  // FVF the buffer was created with (0 for a non-FVF buffer). ProcessVertices
  // reads it to synthesise the destination layout when the caller passes no
  // output declaration, matching DXVK (dst->Desc()->FVF).
  DWORD
  fvf() const {
    return m_fvf;
  }

private:
  MTLD3D9Device *m_device;
  // The dxmt::Buffer the DynamicBuffer wraps, held only to anchor the raw
  // Buffer pointer m_dynamic keeps (same role as d3d11's
  // D3D11Buffer::buffer_ behind its dynamic_); the current allocation and
  // the GPU-idle recycle FIFO live in m_dynamic. Declared before m_dynamic so
  // it outlives the wrapper's raw pointer at teardown.
  Rc<dxmt::Buffer> m_dxmtBuffer;
  // The DynamicBuffer recycling wrapper (the same one d3d11 uses,
  // d3d11_buffer.cpp). Owns the current allocation name and a FIFO of
  // retired allocations recycled after GPU completion. Lock renames aliases;
  // refresh renames mirrors. Each draw pins the name it actually captured.
  Rc<dxmt::DynamicBuffer> m_dynamic;
  // See MTLD3D9VertexBuffer::m_dirty and ::m_writeLocked.
  bool m_dirty = false;
  bool m_writeLocked = false;
  // Nested Lock/Unlock depth; outstanding alias pointers must stay valid.
  D3D9BufferLockCount m_lockCount;
  // Current app-visible memory. Aliased storage is owned by BufferAllocation;
  // only a fallback mirror is freed directly by this resource.
  void *m_hostPtr;
  bool m_aliased = false;
  uint64_t m_lastUseSeq = 0;
  HRESULT prepareAliasLock(DWORD flags);
  bool m_writeWatch; // Also selects VirtualFree, even if watch queries fail.
  D3D9BufferRefreshState m_refreshState;
  std::vector<void *> m_writtenPages; // Reused across draws; no hot-path allocation.
  UINT m_size;
  DWORD m_usage;
  DWORD m_fvf;
  D3DPOOL m_pool;
  DWORD m_priority = 0;
  // Same exactly-once-drop pattern as MTLD3D9Surface / MTLD3D9Texture:
  // the ctor self-pin must be released only on the FIRST pub->0
  // transition, otherwise a Get/Release cycle on a slot-pinned buffer
  // (m_vertexBuffers[N]) over-decrements priv and destructs.
  bool m_self_pinned = true;
  // Losable-resource accounting: see d3d9_surface.hpp.
  bool m_isLosable = false;

public:
  void markLosable();

private:
  ComPrivateData m_privateData;
};

// IDirect3DIndexBuffer9: same lifetime / pool / storage shape as
// MTLD3D9VertexBuffer; the only meaningful differences are the
// D3DFORMAT (D3DFMT_INDEX16 / D3DFMT_INDEX32) instead of FVF, the
// resource type, and the descriptor struct.
class MTLD3D9IndexBuffer final : public ComObject<IDirect3DIndexBuffer9> {
public:
  MTLD3D9IndexBuffer(
      MTLD3D9Device *device, UINT size, DWORD usage, D3DFORMAT format, D3DPOOL pool, void *host_ptr, bool write_watch,
      Rc<dxmt::Buffer> dxmt_buffer
  );
  ~MTLD3D9IndexBuffer();

  ULONG STDMETHODCALLTYPE AddRef() override;
  ULONG STDMETHODCALLTYPE Release() override;
  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void **ppvObject) override;

  // IDirect3DResource9
  HRESULT STDMETHODCALLTYPE GetDevice(IDirect3DDevice9 **ppDevice) override;
  HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID refguid, const void *pData, DWORD SizeOfData, DWORD Flags) override;
  HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID refguid, void *pData, DWORD *pSizeOfData) override;
  HRESULT STDMETHODCALLTYPE FreePrivateData(REFGUID refguid) override;
  DWORD STDMETHODCALLTYPE SetPriority(DWORD PriorityNew) override;
  DWORD STDMETHODCALLTYPE GetPriority() override;
  void STDMETHODCALLTYPE PreLoad() override;
  D3DRESOURCETYPE STDMETHODCALLTYPE GetType() override;

  // IDirect3DIndexBuffer9
  HRESULT STDMETHODCALLTYPE Lock(UINT OffsetToLock, UINT SizeToLock, void **ppbData, DWORD Flags) override;
  HRESULT STDMETHODCALLTYPE Unlock() override;
  HRESULT STDMETHODCALLTYPE GetDesc(D3DINDEXBUFFER_DESC *pDesc) override;

  // See MTLD3D9VertexBuffer::metalBuffer.
  WMT::Buffer
  metalBuffer() const {
    return m_dynamic->immediateName()->buffer();
  }
  // Byte offset of the current allocation within metalBuffer(). Always 0:
  // each DynamicBuffer allocation is its own MTLBuffer starting at 0. Kept
  // for caller compatibility (BuildDrawCapture and the index fan-remap path).
  uint64_t
  currentOffset() const {
    return 0;
  }
  D3DFORMAT
  indexFormat() const {
    return m_format;
  }
  // See MTLD3D9VertexBuffer::immediateAllocation.
  Rc<dxmt::BufferAllocation>
  immediateAllocation() const {
    return m_dynamic->immediateName();
  }
  // See MTLD3D9VertexBuffer for the refresh model.
  bool
  isDirty() const {
    return m_dirty;
  }
  void refreshWholeMirror();
  // Called when a captured draw is queued, before asynchronous encoding.
  void markUsed(uint64_t seq) { m_lastUseSeq = seq; }
  // Host pointer to the current generation (or the authoritative mirror). The index fan-remap path in d3d9_device.cpp reads the
  // source indices through it to remap them; callers must null-check.
  const void *
  hostPointer() const {
    return m_hostPtr;
  }
  MTLD3D9Device *
  deviceRaw() const {
    return m_device;
  }
  UINT
  size() const {
    return m_size;
  }

private:
  MTLD3D9Device *m_device;
  // See MTLD3D9VertexBuffer::m_dxmtBuffer / m_dynamic / m_hostPtr for the
  // per-map-mode lifecycle. m_dxmtBuffer is declared before m_dynamic so
  // it outlives the wrapper's raw Buffer pointer at teardown.
  Rc<dxmt::Buffer> m_dxmtBuffer;
  Rc<dxmt::DynamicBuffer> m_dynamic;
  void *m_hostPtr;
  bool m_aliased = false;
  uint64_t m_lastUseSeq = 0;
  HRESULT prepareAliasLock(DWORD flags);
  bool m_writeWatch; // Also selects VirtualFree, even if watch queries fail.
  D3D9BufferRefreshState m_refreshState;
  std::vector<void *> m_writtenPages; // Reused across draws; no hot-path allocation.
  // In fallback mode the mirror is authoritative. The GPU-side cache no
  // longer matches it; write-watch tracks actual writes, not the untrusted
  // range supplied by the application (d3d9_buffer_map.hpp).
  bool m_dirty = false;
  // A fallback write lock that is still outstanding. An application may hold a buffer
  // mapped across many draws and write each region just before the draw that
  // reads it, without ever re-locking, so the lock is the only announcement we
  // get and it arrives once. Clearing the dirty flag on the first refresh would
  // then leave every later write unpublished. While this is set, a refresh
  // re-uploads unconditionally.
  bool m_writeLocked = false;
  D3D9BufferLockCount m_lockCount;
  UINT m_size;
  DWORD m_usage;
  D3DFORMAT m_format;
  D3DPOOL m_pool;
  DWORD m_priority = 0;
  // See MTLD3D9VertexBuffer::m_self_pinned for the rationale.
  bool m_self_pinned = true;
  // Losable-resource accounting: see d3d9_surface.hpp.
  bool m_isLosable = false;

public:
  void markLosable();

private:
  ComPrivateData m_privateData;
};

} // namespace dxmt
