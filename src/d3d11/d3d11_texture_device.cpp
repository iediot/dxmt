#include "com/com_pointer.hpp"
#include "d3d11_device.hpp"
#include "d3d11_enumerable.hpp"
#include "d3d11_view.hpp"
#include "dxmt_dynamic.hpp"
#include "dxmt_staging.hpp"
#include "dxmt_texture.hpp"
#include "d3d11_resource.hpp"
#include "util_win32_compat.h"
#include "config/config.hpp"   /* ml675: d3d11.mipClampBC */
#include "d3d11_mip_clamp_policy.hpp"   /* ml2000: d3d11.mipClampAuto */
#include "dxmt_bcn_downscale.hpp"       /* ml2100: single-level BC downscale */
#include <vector>
#include <algorithm>
#include "util_env.hpp"
#include "util_madeira_switch.hpp"
#include "winemetal.h"
#include <atomic>
#include <cstring>

namespace dxmt {

#pragma region DeviceTexture

template <typename tag_texture>
class DeviceTexture : public TResourceBase<tag_texture, IMTLMinLODClampable> {
private:
  Rc<Texture> underlying_texture_;
  Rc<RenamableTexturePool> renamable_;
  /* ml675: top mip levels dropped from the PHYSICAL Metal texture. The D3D
   * descriptor this object reports stays LOGICAL, so GetDesc keeps telling the
   * application the size it asked for. ml670 got this wrong -- it handed the
   * shrunken descriptor to the resource, so a 2048x2048 texture reported itself
   * as 512x512 and any UI computing layout from GetDesc came out wrong. */
  uint32_t mip_bias_ = 0;
  float min_lod = 0.0;
  D3DKMT_HANDLE local_kmt_ = 0;
  D3DKMT_HANDLE global_kmt_ = 0;

  using SRVBase =
      TResourceViewBase<tag_shader_resource_view<DeviceTexture<tag_texture>>>;
  class TextureSRV : public SRVBase {
  private:
    TextureViewKey view_key_;

  public:
    TextureSRV(TextureViewKey view_key,
               const tag_shader_resource_view<>::DESC1 *pDesc,
               DeviceTexture *pResource, MTLD3D11Device *pDevice)
        : SRVBase(pDesc, pResource, pDevice), view_key_(view_key) {}

    Rc<Buffer> buffer() final { return {}; };
    Rc<Texture> texture() final { return this->resource->underlying_texture_; };
    unsigned viewId() final { return view_key_;};
    BufferSlice bufferSlice() final { return {};}
  };

  using UAVBase =
      TResourceViewBase<tag_unordered_access_view<DeviceTexture<tag_texture>>>;
  class TextureUAV : public UAVBase {
  private:
    TextureViewKey view_key_;

  public:
    TextureUAV(TextureViewKey view_key,
               const tag_unordered_access_view<>::DESC1 *pDesc,
               DeviceTexture *pResource, MTLD3D11Device *pDevice)
        : UAVBase(pDesc, pResource, pDevice), view_key_(view_key){}

    Rc<Buffer> buffer() final { return {}; };
    Rc<Texture> texture() final { return this->resource->underlying_texture_; };
    unsigned viewId() final { return view_key_;};
    BufferSlice bufferSlice() final { return {};}
    Rc<Buffer> counter() final { return {}; };
  };

  using RTVBase =
      TResourceViewBase<tag_render_target_view<DeviceTexture<tag_texture>>>;
  class TextureRTV : public RTVBase {
  private:
    TextureViewKey view_key_;
    WMTPixelFormat view_format_;
    MTL_RENDER_PASS_ATTACHMENT_DESC attachment_desc;

  public:
    TextureRTV(
        TextureViewKey view_key, WMTPixelFormat view_format, const tag_render_target_view<>::DESC1 *pDesc,
        DeviceTexture *pResource, MTLD3D11Device *pDevice, const MTL_RENDER_PASS_ATTACHMENT_DESC &mtl_rtv_desc
    ) :
        RTVBase(pDesc, pResource, pDevice),
        view_key_(view_key),
        view_format_(view_format),
        attachment_desc(mtl_rtv_desc) {}

    WMTPixelFormat
    pixelFormat() final {
      return view_format_;
    }

    MTL_RENDER_PASS_ATTACHMENT_DESC &description() final {
      return attachment_desc;
    };

    Rc<Texture> texture() final {
      return this->resource->underlying_texture_;
    }

    unsigned viewId() final {
      return view_key_;
    }
  };

  using DSVBase =
      TResourceViewBase<tag_depth_stencil_view<DeviceTexture<tag_texture>>>;
  class TextureDSV : public DSVBase {
  private:
    TextureViewKey view_key_;
    WMTPixelFormat view_format_;
    MTL_RENDER_PASS_ATTACHMENT_DESC attachment_desc;

  public:
    TextureDSV(
        TextureViewKey view_key, WMTPixelFormat view_format, const tag_depth_stencil_view<>::DESC1 *pDesc,
        DeviceTexture *pResource, MTLD3D11Device *pDevice, const MTL_RENDER_PASS_ATTACHMENT_DESC &attachment_desc
    ) :
        DSVBase(pDesc, pResource, pDevice),
        view_key_(view_key),
        view_format_(view_format),
        attachment_desc(attachment_desc) {}

    WMTPixelFormat
    pixelFormat() final {
      return view_format_;
    }

    MTL_RENDER_PASS_ATTACHMENT_DESC &description() final {
      return attachment_desc;
    };

    UINT readonlyFlags() final {
      return this->desc.Flags;
    };

    Rc<Texture> texture() final {
      return this->resource->underlying_texture_;
    }

    unsigned viewId() final {
      return view_key_;
    }

    dxmt::Rc<dxmt::RenamableTexturePool> renamable() final {
      return this->resource->renamable_;
    }
  };

public:
  DeviceTexture(const tag_texture::DESC1 *pDesc, Rc<Texture> &&u_texture, MTLD3D11Device *pDevice) :
      TResourceBase<tag_texture, IMTLMinLODClampable>(*pDesc, pDevice),
      underlying_texture_(std::move(u_texture)) {}

  DeviceTexture(
      const tag_texture::DESC1 *pDesc, Rc<Texture> &&u_texture, Rc<RenamableTexturePool> &&renamable,
      MTLD3D11Device *pDevice
  ) :
      TResourceBase<tag_texture, IMTLMinLODClampable>(*pDesc, pDevice),
      underlying_texture_(std::move(u_texture)),
      renamable_(std::move(renamable)) {}

  DeviceTexture(
      const tag_texture::DESC1 *pDesc, Rc<Texture> &&u_texture, D3DKMT_HANDLE localHandle, D3DKMT_HANDLE globalHandle,
      MTLD3D11Device *pDevice
  ) :
      TResourceBase<tag_texture, IMTLMinLODClampable>(*pDesc, pDevice),
      underlying_texture_(std::move(u_texture)), local_kmt_(localHandle), global_kmt_(globalHandle) {}

  /* ml675: the bias lives on the resource, not in a process-global pointer map.
   * ml670 keyed it on the COM pointer and never erased on destruction, so a
   * reused address inherited a stale clamp. Ownership here makes that
   * impossible by construction. */
  uint32_t MipBias() const { return mip_bias_; }
  void SetMipBias(uint32_t bias) { mip_bias_ = bias; }

  /* Logical subresource -> physical, or ~0u when the level was clamped away.
   * USAGE_DEFAULT legally permits UpdateSubresource/CopySubresourceRegion, and
   * those indices shift when mips are dropped. Book of the Dead measured zero
   * such calls, but "measured zero" is not "cannot happen". */
  uint32_t TranslateSubresource(uint32_t logical, uint32_t logicalMips) const {
    if (!mip_bias_) return logical;
    uint32_t mip = logical % logicalMips, slice = logical / logicalMips;
    if (mip < mip_bias_) return ~0u;                 /* level does not exist physically */
    return (mip - mip_bias_) + slice * (logicalMips - mip_bias_);
  }

  ~DeviceTexture() {
    if (local_kmt_) {
      D3DKMT_DESTROYALLOCATION destroy = {};
      destroy.hDevice = this->m_parent->GetLocalD3DKMT();
      destroy.hResource = local_kmt_;
      D3DKMTDestroyAllocation(&destroy);
    }
  }

  Rc<Buffer> buffer() final { return {}; };
  Rc<Texture> texture() final { return this->underlying_texture_; };
  BufferSlice bufferSlice() final { return {};}
  Rc<StagingResource> staging(UINT) final { return nullptr; }
  Rc<DynamicBuffer> dynamicBuffer(UINT*, UINT*) final { return {}; }
  Rc<DynamicLinearTexture> dynamicLinearTexture(UINT*, UINT*) final { return {}; };
  Rc<DynamicBuffer> dynamicTexture(UINT , UINT *, UINT *) final { return {}; };

  HRESULT STDMETHODCALLTYPE CreateRenderTargetView(const D3D11_RENDER_TARGET_VIEW_DESC1 *pDesc,
                                 ID3D11RenderTargetView1 **ppView) override {
    D3D11_RENDER_TARGET_VIEW_DESC1 finalDesc;
    if (FAILED(ExtractEntireResourceViewDescription(&this->desc, pDesc,
                                                    &finalDesc))) {
      return E_INVALIDARG;
    }
    TextureViewDescriptor descriptor;
    uint32_t arraySize;
    if constexpr (std::is_same_v<typename tag_texture::DESC1, D3D11_TEXTURE3D_DESC1>) {
      arraySize = this->desc.Depth;
    } else {
      arraySize = this->desc.ArraySize;
    }
    MTL_RENDER_PASS_ATTACHMENT_DESC attachment_desc;
    if (FAILED(InitializeAndNormalizeViewDescriptor(
            this->m_parent, this->desc.MipLevels, arraySize, this->underlying_texture_.ptr(), finalDesc,
            attachment_desc, descriptor
        ))) {
      return E_FAIL;
    }
    if (!ppView) {
      return S_FALSE;
    }
    TextureViewKey key = underlying_texture_->createView(descriptor);
    *ppView = ref(new TextureRTV(key, descriptor.format, &finalDesc, this, this->m_parent, attachment_desc));
    return S_OK;
  };

  HRESULT STDMETHODCALLTYPE CreateDepthStencilView(const D3D11_DEPTH_STENCIL_VIEW_DESC *pDesc,
                                 ID3D11DepthStencilView **ppView) override {
    D3D11_DEPTH_STENCIL_VIEW_DESC finalDesc;
    if (FAILED(ExtractEntireResourceViewDescription(&this->desc, pDesc,
                                                    &finalDesc))) {
      return E_INVALIDARG;
    }
    TextureViewDescriptor descriptor;
    uint32_t arraySize;
    if constexpr (std::is_same_v<typename tag_texture::DESC1, D3D11_TEXTURE3D_DESC1>) {
      arraySize = this->desc.Depth;
    } else {
      arraySize = this->desc.ArraySize;
    }
    MTL_RENDER_PASS_ATTACHMENT_DESC attachment_desc;
    if (FAILED(InitializeAndNormalizeViewDescriptor(
            this->m_parent, this->desc.MipLevels, arraySize, this->underlying_texture_.ptr(), finalDesc,
            attachment_desc, descriptor
        ))) {
      return E_FAIL;
    }
    if (!ppView) {
      return S_FALSE;
    }
    TextureViewKey key = underlying_texture_->createView(descriptor);
    *ppView = ref(new TextureDSV(key, descriptor.format, &finalDesc, this, this->m_parent, attachment_desc));
    return S_OK;
  };

  HRESULT
  STDMETHODCALLTYPE
  CreateShaderResourceView(const D3D11_SHADER_RESOURCE_VIEW_DESC1 *pDesc,
                           ID3D11ShaderResourceView1 **ppView) override {
    D3D11_SHADER_RESOURCE_VIEW_DESC1 finalDesc;
    if (FAILED(ExtractEntireResourceViewDescription(&this->desc, pDesc,
                                                    &finalDesc))) {
      ERR("DeviceTexture: Failed to create SRV descriptor");
      return E_INVALIDARG;
    }
    TextureViewDescriptor descriptor;
    uint32_t arraySize;
    if constexpr (std::is_same_v<typename tag_texture::DESC1, D3D11_TEXTURE3D_DESC1>) {
      arraySize = this->desc.Depth;
    } else {
      arraySize = this->desc.ArraySize;
    }
    /* ml675: TWO descriptors. finalDesc is LOGICAL and is what the view reports
     * from GetDesc; physDesc is the translated copy that addresses the smaller
     * Metal texture. Storing the translated one in the COM view would just move
     * ml670's abstraction leak from resource GetDesc to SRV GetDesc.
     *
     * A null caller descriptor has already been expanded by
     * ExtractEntireResourceViewDescription into a full LOGICAL view, so it needs
     * translating too -- it is not a pass-through case. */
    D3D11_SHADER_RESOURCE_VIEW_DESC1 physDesc = finalDesc;
    uint32_t physMips = this->desc.MipLevels;
    if (mip_bias_) {
      physMips = this->desc.MipLevels > mip_bias_ ? this->desc.MipLevels - mip_bias_ : 1;
      uint32_t most = physDesc.Texture2D.MostDetailedMip;
      uint32_t count = physDesc.Texture2D.MipLevels;
      physDesc.Texture2D.MostDetailedMip = most > mip_bias_ ? most - mip_bias_ : 0;
      if (count != (uint32_t)-1) {
        uint32_t drop = most < mip_bias_ ? mip_bias_ - most : 0;
        physDesc.Texture2D.MipLevels = count > drop ? count - drop : 1;
      }
      if (physDesc.Texture2D.MostDetailedMip >= physMips) {
        static uint32_t oob_n;
        if (oob_n++ < 8)
          ERR("[mip-clamp] ml675 SRV MostDetailedMip ", physDesc.Texture2D.MostDetailedMip,
              " >= physical mips ", physMips, " after bias ", mip_bias_, " -- clamping");
        physDesc.Texture2D.MostDetailedMip = physMips - 1;
        physDesc.Texture2D.MipLevels = 1;
      }
    }
    if (FAILED(InitializeAndNormalizeViewDescriptor(
            this->m_parent, physMips, arraySize, this->underlying_texture_.ptr(), physDesc, descriptor
        ))) {
      ERR("DeviceTexture: Failed to create texture SRV");
      return E_FAIL;
    }
    if (!ppView) {
      return S_FALSE;
    }
    TextureViewKey key = underlying_texture_->createView(descriptor);
    *ppView = ref(new TextureSRV(key, &finalDesc, this, this->m_parent));
    return S_OK;
  };

  HRESULT
  STDMETHODCALLTYPE
  CreateUnorderedAccessView(const D3D11_UNORDERED_ACCESS_VIEW_DESC1 *pDesc,
                            ID3D11UnorderedAccessView1 **ppView) override {
    D3D11_UNORDERED_ACCESS_VIEW_DESC1 finalDesc;
    if (FAILED(ExtractEntireResourceViewDescription(&this->desc, pDesc,
                                                    &finalDesc))) {
      return E_INVALIDARG;
    }
    TextureViewDescriptor descriptor;
    uint32_t arraySize;
    if constexpr (std::is_same_v<typename tag_texture::DESC1, D3D11_TEXTURE3D_DESC1>) {
      arraySize = this->desc.Depth;
    } else {
      arraySize = this->desc.ArraySize;
    }
    if (FAILED(InitializeAndNormalizeViewDescriptor(
            this->m_parent, this->desc.MipLevels, arraySize, this->underlying_texture_.ptr(), finalDesc, descriptor
        ))) {
      ERR("DeviceTexture: Failed to create texture UAV");
      return E_FAIL;
    }
    if (!ppView) {
      return S_FALSE;
    }
    TextureViewKey key = underlying_texture_->createView(descriptor);
    *ppView = ref(new TextureUAV(key, &finalDesc, this, this->m_parent));
    return S_OK;
  };

  virtual HRESULT
  GetSharedHandle(HANDLE *pSharedHandle) override {
    if (pSharedHandle == nullptr || (this->desc.MiscFlags & D3D11_RESOURCE_MISC_SHARED_NTHANDLE)) {
      return E_INVALIDARG;
    }

    if (!(this->desc.MiscFlags & (D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX))) {
      *pSharedHandle = NULL;
      return S_OK;
    }

    if (!global_kmt_) {
      return E_INVALIDARG;
    }

    *pSharedHandle = reinterpret_cast<HANDLE>(global_kmt_);
    return S_OK;
  }

  virtual HRESULT
  CreateSharedHandle(const SECURITY_ATTRIBUTES *Attributes, DWORD Access, const WCHAR *pName, HANDLE *pNTHandle)
      override {
    InitReturnPtr(pNTHandle);
    if (!local_kmt_)
      return E_INVALIDARG;

    OBJECT_ATTRIBUTES attr = {};
    attr.Length = sizeof(attr);
    attr.SecurityDescriptor = const_cast<SECURITY_ATTRIBUTES*>(Attributes);

    WCHAR buffer[MAX_PATH];
    UNICODE_STRING name_str;
    if (pName) {
      DWORD session, len, name_len = wcslen(pName);

      ProcessIdToSessionId(GetCurrentProcessId(), &session);
      len = swprintf(buffer, ARRAYSIZE(buffer), L"\\Sessions\\%u\\BaseNamedObjects\\", session);
      memcpy(buffer + len, pName, (name_len + 1) * sizeof(WCHAR));
      name_str.MaximumLength = name_str.Length = (len + name_len) * sizeof(WCHAR);
      name_str.MaximumLength += sizeof(WCHAR);
      name_str.Buffer = buffer;

      attr.ObjectName = &name_str;
      attr.Attributes = OBJ_CASE_INSENSITIVE;
    }

    if (D3DKMTShareObjects(1, &local_kmt_, &attr, Access, pNTHandle)) {
      ERR("DeviceTexture: Failed to create shared handle");
      return E_FAIL;
    }

    return S_OK;
  }

  void SetMinLOD(float MinLod) override { min_lod = MinLod; }

  float GetMinLOD() override { return min_lod; }
};

struct SharedResourceData {
  char mach_port_name[54];
  D3D11_RESOURCE_DIMENSION dimension;
  union {
    D3D11_TEXTURE1D_DESC desc1d;
    D3D11_TEXTURE2D_DESC1 desc2d;
    D3D11_TEXTURE3D_DESC1 desc3d;
  } desc;
};


/* ml675: BC formats DXGI-wise. Kept local -- the census copy in d3d11_device.cpp
 * lives in an anonymous namespace and is not linkable from here. */
static inline bool IsBCFormatForClamp(uint32_t f) {
  return (f >= 70 && f <= 84) ||    /* BC1..BC5 incl. typeless/sRGB */
         (f >= 94 && f <= 99);      /* BC6H, BC7 */
}

/* ml2101: 4-byte, 8-bit-channel formats the clamp and the downscale also
 * take: R8G8B8A8 (27-29) and B8G8R8A8 / B8G8R8X8 (87, 88, 90-93). Uncompressed
 * sprite art is commonly stored this way, at 4 bytes a texel. */
static inline bool IsRGBA8ForClamp(uint32_t f) {
  return (f >= 27 && f <= 29) || f == 87 || f == 88 || (f >= 90 && f <= 93);
}

/* ---- ml2101: texture census, every format --------------------------------
 *
 * The BC census (d3d11_device.cpp) cannot see uncompressed textures, so a
 * title whose memory goes to RGBA sprites or render targets showed a Metal
 * total nobody could account for. This counts every Texture2D by DXGI format,
 * with its logical and physical bytes, how much of it the clamp and the
 * downscale took, and how much was ineligible, and logs the top formats by
 * physical bytes every 512 textures. Byte counts are estimates from the
 * descriptor (texel/block size x extents x mips x slices). */
static inline uint64_t
EstimateTextureBytes(uint32_t f, uint32_t w, uint32_t h, uint32_t mips, uint32_t slices) {
  uint32_t bpb = 4, block = 1;   /* bytes per block, block edge in texels */
  if ((f >= 70 && f <= 72) || (f >= 79 && f <= 81)) { bpb = 8; block = 4; }       /* BC1, BC4 */
  else if (IsBCFormatForClamp(f)) { bpb = 16; block = 4; }                          /* BC2/3/5/6H/7 */
  else if (f >= 1 && f <= 4) bpb = 16;                                              /* R32G32B32A32 */
  else if (f >= 5 && f <= 8) bpb = 12;                                              /* R32G32B32 */
  else if (f >= 9 && f <= 22) bpb = 8;                                              /* 64-bit */
  else if (f >= 48 && f <= 59) bpb = 2;                                             /* R8G8, R16 */
  else if ((f >= 60 && f <= 65) || f == 85 || f == 86 || f == 115) bpb = 1;         /* R8, A8 */
  uint64_t total = 0;
  for (uint32_t m = 0; m < std::max(1u, mips); m++) {
    uint64_t bw = (std::max(1u, w >> m) + block - 1) / block, bh = (std::max(1u, h >> m) + block - 1) / block;
    total += bw * bh * bpb;
  }
  return total * std::max(1u, slices);
}

struct TexCensusEntry {
  std::atomic<uint64_t> n{0}, logical{0}, physical{0}, clamped{0}, downscaled{0};
  /* 1 MB+ textures neither saving applied to, by why: rendered to / written by
   * the GPU (never shrinkable), single-level without initial data, other. */
  std::atomic<uint64_t> big_rt{0}, big_nodata{0}, big_other{0};
};
static TexCensusEntry g_tex_census[192];
static std::atomic<uint64_t> g_tex_census_n{0};
/* ml2113: the 1 MB+ "other" bytes by the reason the clamp and the downscale
 * passed them over (bitmask, see TexSkipReason). ml2101 showed 544 MB of
 * R8G8B8A8_SRGB in Stick It to the Stickman under "other" with no way to tell
 * which eligibility rule excluded it. */
static std::atomic<uint64_t> g_tex_skip_bytes[256];
static std::atomic<uint64_t> g_tex_skip_n[256];

static void
TexCensusRecord(uint32_t f, uint64_t logical, uint64_t physical, uint32_t mip_bias, bool downscaled,
                bool srv_only, bool single_level, bool has_data, uint32_t skip_reason = 0) {
  if (f >= 192) f = 0;
  auto &e = g_tex_census[f];
  e.n.fetch_add(1, std::memory_order_relaxed);
  e.logical.fetch_add(logical, std::memory_order_relaxed);
  e.physical.fetch_add(physical, std::memory_order_relaxed);
  if (mip_bias) e.clamped.fetch_add(logical - physical, std::memory_order_relaxed);
  if (downscaled) e.downscaled.fetch_add(logical - physical, std::memory_order_relaxed);
  /* "Big and untouched": 1 MB or more that neither saving applied to. */
  if (!mip_bias && !downscaled && logical >= (1u << 20)) {
    if (!srv_only) e.big_rt.fetch_add(logical, std::memory_order_relaxed);
    else if (single_level && !has_data) e.big_nodata.fetch_add(logical, std::memory_order_relaxed);
    else {
      e.big_other.fetch_add(logical, std::memory_order_relaxed);
      g_tex_skip_bytes[skip_reason & 255].fetch_add(logical, std::memory_order_relaxed);
      g_tex_skip_n[skip_reason & 255].fetch_add(1, std::memory_order_relaxed);
    }
  }
  const uint64_t n = g_tex_census_n.fetch_add(1, std::memory_order_relaxed) + 1;
  if (n % 512)
    return;
  uint32_t order[192];
  for (uint32_t i = 0; i < 192; i++) order[i] = i;
  std::sort(order, order + 192, [](uint32_t a, uint32_t b) {
    return g_tex_census[a].physical.load(std::memory_order_relaxed) > g_tex_census[b].physical.load(std::memory_order_relaxed);
  });
  uint64_t all = 0;
  for (auto &c : g_tex_census) all += c.physical.load(std::memory_order_relaxed);
  ERR("[tex-census] ml2101 after ", n, " textures: physical ", all >> 20, " MB in total; top formats:");
  for (uint32_t i = 0; i < 8; i++) {
    auto &c = g_tex_census[order[i]];
    if (!c.physical.load(std::memory_order_relaxed)) break;
    ERR("[tex-census]   fmt=", order[i], " n=", c.n.load(std::memory_order_relaxed),
        " logical=", c.logical.load(std::memory_order_relaxed) >> 20, "MB physical=",
        c.physical.load(std::memory_order_relaxed) >> 20, "MB clamp-saved=",
        c.clamped.load(std::memory_order_relaxed) >> 20, "MB downscale-saved=",
        c.downscaled.load(std::memory_order_relaxed) >> 20, "MB | big untouched: rendered-to=",
        c.big_rt.load(std::memory_order_relaxed) >> 20, "MB no-initial-data=",
        c.big_nodata.load(std::memory_order_relaxed) >> 20, "MB other=",
        c.big_other.load(std::memory_order_relaxed) >> 20, "MB");
  }
  /* ml2113: why "other" was skipped. Bits: 1 single-level, 2 usage not
   * DEFAULT/IMMUTABLE, 4 CPU access, 8 misc flags beyond cube/resource-clamp,
   * 16 array, 32 odd width or height, 64 no initial data, 128 cube or under
   * 256 px (downscale size floor). */
  uint32_t ro[256];
  for (uint32_t i = 0; i < 256; i++) ro[i] = i;
  std::sort(ro, ro + 256, [](uint32_t a, uint32_t b) {
    return g_tex_skip_bytes[a].load(std::memory_order_relaxed) > g_tex_skip_bytes[b].load(std::memory_order_relaxed);
  });
  for (uint32_t i = 0; i < 5; i++) {
    const uint64_t bytes = g_tex_skip_bytes[ro[i]].load(std::memory_order_relaxed);
    if (!bytes) break;
    ERR("[tex-census]   other skipped by reason=0x", std::hex, ro[i], std::dec, ": ", bytes >> 20, "MB in ",
        g_tex_skip_n[ro[i]].load(std::memory_order_relaxed), " textures (ml2113)");
  }
}

/* ---- ml2000: memory-pressure-aware automatic clamp (d3d11.mipClampAuto) ----
 *
 * A D3D11 scene load created ~500 SRV-only BC textures (mostly 2048/4096
 * square, 1.23 GB logical) and ran the process from 4.35 GB to the 6.1 GB
 * jetsam limit in ~30 s. mipClampBC would have saved it, but it is a blanket
 * setting that costs resolution even with memory to spare. This applies the
 * same, already-proven clamp -- bias 1, large textures only -- to textures
 * created while the process is within d3d11.mipClampAutoMB (default 1536) of
 * its limit, as reported by os_proc_available_memory() through MadeiraCtl op 7.
 *
 * A Madeira switch (util_madeira_switch.hpp): on by default only in the
 * i386 build; every 64-bit build leaves textures alone unless
 * d3d11.mipClampAuto (dxmt.conf / madeira.cfg) or MADEIRA_MIP_CLAMP_AUTO is
 * set to an "on" value. Either one set to 0/false/off/no disables it. An
 * explicit d3d11.mipClampBC=N>0 takes precedence.
 * Where the query is unavailable (remote Metal, an older unix side) the
 * reading is marked unavailable once and nothing is ever auto-clamped. */
struct MipClampAutoConfig {
  bool enabled;
  uint32_t threshold_mb;
};

static const MipClampAutoConfig &
GetMipClampAutoConfig() {
  static const MipClampAutoConfig cfg = [] {
    MipClampAutoConfig c;
    const std::string opt = Config::getInstance().getOption<std::string>("d3d11.mipClampAuto", "");
    const std::string env_v = env::getEnvVar("MADEIRA_MIP_CLAMP_AUTO");
    if (mip_clamp::IsOffValue(opt.c_str()) || mip_clamp::IsOffValue(env_v.c_str()))
      c.enabled = false;
    else if (!opt.empty() || !env_v.empty())
      c.enabled = true;
    else
      c.enabled = true;   /* ml2120: 64-bit too (was 32-bit only, see below) */
    /* ml2120: Stick It to the Stickman (64-bit) loads 3.8 GB of large BC and
     * RGBA8 art, nearly all of it eligible, and was killed at the iPhone's 6 GB
     * limit with nothing clamped because the auto clamp only ran in the i386
     * module. A 64-bit process also has more besides textures (FEX, Wine, the
     * game's own heap), so it starts clamping a little earlier. */
    c.threshold_mb = (uint32_t)std::max(0, Config::getInstance().getOption<int>("d3d11.mipClampAutoMB",
                                                                            kMadeira32BitModule ? 1536 : 2048));
    if (c.enabled)
      ERR("[mip-clamp] ml2000 auto on threshold=", c.threshold_mb,
          " MB (d3d11.mipClampAuto=0 or MADEIRA_MIP_CLAMP_AUTO=0 disables; d3d11.mipClampAutoMB sets the threshold)");
    return c;
  }();
  return cfg;
}

/* MadeiraCtl op 7: headroom and footprint in MB, or kHeadroomUnavailable. */
static int64_t
QueryHeadroomMB(int64_t *footprint_mb) {
  struct madeira_ctl_args a;
  memset(&a, 0, sizeof a);
  a.op = 7;
  MadeiraCtl(&a);
  if (a.ret != 1)
    return mip_clamp::kHeadroomUnavailable;
  *footprint_mb = (int64_t)(a.ptr >> 20);
  return (int64_t)(a.len >> 20);
}

static std::atomic<uint64_t> g_auto_candidates{0};
static std::atomic<uint64_t> g_auto_clamped{0};
static std::atomic<int64_t> g_auto_headroom_mb{mip_clamp::kHeadroomUnknown};
static std::atomic<int64_t> g_auto_footprint_mb{0};
static std::atomic<bool> g_auto_pressure_logged{false};

/* Is the process within the auto threshold of its memory limit? Counts one
 * auto candidate; re-reads the headroom on the ShouldRequery cadence. */
static bool
MipClampAutoUnderPressure(const MipClampAutoConfig &cfg) {
  const uint64_t n = g_auto_candidates.fetch_add(1, std::memory_order_relaxed) + 1;
  int64_t headroom = g_auto_headroom_mb.load(std::memory_order_relaxed);
  if (mip_clamp::ShouldRequery(n, headroom, cfg.threshold_mb)) {
    int64_t foot = 0;
    const int64_t fresh = QueryHeadroomMB(&foot);
    if (fresh == mip_clamp::kHeadroomUnavailable && headroom == mip_clamp::kHeadroomUnknown)
      ERR("[mip-clamp] ml2000 auto: memory headroom unavailable here; auto clamp inactive");
    headroom = fresh;
    g_auto_headroom_mb.store(headroom, std::memory_order_relaxed);
    g_auto_footprint_mb.store(foot, std::memory_order_relaxed);
  }

  const bool pressure = mip_clamp::UnderPressure(headroom, cfg.threshold_mb);
  /* Log each entry into and exit from the pressure region once. */
  bool was = g_auto_pressure_logged.load(std::memory_order_relaxed);
  if (pressure != was && g_auto_pressure_logged.compare_exchange_strong(was, pressure))
    ERR("[mip-clamp] ml2000 auto headroom=", headroom, " MB footprint=",
        g_auto_footprint_mb.load(std::memory_order_relaxed), " MB -> ", pressure ? "clamping" : "not clamping",
        " large BC textures (threshold ", cfg.threshold_mb, " MB) clamped=",
        g_auto_clamped.load(std::memory_order_relaxed));
  return pressure;
}

/* Bias for an auto candidate already known to pass the mipClampBC filter. */
static uint32_t
MipClampAutoBias(uint32_t width, uint32_t height, uint32_t mip_levels) {
  const MipClampAutoConfig &cfg = GetMipClampAutoConfig();
  if (!cfg.enabled || !cfg.threshold_mb || !mip_clamp::AutoSizeEligible(width, height))
    return 0;
  const uint32_t bias = mip_clamp::ClampBias(1, width, height, mip_levels, true);
  if (!bias)
    return 0;
  if (!MipClampAutoUnderPressure(cfg))
    return 0;
  const int64_t headroom = g_auto_headroom_mb.load(std::memory_order_relaxed);

  const uint64_t clamped = g_auto_clamped.fetch_add(1, std::memory_order_relaxed) + 1;
  if (mip_clamp::ShouldLogClamp(clamped))
    ERR("[mip-clamp] ml2000 auto headroom=", headroom, " MB clamped=", clamped, " (", width, "x", height,
        " mips ", mip_levels, " -> ", width >> bias, "x", height >> bias, " mips ", mip_levels - bias, ")");
  return bias;
}

template <typename tag>
HRESULT CreateDeviceTextureInternal(MTLD3D11Device *pDevice,
                                    const typename tag::DESC1 *pDesc,
                                    const D3D11_SUBRESOURCE_DATA *pInitialData,
                                    typename tag::COM_IMPL **ppTexture) {
  WMTTextureInfo info;
  typename tag::DESC1 finalDesc;
  if (FAILED(CreateMTLTextureDescriptor(pDevice, pDesc, &finalDesc, &info))) {
    return E_INVALIDARG;
  }

  /* ---- ml675: PHYSICAL mip clamp, decided here rather than in the device ----
   *
   * A15 cannot sample BC, so BC formats are decoded to RGBA8 at upload and cost
   * 4-8x their compressed size. Dropping top mips is the cheapest way back under
   * the jetsam limit.
   *
   * This function already had the right abstraction: finalDesc is the LOGICAL
   * normalised descriptor the resource reports, and `info` describes the Metal
   * texture. ml670 clamped one layer up in CreateTexture2D1 and passed the
   * shrunken descriptor onward, which made GetDesc lie to the application.
   * Eligibility and the clamp amount are computed from the normalised LOGICAL
   * descriptor; only `info` and the upload loop see reduced numbers. */
  uint32_t mip_bias = 0;
  typename tag::DESC1 physDesc = finalDesc;
  bool downscaled = false;                  /* ml2100 */
  std::vector<uint8_t> downscaled_pixels;   /* ml2100: outlives initialize() below */
  D3D11_SUBRESOURCE_DATA downscaled_init = {};
  if constexpr (std::is_same_v<typename tag::DESC1, D3D11_TEXTURE2D_DESC1>) {
    static int cached_clamp = -1;
    if (cached_clamp < 0)
      cached_clamp = std::max(0, std::min(4, Config::getInstance().getOption<int>("d3d11.mipClampBC", 0)));
    /* ml745: pInitialData is NO LONGER required. Requiring it meant the clamp
     * could never fire for a title that creates BC textures empty and streams
     * them later -- the case that actually matters here: 13 textures arrived
     * with initial data against 2,025 streamed updates. Those updates carry
     * LOGICAL mip indices, which UpdateTexture now translates through the bias,
     * dropping levels that no longer exist physically. */
    const bool eligible =
        (IsBCFormatForClamp((uint32_t)finalDesc.Format) || IsRGBA8ForClamp((uint32_t)finalDesc.Format)) &&
        (finalDesc.Usage == D3D11_USAGE_DEFAULT || finalDesc.Usage == D3D11_USAGE_IMMUTABLE) &&
        finalDesc.BindFlags == D3D11_BIND_SHADER_RESOURCE && !finalDesc.CPUAccessFlags &&
        finalDesc.MipLevels >= 2 && finalDesc.SampleDesc.Count <= 1 &&
        finalDesc.Width >= 8 && finalDesc.Height >= 8;
    /* ml2000: the automatic clamp additionally refuses any misc flag except
     * TEXTURECUBE / RESOURCE_CLAMP. A shared texture is re-created at LOGICAL
     * size by whoever opens it (ImportSharedTextureInternal) and would then
     * address a smaller allocation; tiled and other exotic resources have no
     * business losing levels behind the application's back. */
    const bool auto_misc_ok =
        !(finalDesc.MiscFlags & ~(UINT)(D3D11_RESOURCE_MISC_TEXTURECUBE | D3D11_RESOURCE_MISC_RESOURCE_CLAMP));
    if (eligible && cached_clamp) {
      mip_bias = mip_clamp::ClampBias((uint32_t)cached_clamp, finalDesc.Width, finalDesc.Height,
                                      finalDesc.MipLevels, false);
      if (mip_bias) {
        static uint32_t clamp_n;
        if (++clamp_n <= 8 || (clamp_n % 256) == 0)
          ERR("[mip-clamp] ml675 #", clamp_n, " logical ", finalDesc.Width, "x", finalDesc.Height,
              " mips ", finalDesc.MipLevels, " -> physical ", finalDesc.Width >> mip_bias, "x",
              finalDesc.Height >> mip_bias, " mips ", finalDesc.MipLevels - mip_bias, " arr=", finalDesc.ArraySize);
      }
    } else if (eligible && auto_misc_ok) {
      /* ml2000: MipLevels >= 2 is part of `eligible`, so a single-level texture
       * is never clamped; arrays and cubes need nothing extra (see the
       * per-slice initial-data remap below and TranslateSubresource). */
      mip_bias = MipClampAutoBias(finalDesc.Width, finalDesc.Height, finalDesc.MipLevels);
    }
    if (mip_bias) {
      physDesc.Width     = std::max(1u, finalDesc.Width  >> mip_bias);
      physDesc.Height    = std::max(1u, finalDesc.Height >> mip_bias);
      physDesc.MipLevels = finalDesc.MipLevels - mip_bias;
      info.width              = physDesc.Width;
      info.height             = physDesc.Height;
      info.mipmap_level_count = physDesc.MipLevels;
    }

    /* ---- ml2100: halve single-level BC1/BC3 textures ----------------------
     *
     * A texture with one level has nothing for the clamp to drop, and those are
     * often the largest ones (full-screen 2D art). Halve them instead, in the
     * compressed domain (dxmt_bcn_downscale.hpp): a quarter of the memory, and
     * normalised sampling is unchanged. Only with initial data -- that is the
     * only content we can halve -- and only for textures the application can
     * write solely through the runtime ops that the context drops for these.
     *
     * Applies when d3d11.downscaleBC=1; with it unset, whenever the clamp would
     * (an explicit mipClampBC, or the auto clamp under memory pressure);
     * d3d11.downscaleBC=0 turns it off. */
    /* ml2101: 8 = uncompressed 8-bit RGBA/BGRA (IsRGBA8ForClamp), plain box filter. */
    const int bc_kind = (finalDesc.Format >= 70 && finalDesc.Format <= 72) ? 1
                      : (finalDesc.Format >= 76 && finalDesc.Format <= 78) ? 3
                      : IsRGBA8ForClamp((uint32_t)finalDesc.Format) ? 8
                      /* ml2113: 1-byte UNORM texels, R8_UNORM (61) and A8_UNORM (65) */
                      : (finalDesc.Format == 61 || finalDesc.Format == 65) ? 9 : 0;
    const bool texel_kind = bc_kind == 8 || bc_kind == 9;   /* uncompressed: any size (ml2113) */
    if (!mip_bias && bc_kind && pInitialData && pInitialData[0].pSysMem && finalDesc.MipLevels == 1 &&
        finalDesc.ArraySize == 1 && auto_misc_ok && !(finalDesc.MiscFlags & D3D11_RESOURCE_MISC_TEXTURECUBE) &&
        (finalDesc.Usage == D3D11_USAGE_DEFAULT || finalDesc.Usage == D3D11_USAGE_IMMUTABLE) &&
        finalDesc.BindFlags == D3D11_BIND_SHADER_RESOURCE && !finalDesc.CPUAccessFlags &&
        finalDesc.SampleDesc.Count <= 1 && (texel_kind || (!(finalDesc.Width & 7) && !(finalDesc.Height & 7))) &&
        /* ml2102: an explicit clamp is a request to save memory, so it also takes
         * mid-sized textures (256+); the auto clamp keeps its 1024 cutoff. A 2D
         * title kept 650 MB of 512-1000 px sprites out of reach of the 1024 one. */
        (cached_clamp > 0 ? std::max(finalDesc.Width, finalDesc.Height) >= 256
                          : mip_clamp::AutoSizeEligible(finalDesc.Width, finalDesc.Height))) {
      static int downscale_opt = -2;
      if (downscale_opt == -2)
        downscale_opt = Config::getInstance().getOption<int>("d3d11.downscaleBC", -1);
      const MipClampAutoConfig &cfg = GetMipClampAutoConfig();
      const bool want = downscale_opt > 0 ||
                        (downscale_opt < 0 && (cached_clamp > 0 ||
                                               (cfg.enabled && cfg.threshold_mb && MipClampAutoUnderPressure(cfg))));
      if (want) {
        const bool rgba = texel_kind;
        const uint32_t bpp = bc_kind == 9 ? 1u : 4u;
        const uint32_t blk = bc_kind == 1 ? 8u : 16u;
        /* Pitch of the half-size image: BC rows are rows of 4x4 blocks. */
        /* ml2113: uncompressed texels halve to ceil(w/2) x ceil(h/2), so odd sizes qualify. */
        const uint32_t half_w = rgba ? (finalDesc.Width + 1) / 2 : finalDesc.Width / 2;
        const uint32_t half_h = rgba ? (finalDesc.Height + 1) / 2 : finalDesc.Height / 2;
        const size_t dst_pitch = rgba ? (size_t)half_w * bpp : (size_t)(finalDesc.Width / 8) * blk;
        downscaled_pixels.resize(dst_pitch * (rgba ? half_h : finalDesc.Height / 8));
        const auto *src = (const uint8_t *)pInitialData[0].pSysMem;
        const bool ok = rgba ? texel8_downscale_half(bpp, src, pInitialData[0].SysMemPitch, finalDesc.Width,
                                                     finalDesc.Height, downscaled_pixels.data(), dst_pitch)
                             : bcn_downscale_half(bc_kind, src, pInitialData[0].SysMemPitch, finalDesc.Width,
                                                  finalDesc.Height, downscaled_pixels.data(), dst_pitch);
        if (ok) {
          downscaled = true;
          downscaled_init.pSysMem = downscaled_pixels.data();
          downscaled_init.SysMemPitch = (UINT)dst_pitch;
          downscaled_init.SysMemSlicePitch = (UINT)downscaled_pixels.size();
          pInitialData = &downscaled_init;
          physDesc.Width  = half_w;
          physDesc.Height = half_h;
          info.width  = physDesc.Width;
          info.height = physDesc.Height;
          static std::atomic<uint64_t> downscale_n{0};
          static std::atomic<uint64_t> downscale_saved_kb{0};
          const uint64_t n = downscale_n.fetch_add(1, std::memory_order_relaxed) + 1;
          const uint64_t saved =
              downscale_saved_kb.fetch_add(downscaled_pixels.size() * 3 / 1024, std::memory_order_relaxed) +
              downscaled_pixels.size() * 3 / 1024;
          if (mip_clamp::ShouldLogClamp(n))
            ERR("[bc-downscale] ml2100 #", n, rgba ? (bpp == 1 ? " R8 " : " RGBA8 ") : (bc_kind == 1 ? " BC1 " : " BC3 "), finalDesc.Width, "x", finalDesc.Height,
                " -> ", physDesc.Width, "x", physDesc.Height, " saved ", saved / 1024, " MB in total");
        }
      }
    }
  }
  bool single_subresource = info.mipmap_level_count == 1 && info.array_length == 1 &&
                            !(physDesc.MiscFlags & D3D11_RESOURCE_MISC_TEXTURECUBE);
  auto texture = Rc<Texture>(new Texture(info, pDevice->GetMTLDevice()));
  /* ml745: the immediate context resolves streamed uploads through the dxmt
   * Texture, not the D3D wrapper, so the bias must be visible there too. */
  texture->setMipBias(mip_bias);
  texture->setDownscaled(downscaled);
  if constexpr (std::is_same_v<typename tag::DESC1, D3D11_TEXTURE2D_DESC1>) {
    TexCensusRecord((uint32_t)finalDesc.Format,
                    EstimateTextureBytes((uint32_t)finalDesc.Format, finalDesc.Width, finalDesc.Height,
                                         finalDesc.MipLevels, finalDesc.ArraySize),
                    EstimateTextureBytes((uint32_t)finalDesc.Format, physDesc.Width, physDesc.Height,
                                         physDesc.MipLevels, physDesc.ArraySize),
                    mip_bias, downscaled, finalDesc.BindFlags == D3D11_BIND_SHADER_RESOURCE,
                    finalDesc.MipLevels == 1, pInitialData != nullptr,
                    (finalDesc.MipLevels == 1 ? 1u : 0u) |
                    ((finalDesc.Usage != D3D11_USAGE_DEFAULT && finalDesc.Usage != D3D11_USAGE_IMMUTABLE) ? 2u : 0u) |
                    (finalDesc.CPUAccessFlags ? 4u : 0u) |
                    ((finalDesc.MiscFlags & ~(UINT)(D3D11_RESOURCE_MISC_TEXTURECUBE | D3D11_RESOURCE_MISC_RESOURCE_CLAMP)) ? 8u : 0u) |
                    (finalDesc.ArraySize > 1 ? 16u : 0u) |
                    (((finalDesc.Width | finalDesc.Height) & 1) ? 32u : 0u) |
                    (pInitialData == nullptr ? 64u : 0u) |
                    (((finalDesc.MiscFlags & D3D11_RESOURCE_MISC_TEXTURECUBE) ||
                      std::max(finalDesc.Width, finalDesc.Height) < 256) ? 128u : 0u));
  }

  auto &initializer = pDevice->GetDXMTDevice().queue().initializer;

  auto initialize = [&](Rc<TextureAllocation> &&allocation) {
    texture->rename(std::move(allocation));
    if (!pInitialData) {
      for (auto sub : EnumerateSubresources(physDesc)) {
        initializer.initWithZero(texture.ptr(), texture->current(), sub.ArraySlice, sub.MipLevel);
      }
    } else {
      for (auto sub : EnumerateSubresources(physDesc)) {
        /* ml675: subresource index is mip + slice * MipLevels, so a clamp shifts
         * EVERY index. Walk the PHYSICAL subresources but read the caller's
         * LOGICAL array -- correct for arrays and cubes alike, since D3D counts
         * cube faces in ArraySize. A flat pointer bump would corrupt every slice
         * after the first. */
        auto &data = pInitialData[(sub.MipLevel + mip_bias) +
                                  (size_t)sub.ArraySlice * finalDesc.MipLevels];
        initializer.initWithData(
            texture.ptr(), texture->current(), sub.ArraySlice, sub.MipLevel, data.pSysMem, data.SysMemPitch,
            data.SysMemSlicePitch
        );
      }
    }
  };

  auto shared_flag =
      D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;
  if (finalDesc.MiscFlags & shared_flag) {
    if (!(pDevice->GetLocalD3DKMT() & 0xc0000000)) {
      ERR("DeviceTexture: Invalid device handle");
      return E_FAIL;
    }
    // use a dedicated path for now, because there are other works for private storage mode

    Flags<TextureAllocationFlag> flags;
    flags.set(TextureAllocationFlag::GpuPrivate);
    if (finalDesc.Usage == D3D11_USAGE_IMMUTABLE)
      flags.set(TextureAllocationFlag::GpuReadonly);
    flags.set(TextureAllocationFlag::Shared);
    auto allocation = texture->allocate(flags);

    mach_port_t mach_port = allocation->machPort;
    if (!mach_port) {
      /* ml866: no shareable port means sharing is unavailable here (the
       * remote backend cannot pass IOSurfaces between machines). The texture
       * itself exists, so keep it and carry on unshared: an engine asking for
       * a shared flag it will never exercise loses nothing, while E_FAIL was
       * a fatal in the caller. */
      static bool said = false;
      if (!said) { said = true; ERR("DeviceTexture: no mach port for a shared texture; continuing unshared"); }
      initialize(std::move(allocation));
      auto *tex = ref(new DeviceTexture<tag>(&finalDesc, std::move(texture), pDevice));
      tex->SetMipBias(mip_bias);
      *ppTexture = reinterpret_cast<typename tag::COM_IMPL *>(tex);
      return S_OK;
    }
    SharedResourceData runtimeData;
    MakeUniqueSharedName(runtimeData.mach_port_name);
    if (!WMTBootstrapRegister(runtimeData.mach_port_name, mach_port)) {
      ERR("DeviceTexture: Failed to register mach port for shared texture");
      return E_FAIL;
    }
    runtimeData.dimension = tag::dimension;
    memcpy(&runtimeData.desc, pDesc, sizeof(typename tag::DESC1));

    D3DKMT_CREATEALLOCATION create = {};
    create.hDevice = pDevice->GetLocalD3DKMT();
    create.pPrivateRuntimeData = &runtimeData;
    create.PrivateRuntimeDataSize = sizeof(runtimeData);
    create.Flags.StandardAllocation = 1;
    create.NumAllocations = 1;
    D3DDDI_ALLOCATIONINFO2 allocationInfo = {};
    create.pAllocationInfo2 = &allocationInfo;
    D3DKMT_CREATESTANDARDALLOCATION standardAllocation = {};
    create.pStandardAllocation = &standardAllocation;
    standardAllocation.Type = D3DKMT_STANDARDALLOCATIONTYPE_EXISTINGHEAP;
    create.Flags.ExistingSysMem = 1;
    D3DDDI_ALLOCATIONINFO systemMem;
    allocationInfo.pSystemMem = &systemMem;
    create.Flags.CreateResource = 1;
    create.Flags.CreateShared = 1;
    create.Flags.NtSecuritySharing = !!(finalDesc.MiscFlags & D3D11_RESOURCE_MISC_SHARED_NTHANDLE);
    if (D3DKMTCreateAllocation2(&create)) {
      ERR("DeviceTexture: Failed to create D3DKMT for shared texture");
      return E_FAIL;
    }

    // TODO: handle keyed mutex

    initialize(std::move(allocation));
    {
      auto *tex = ref(new DeviceTexture<tag>(&finalDesc, std::move(texture), create.hResource,
                                             create.hGlobalShare, pDevice));
      tex->SetMipBias(mip_bias);
      *ppTexture = reinterpret_cast<typename tag::COM_IMPL *>(tex);
    }
    return S_OK;
  }

  Flags<TextureAllocationFlag> flags;
  flags.set(finalDesc.CPUAccessFlags ? TextureAllocationFlag::GpuManaged : TextureAllocationFlag::GpuPrivate);
  if (finalDesc.Usage == D3D11_USAGE_IMMUTABLE)
    flags.set(TextureAllocationFlag::GpuReadonly);
  if (single_subresource && (finalDesc.BindFlags & D3D11_BIND_DEPTH_STENCIL)) {
    Rc<RenamableTexturePool> renamable = new RenamableTexturePool(texture.ptr(), 32, flags);
    initialize(renamable->getNext(0));
    auto *tex = ref(new DeviceTexture<tag>(&finalDesc, std::move(texture), std::move(renamable), pDevice));
    tex->SetMipBias(mip_bias);
    *ppTexture = reinterpret_cast<typename tag::COM_IMPL *>(tex);
  } else {
    initialize(texture->allocate(flags));
    auto *tex = ref(new DeviceTexture<tag>(&finalDesc, std::move(texture), pDevice));
    tex->SetMipBias(mip_bias);
    *ppTexture = reinterpret_cast<typename tag::COM_IMPL *>(tex);
  }
  return S_OK;
}

HRESULT
CreateDeviceTexture1D(MTLD3D11Device *pDevice,
                      const D3D11_TEXTURE1D_DESC *pDesc,
                      const D3D11_SUBRESOURCE_DATA *pInitialData,
                      ID3D11Texture1D **ppTexture) {
  return CreateDeviceTextureInternal<tag_texture_1d>(pDevice, pDesc,
                                                     pInitialData, ppTexture);
}

HRESULT
CreateDeviceTexture2D(MTLD3D11Device *pDevice,
                      const D3D11_TEXTURE2D_DESC1 *pDesc,
                      const D3D11_SUBRESOURCE_DATA *pInitialData,
                      ID3D11Texture2D1 **ppTexture) {
  return CreateDeviceTextureInternal<tag_texture_2d>(pDevice, pDesc,
                                                     pInitialData, ppTexture);
}

HRESULT
CreateDeviceTexture3D(MTLD3D11Device *pDevice,
                      const D3D11_TEXTURE3D_DESC1 *pDesc,
                      const D3D11_SUBRESOURCE_DATA *pInitialData,
                      ID3D11Texture3D1 **ppTexture) {
  return CreateDeviceTextureInternal<tag_texture_3d>(pDevice, pDesc,
                                                     pInitialData, ppTexture);
}

template <typename tag>
HRESULT
ImportSharedTextureInternal(
    MTLD3D11Device *pDevice, const typename tag::DESC1 *pDescUnchecked, mach_port_t MachPort, REFIID riid,
    void **ppTexture
) {
  WMTTextureInfo info;
  typename tag::DESC1 finalDesc;
  if (FAILED(CreateMTLTextureDescriptor(pDevice, pDescUnchecked, &finalDesc, &info)))
    return E_INVALIDARG;

  auto texture = Rc<Texture>(new Texture(info, pDevice->GetMTLDevice()));
  auto allocation = texture->import(MachPort);
  if (!allocation)
    return E_FAIL;
  texture->rename(std::move(allocation));

  Com<DeviceTexture<tag>> device_texture = (ref(new DeviceTexture<tag>(&finalDesc, std::move(texture), pDevice)));
  return device_texture->QueryInterface(riid, ppTexture);
}

HRESULT
ImportSharedTexture(MTLD3D11Device *pDevice, HANDLE hResource, REFIID riid, void **ppTexture) {
  InitReturnPtr(ppTexture);

  if (!(reinterpret_cast<uintptr_t>(hResource) & 0xc0000000)) {
    WARN("ImportSharedTexture: Invalid shared handle type");
    return E_INVALIDARG;
  }

  if (ppTexture == nullptr)
    return S_FALSE;

  struct SharedResourceData runtimeData;

  D3DKMT_QUERYRESOURCEINFO query = {};
  query.hDevice = pDevice->GetLocalD3DKMT();
  query.hGlobalShare = reinterpret_cast<uintptr_t>(hResource);
  query.pPrivateRuntimeData = &runtimeData;
  query.PrivateRuntimeDataSize = sizeof(runtimeData);

  if (D3DKMTQueryResourceInfo(&query)) {
    WARN("ImportSharedTexture: Failed to query resource: ", hResource);
    return E_INVALIDARG;
  }

  if (query.PrivateRuntimeDataSize != sizeof(runtimeData)) {
    WARN("ImportSharedTexture: Unexpected size: ", query.PrivateRuntimeDataSize);
    return E_INVALIDARG;
  } 

  D3DDDI_OPENALLOCATIONINFO2 alloc = {};
  D3DKMT_OPENRESOURCE open = {};
  open.hDevice = pDevice->GetLocalD3DKMT();
  open.hGlobalShare = reinterpret_cast<uintptr_t>(hResource);
  open.NumAllocations = 1;
  open.pOpenAllocationInfo2 = &alloc;
  open.pPrivateRuntimeData = &runtimeData;
  open.PrivateRuntimeDataSize = query.PrivateRuntimeDataSize;

  if (D3DKMTOpenResource2(&open)) {
    WARN("ImportSharedTexture: Failed to open resource: ", hResource);
    return E_INVALIDARG;
  }

  D3DKMT_DESTROYALLOCATION destroy = {};
  destroy.hDevice = pDevice->GetLocalD3DKMT();
  destroy.hResource = open.hResource;
  D3DKMTDestroyAllocation(&destroy);

  mach_port_t mach_port;
  if (!WMTBootstrapLookUp(runtimeData.mach_port_name, &mach_port)) {
    ERR("ImportSharedTexture: Failed to look up mach port");
    return E_INVALIDARG;
  }

  switch (runtimeData.dimension)
  {
  case D3D11_RESOURCE_DIMENSION_TEXTURE1D:
    return ImportSharedTextureInternal<tag_texture_1d>(
        pDevice, &runtimeData.desc.desc1d, mach_port, riid, ppTexture
    );
  case D3D11_RESOURCE_DIMENSION_TEXTURE2D:
    return ImportSharedTextureInternal<tag_texture_2d>(
        pDevice, &runtimeData.desc.desc2d, mach_port, riid, ppTexture
    );
  case D3D11_RESOURCE_DIMENSION_TEXTURE3D:
    return ImportSharedTextureInternal<tag_texture_3d>(
        pDevice, &runtimeData.desc.desc3d, mach_port, riid, ppTexture
    );
  default:
    ERR("ImportSharedTexture: Unsupported resource dimension");
    return E_INVALIDARG;
  }
}

HRESULT
ImportSharedTextureFromNtHandle(MTLD3D11Device *pDevice, HANDLE hResource, REFIID riid, void **ppTexture) {
  InitReturnPtr(ppTexture);

  if (reinterpret_cast<uintptr_t>(hResource) & 0xc0000000) {
    WARN("ImportSharedTextureFromNtHandle: Invalid shared handle type");
    return E_INVALIDARG;
  }

  if (ppTexture == nullptr)
    return S_FALSE;

  struct SharedResourceData runtimeData;

  D3DKMT_QUERYRESOURCEINFOFROMNTHANDLE query = {};
  query.hDevice = pDevice->GetLocalD3DKMT();
  query.hNtHandle = hResource;
  query.pPrivateRuntimeData = &runtimeData;
  query.PrivateRuntimeDataSize = sizeof(runtimeData);

  if (D3DKMTQueryResourceInfoFromNtHandle(&query)) {
    WARN(str::format("ImportSharedTextureFromNtHandle: Failed to query resource: ", hResource));
    return E_INVALIDARG;
  }
  
  if (query.PrivateRuntimeDataSize != sizeof(runtimeData)) {
    WARN(str::format("ImportSharedTextureFromNtHandle: Unexpected size: ", query.PrivateRuntimeDataSize));
    return E_INVALIDARG;
  }

  D3DDDI_OPENALLOCATIONINFO2 alloc = {};
  D3DKMT_OPENRESOURCEFROMNTHANDLE open = {};
  char dummy;

  open.hDevice = pDevice->GetLocalD3DKMT();
  open.hNtHandle = hResource;
  open.NumAllocations = 1;
  open.pOpenAllocationInfo2 = &alloc;
  open.pPrivateRuntimeData = &runtimeData;
  open.PrivateRuntimeDataSize = query.PrivateRuntimeDataSize;
  open.pTotalPrivateDriverDataBuffer = &dummy;
  open.TotalPrivateDriverDataBufferSize = 0;

  if (D3DKMTOpenResourceFromNtHandle(&open)) {
    WARN(str::format("ImportSharedTextureFromNtHandle: Failed to open resource: ", hResource));
    return E_INVALIDARG;
  }

  D3DKMT_DESTROYALLOCATION destroy = {};
  destroy.hDevice = pDevice->GetLocalD3DKMT();
  destroy.hResource = open.hResource;
  D3DKMTDestroyAllocation(&destroy);

  if (open.hSyncObject) {
    WARN(str::format("ImportSharedTextureFromNtHandle: Ignoring bundled sync object"));
    D3DKMT_DESTROYSYNCHRONIZATIONOBJECT destroySync = {};
    destroySync.hSyncObject = open.hSyncObject;
    D3DKMTDestroySynchronizationObject(&destroySync);
  }
  if (open.hKeyedMutex) {
    WARN(str::format("ImportSharedTextureFromNtHandle: Ignoring bundled keyed mutex"));
    D3DKMT_DESTROYKEYEDMUTEX destroyMutex = {};
    destroyMutex.hKeyedMutex = open.hKeyedMutex;
    D3DKMTDestroyKeyedMutex(&destroyMutex);
  }

  mach_port_t mach_port;
  if (!WMTBootstrapLookUp(runtimeData.mach_port_name, &mach_port)) {
    ERR("ImportSharedTexture: Failed to look up mach port");
    return E_INVALIDARG;
  }

  switch (runtimeData.dimension)
  {
  case D3D11_RESOURCE_DIMENSION_TEXTURE1D:
    return ImportSharedTextureInternal<tag_texture_1d>(
        pDevice, &runtimeData.desc.desc1d, mach_port, riid, ppTexture
    );
  case D3D11_RESOURCE_DIMENSION_TEXTURE2D:
    return ImportSharedTextureInternal<tag_texture_2d>(
        pDevice, &runtimeData.desc.desc2d, mach_port, riid, ppTexture
    );
  case D3D11_RESOURCE_DIMENSION_TEXTURE3D:
    return ImportSharedTextureInternal<tag_texture_3d>(
        pDevice, &runtimeData.desc.desc3d, mach_port, riid, ppTexture
    );
  default:
    ERR("ImportSharedTexture: Unsupported resource dimension");
    return E_INVALIDARG;
  }
}

HRESULT
ImportSharedTextureByName(
    MTLD3D11Device *pDevice, LPCWSTR lpName, DWORD dwDesiredAccess, REFIID riid, void **ppTexture
) {
  D3DKMT_OPENNTHANDLEFROMNAME openFromName = {};
  openFromName.dwDesiredAccess = dwDesiredAccess;

  OBJECT_ATTRIBUTES attr = {};
  attr.Length = sizeof(attr);

  WCHAR buffer[MAX_PATH];
  UNICODE_STRING name_str;
  DWORD session, len, name_len = wcslen(lpName);

  ProcessIdToSessionId(GetCurrentProcessId(), &session);
  len = swprintf(buffer, ARRAYSIZE(buffer), L"\\Sessions\\%u\\BaseNamedObjects\\", session);
  memcpy(buffer + len, lpName, (name_len + 1) * sizeof(WCHAR));
  name_str.MaximumLength = name_str.Length = (len + name_len) * sizeof(WCHAR);
  name_str.MaximumLength += sizeof(WCHAR);
  name_str.Buffer = buffer;

  attr.ObjectName = &name_str;
  attr.Attributes = OBJ_CASE_INSENSITIVE;
  openFromName.pObjAttrib = &attr;

  if (D3DKMTOpenNtHandleFromName(&openFromName)) {
    WARN(str::format("ImportSharedTextureByName: Failed to open NT handle from name: ", lpName));
    return E_INVALIDARG;
  }

  HRESULT res = ImportSharedTextureFromNtHandle(pDevice, openFromName.hNtHandle, riid, ppTexture);
  CloseHandle(openFromName.hNtHandle);
  return res;
}

#pragma endregion

} // namespace dxmt