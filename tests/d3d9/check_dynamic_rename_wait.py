#!/usr/bin/env python3
"""Run both production mirror-refresh methods (via the fork's refreshMirror) with failed rename and GPU-drain fixtures."""
from pathlib import Path
import subprocess
import tempfile

source = (Path(__file__).resolve().parents[2] / 'src/d3d9/d3d9_buffer.cpp').read_text()
# The fork keeps its aliased/write-watch buffers: both refreshWholeMirror methods
# delegate to the shared refreshMirror helper, which owns the rename path.
start = source.index('template <typename Sequences>\nbool\nrefreshMirror(')
helper = source[start:source.index('\n}', start) + 2]
methods = []
for name in ['MTLD3D9VertexBuffer', 'MTLD3D9IndexBuffer']:
    start = source.index('void\n' + name + '::refreshWholeMirror()')
    methods.append(source[start:source.index('\n}', start) + 2])

fixture = r'''
#include <atomic>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <tuple>
#include <utility>
#include <vector>
typedef unsigned UINT;
static int failures, calls, writes, waits, in_place;
static bool wait_ok, gpu_idle;
static uint64_t last_seq, last_signal;
#define FAILED(hr) ((hr) < 0)
struct Logger { static void warn(const char *) {} };
struct Name { void updateContents(int, void *, unsigned, int) {
    assert(calls <= failures ? gpu_idle : true); ++writes;
} };
struct DynamicBuffer {
    Name name;
    DynamicBuffer *ptr() { return this; }
    Name *immediateName() { return &name; }
    int immediateSuballocation() { return 0; }
};
struct D3D9BufferRefreshState { bool whole = true; void refreshed() { whole = false; } };
struct MTLD3D9Device {
    uint64_t m_currentCmdSeq = 2;
    std::atomic<uint64_t> m_cachedSignaled{1};
    bool drainForDynamicRename() {
        ++waits; gpu_idle = wait_ok;
        if (wait_ok) { m_currentCmdSeq = 3; m_cachedSignaled = 2; }
        return wait_ok;
    }
    void noteRenameInPlace() { assert(gpu_idle); ++in_place; }
};
int renameDynamicBuffer(MTLD3D9Device *, DynamicBuffer *, uint64_t seq, uint64_t signal) {
    last_seq = seq; last_signal = signal;
    return ++calls <= failures ? -1 : 0;
}
void recordRefresh(bool, bool, uint64_t, size_t) {}
''' + helper + r'''
struct Mirror {
    bool m_dirty = true, m_aliased = false, m_writeLocked = false, m_writeWatch = false;
    MTLD3D9Device device;
    MTLD3D9Device *m_device = &device;
    DynamicBuffer m_dynamic;
    char storage[16];
    void *m_hostPtr = storage;
    UINT m_size = 16;
    D3D9BufferRefreshState m_refreshState;
    std::vector<void *> m_writtenPages;
};
struct MTLD3D9VertexBuffer : Mirror { void refreshWholeMirror(); };
struct MTLD3D9IndexBuffer : Mirror { void refreshWholeMirror(); };
'''
fixture += '\n'.join(methods)
fixture += r'''
template<class Buffer> void check() {
    for (int fail : {0, 1, 2}) for (bool wait : {false, true}) {
        failures = fail; calls = writes = waits = in_place = 0; gpu_idle = false; wait_ok = wait;
        Buffer buffer;
        buffer.refreshWholeMirror();
        assert(calls == (fail && wait ? 2 : 1));
        assert(waits == (fail ? 1 : 0));
        assert(writes == (!fail || wait ? 1 : 0));
        assert(in_place == (fail == 2 && wait ? 1 : 0));
        assert(buffer.m_dirty == (fail && !wait));
        if (fail && wait) assert(last_seq == 3 && last_signal == 2);
    }
    calls = 0; Buffer aliased; aliased.m_aliased = true;
    aliased.refreshWholeMirror(); assert(calls == 0);
    Buffer clean; clean.m_dirty = false;
    clean.refreshWholeMirror(); assert(calls == 0);
    failures = 0; Buffer locked; locked.m_writeLocked = true;
    locked.refreshWholeMirror(); assert(locked.m_dirty);
}
int main() { check<MTLD3D9VertexBuffer>(); check<MTLD3D9IndexBuffer>(); }
'''
with tempfile.TemporaryDirectory(prefix='dxmt-rename-') as folder:
    path = Path(folder)
    (path / 'check.cpp').write_text(fixture)
    subprocess.run(['c++', '-std=c++20', '-Wall', '-Wextra', '-Werror', '-Wno-unused-parameter', '-fsanitize=address,undefined',
                    str(path / 'check.cpp'), '-o', str(path / 'check')], check=True)
    subprocess.run([str(path / 'check')], check=True)
print('PASS: both mirror callers retry after the GPU drain, preserve failed waits and skip aliased buffers')
