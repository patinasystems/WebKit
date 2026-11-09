// Weld glue (patinasystems/WebKit, API/Weld*): memory back to the OS after a realm closes, length-counted Latin-1
// strings and heap snapshots, over a C ABI weld-jsc calls (Kitchen lib/Weld/crates/jsc/src/sys.rs).
#include "config.h"

#include "APICast.h"
#include "DeferGCInlines.h"
#include "Heap.h"
#include "HeapProfiler.h"
#include "HeapSnapshotBuilder.h"
#include "InitializeThreading.h"
#include "JSCInlines.h"
#include "OpaqueJSString.h"
#include "SourceProviderCache.h"
#include <stdio.h>
#include <wtf/FastMalloc.h>
#include <wtf/text/CString.h>

using namespace JSC;

// What a closed realm left behind, back to the OS, keeping the code cache: a full collection now, the heap's empty
// blocks freed (sweepSynchronously), the parser's per-source caches dropped (they hold their SourceProviders, and so
// a closed page's source text), then malloc's free pages returned (WTF::releaseFastMallocFreeMemory: the bmalloc or
// libpas scavenger, synchronously). Linked and unlinked code stay: no deleteAllCode, so a page loaded again finds
// its code cached. Under JS (a host call) only the collection and the scavenge run; the caches stay.
extern "C" __attribute__((visibility("default"))) void weld_jsc_release_free_memory(JSContextGroupRef group)
{
    VM& vm = *toJS(group);
    {
        JSLockHolder lock(vm);
        bool idle = !vm.entryScope && !vm.heap.currentThreadIsDoingGCWork();
        if (idle) {
            sanitizeStackForVM(vm);
            // The last exception keeps its captured stack (and the code on it) alive. No JS is running.
            vm.clearLastException();
            vm.sourceProviderCacheMap.clear();
        }
        vm.heap.collectNow(Synchronousness::Sync, CollectionScope::Full);
        if (idle)
            vm.heap.sweepSynchronously();
    }
    WTF::releaseFastMallocFreeMemory();
}

// An 8-bit string from `length` Latin-1 bytes, NULs included (JSStringCreateWithUTF8CString stops at the first and
// decodes UTF-8; JSStringCreateWithCharacters takes UTF-16 at 2 bytes per char). One copy, 1 byte per char.
extern "C" __attribute__((visibility("default"))) JSStringRef weld_jsc_string_latin1(const uint8_t* chars, size_t length)
{
    JSC::initialize();
    return &OpaqueJSString::create(std::span { reinterpret_cast<const Latin1Character*>(chars), length }).leakRef();
}

// A GC-debugging heap snapshot of `group`'s VM written to `path`: JSC's JSON (as jsc's
// generateHeapSnapshotForGCDebugging makes), every live cell with its edges, labels and, for roots, why each is a root,
// so a tool can name the path that keeps a cell alive. It runs a full collection. Returns the bytes written, or 0 when
// the snapshot overflowed or the file could not be written. Not under JS.
extern "C" __attribute__((visibility("default"))) size_t weld_jsc_heap_snapshot(JSContextGroupRef group, const char* path)
{
    VM& vm = *toJS(group);
    JSLockHolder lock(vm);
    if (vm.entryScope)
        return 0;
    String json;
    {
        DeferGCForAWhile deferGC(vm);
        HeapSnapshotBuilder builder(vm.ensureHeapProfiler(), HeapSnapshotBuilder::SnapshotType::GCDebuggingSnapshot);
        builder.buildSnapshot();
        if (builder.hasOverflowed())
            return 0;
        json = builder.json();
    }
    vm.ensureHeapProfiler().clearSnapshots();
    CString utf8 = json.utf8();
    FILE* file = fopen(path, "wb");
    if (!file)
        return 0;
    size_t written = fwrite(utf8.data(), 1, utf8.length(), file);
    if (fclose(file) || written != utf8.length())
        return 0;
    return written;
}
