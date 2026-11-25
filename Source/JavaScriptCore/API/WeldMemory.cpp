// Weld glue (patinasystems/WebKit, API/Weld*): closing a realm, memory back to the OS after it closes, length-counted
// Latin-1 strings and heap snapshots, over a C ABI weld-jsc calls (Kitchen lib/Weld/crates/jsc/src/sys.rs).
#include "config.h"

#include "APICast.h"
#include "CodeBlock.h"
#include "DeferGCInlines.h"
#include "Heap.h"
#include "HeapProfiler.h"
#include "HeapSnapshotBuilder.h"
#include "InitializeThreading.h"
#include "JSCInlines.h"
#include "OpaqueJSString.h"
#include "JITWorklist.h"
#include "SourceProviderCache.h"
#include "StackVisitor.h"
#include <stdio.h>
#include <wtf/FastMalloc.h>
#include <wtf/text/CString.h>

using namespace JSC;

// Closes a realm: it runs no more script. Its queued microtasks are discarded (microtaskRunnability Discard), its
// pending deferred work (FinalizationRegistry cleanups, Atomics.waitAsync, wasm compiles) is cancelled with its
// dependencies, and deferred work queued for it later is dropped unrun (JSAPIGlobalObject::scriptExecutionStatus answers
// Stopped). The embedder frees what the realm's host functions read when it closes the realm, but on a shared VM the
// global lives on until a collection proves it dead, and a FinalizationRegistry callback run in it meanwhile would read
// freed memory. Call it before the embedder's last JSGlobalContextRelease. Closing twice is a no-op.
extern "C" __attribute__((visibility("default"))) void weld_jsc_global_close(JSGlobalContextRef ctx)
{
    JSGlobalObject* globalObject = toJS(ctx);
    VM& vm = globalObject->vm();
    JSLockHolder lock(vm);
    globalObject->setMicrotaskRunnability(QueuedTaskResult::Discard);
    globalObject->clearWeakTickets();
}

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

// Test probes: read-only, no side effects, for tests that must prove a JIT premise. Not for product code.
//
// The JIT tier of the nearest JS frame below the current host call: 0 LLInt or none, 1 Baseline, 2 DFG, 3 FTL. A frame
// the DFG inlined reports its own Baseline code block, so it reads 1 even inside optimized code. -1 outside a host call.
extern "C" __attribute__((visibility("default"))) int weld_jsc_test_caller_tier(JSContextRef ctx)
{
    JSGlobalObject* globalObject = toJS(ctx);
    VM& vm = globalObject->vm();
    JSLockHolder lock(vm);
    if (!vm.topCallFrame)
        return -1;
    int tier = -1;
    StackVisitor::visit(vm.topCallFrame, vm, [&](StackVisitor& visitor) -> IterationStatus {
        CodeBlock* codeBlock = visitor->codeBlock();
        if (!codeBlock)
            return IterationStatus::Continue;
        switch (codeBlock->jitType()) {
        case JITType::BaselineJIT:
            tier = 1;
            break;
        case JITType::DFGJIT:
            tier = 2;
            break;
        case JITType::FTLJIT:
            tier = 3;
            break;
        default:
            tier = 0;
            break;
        }
        return IterationStatus::Done;
    });
    return tier;
}

// Every JIT plan the worklist holds for the group's VM: queued, compiling, or finished and waiting to be installed (the
// plans completeAllPlansForVM installs); `*ready` (if not null) gets the finished ones. Reads only.
extern "C" __attribute__((visibility("default"))) size_t weld_jsc_test_jit_plans(JSContextGroupRef group, size_t* ready)
{
    VM& vm = *toJS(group);
    JSLockHolder lock(vm);
    size_t count = 0;
    size_t finished = 0;
#if ENABLE(JIT)
    if (auto* worklist = JITWorklist::existingGlobalWorklistOrNull())
        count = worklist->planCountForVM(vm, finished);
#endif
    if (ready)
        *ready = finished;
    return count;
}
