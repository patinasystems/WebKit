// Weld glue (patinasystems/WebKit, API/Weld*): evaluate a script through a SourceProvider that serves
// JSC's bytecode cache and collects what the script compiles, over a C ABI weld-jsc calls.
//
// A script keeps its provider (WeldJscScript) for as long as the embedder holds it. Every function the
// script compiles after it was evaluated (module bodies, lazily compiled functions) is added to the
// provider's CachedBytecode as an update, and weld_jsc_script_commit writes the payload with every
// update so far, so the next process decodes those functions instead of parsing them.
#include "config.h"
#include "WeldBytecodeCache.h"

#include "APICast.h"
#include "Completion.h"
#include "Exception.h"
#include "JSArray.h"
#include "JSCInlines.h"
#include "JSGlobalObjectInlines.h"
#include "JSLock.h"
#include "TopExceptionScope.h"

using namespace JSC;
using namespace Weld;

static Graph* toGraph(WeldJscGraph* graph) { return reinterpret_cast<Graph*>(graph); }

static WeldCachedProvider* toProvider(WeldJscScript* script) { return reinterpret_cast<WeldCachedProvider*>(script); }

// Evaluates `source` (Latin-1 when `latin1` is set, else UTF-8) as a classic script named `url`.
// `payload` (may be empty) is a payload an earlier commit wrote for this source; it is used in place
// and `release(context)` is called once JSC no longer reads it (also at once when it is empty).
// When `script` is not null it receives the script's handle, for weld_jsc_script_commit and
// weld_jsc_script_release. Returns the completion value, or null with `*exception` set.
extern "C" __attribute__((visibility("default"))) JSValueRef weld_jsc_eval_script(JSContextRef ctx, const char* source, size_t length, int latin1,
    const char* url, const uint8_t* payload, size_t payloadLength, ReleasePayload release, void* context, WeldJscScript** script, JSValueRef* exception)
{
    JSGlobalObject* globalObject = toJS(ctx);
    VM& vm = globalObject->vm();
    JSLockHolder lock(vm);
    String text = latin1 ? String(std::span(reinterpret_cast<const Latin1Character*>(source), length))
                         : String::fromUTF8(std::span(reinterpret_cast<const char8_t*>(source), length));
    auto provider = WeldCachedProvider::create(text, String::fromUTF8(url), std::span(payload, payloadLength), release, context);
    SourceCode code(provider.copyRef());
    NakedPtr<Exception> thrown;
    JSValue result = JSC::evaluate(globalObject, code, JSValue(), thrown);
    if (script)
        *script = reinterpret_cast<WeldJscScript*>(&provider.leakRef());
    if (thrown) {
        if (exception)
            *exception = toRef(globalObject, thrown->value());
        return nullptr;
    }
    return toRef(globalObject, result);
}

// The script's payload with everything it compiled so far, malloc'd into `*out`; 0 (and null) when
// nothing was compiled since it was loaded or last committed. `group` is the script's context group;
// on its thread.
extern "C" __attribute__((visibility("default"))) size_t weld_jsc_script_commit(JSContextGroupRef group, WeldJscScript* script, uint8_t** out)
{
    JSLockHolder lock(toJS(group));
    return toProvider(script)->commit(out);
}

// Drops the handle weld_jsc_eval_script returned. `group` is the script's context group; on its thread.
extern "C" __attribute__((visibility("default"))) void weld_jsc_script_release(JSContextGroupRef group, WeldJscScript* script)
{
    JSLockHolder lock(toJS(group));
    toProvider(script)->deref();
}

extern "C" __attribute__((visibility("default"))) void weld_jsc_free(void* bytes) { free(bytes); }

// Evaluates a module graph's scripts in order, each a classic script named by its URL, in one call: the payload
// (`payload`, may be empty) is one a weld_jsc_graph_commit wrote for these modules (the embedder keys it by their
// content ids); each module reads its own slice of it in place, and `release(context)` runs once JSC reads none of it
// (also at once when it is empty). `releaseSource` releases modules' sources (WeldJscModule::sourceContext); it may run
// on any thread. Returns an array whose element i is module i's completion value, or its exception
// when bit i of `threw` (count / 64 rounded up words, zeroed by the caller) is set; null when the array can't be made.
// When `graph` is not null it receives the graph's handle, for weld_jsc_graph_commit and weld_jsc_graph_release.
extern "C" __attribute__((visibility("default"))) JSValueRef weld_jsc_eval_graph(JSContextRef ctx, const WeldJscModule* modules, size_t count,
    const uint8_t* payload, size_t payloadLength, ReleasePayload release, void* context, ReleasePayload releaseSource, uint64_t* threw, WeldJscGraph** graph)
{
    JSGlobalObject* globalObject = toJS(ctx);
    VM& vm = globalObject->vm();
    JSLockHolder lock(vm);
    auto held = GraphRelease::create(release, context);
    if (graph)
        *graph = nullptr;
    auto scope = DECLARE_TOP_EXCEPTION_SCOPE(vm);
    JSArray* results = constructEmptyArray(globalObject, nullptr);
    if (!results || scope.exception()) {
        scope.clearException();
        // Nothing is evaluated; the sources are still the embedder's to release.
        for (size_t i = 0; i < count; ++i) {
            if (modules[i].sourceContext && releaseSource)
                releaseSource(modules[i].sourceContext);
        }
        return nullptr;
    }
    std::span<const uint8_t> bytes(payload, payloadLength);
    auto* handle = graph ? new Graph : nullptr;
    if (handle) {
        handle->modules.reserveInitialCapacity(count);
        handle->flags.reserveInitialCapacity(count);
    }
    for (size_t i = 0; i < count; ++i) {
        auto& module = modules[i];
        auto entry = graphEntry(bytes, count, i);
        auto [text, ascii] = sourceString(module, releaseSource, entry.flags & graphASCII);
        auto provider = WeldCachedProvider::create(text, String::fromUTF8(module.url), entry.payload,
            CachePayload::Destructor([held = held.copyRef()](const void*) { }), module.hash & 0xff'ffff);
        NakedPtr<Exception> thrown;
        JSValue result = JSC::evaluate(globalObject, SourceCode(provider.copyRef()), JSValue(), thrown);
        if (thrown) {
            result = thrown->value();
            threw[i / 64] |= uint64_t(1) << (i % 64);
        }
        results->putDirectIndex(globalObject, i, result);
        scope.clearException();
        if (handle) {
            handle->modules.append(WTF::move(provider));
            handle->flags.append(ascii ? graphASCII : 0);
        }
    }
    if (graph)
        *graph = reinterpret_cast<WeldJscGraph*>(handle);
    return toRef(globalObject, results);
}

// The graph's payload with everything its modules compiled so far, malloc'd into `*out`; 0 (and null) when nothing was
// compiled since it was loaded or last committed. `group` is the graph's context group; on its thread.
extern "C" __attribute__((visibility("default"))) size_t weld_jsc_graph_commit(JSContextGroupRef group, WeldJscGraph* graph, uint8_t** out)
{
    JSLockHolder lock(toJS(group));
    return toGraph(graph)->commit(out);
}

// Drops the handle weld_jsc_eval_graph returned. `group` is the graph's context group; on its thread.
extern "C" __attribute__((visibility("default"))) void weld_jsc_graph_release(JSContextGroupRef group, WeldJscGraph* graph)
{
    JSLockHolder lock(toJS(group));
    delete toGraph(graph);
}
