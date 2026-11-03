// Weld glue (patinasystems/WebKit, API/Weld*): evaluate a script through a SourceProvider that serves
// JSC's bytecode cache and collects what the script compiles, over a C ABI weld-jsc calls.
//
// A script keeps its provider (WeldJscScript) for as long as the embedder holds it. Every function the
// script compiles after it was evaluated (module bodies, lazily compiled functions) is added to the
// provider's CachedBytecode as an update, and weld_jsc_script_commit writes the payload with every
// update so far, so the next process decodes those functions instead of parsing them.
#include "config.h"

#include "APICast.h"
#include "BytecodeCacheError.h"
#include "CachedBytecode.h"
#include "CachedTypes.h"
#include "Completion.h"
#include "Exception.h"
#include "JSCInlines.h"
#include "JSLock.h"
#include "SourceCode.h"
#include "SourceProvider.h"
#include "UnlinkedFunctionExecutable.h"
#include <cstdlib>
#include <cstring>

using namespace JSC;

namespace {

using ReleasePayload = void (*)(void*);

class WeldCachedProvider final : public StringSourceProvider {
public:
    static Ref<WeldCachedProvider> create(const String& source, const String& url, std::span<const uint8_t> payload, ReleasePayload release, void* context)
    {
        return adoptRef(*new WeldCachedProvider(source, url, payload, release, context));
    }

    RefPtr<CachedBytecode> cachedBytecode() const final { return m_cached; }

    void cacheBytecode(const BytecodeCacheGenerator& generator) const final
    {
        if (!m_cached)
            m_cached = CachedBytecode::create();
        if (auto update = generator())
            m_cached->addGlobalUpdate(*update);
    }

    void updateCache(const UnlinkedFunctionExecutable* executable, const SourceCode&, CodeSpecializationKind kind, const UnlinkedFunctionCodeBlock* codeBlock) const final
    {
        if (!m_cached)
            return;
        BytecodeCacheError error;
        RefPtr<CachedBytecode> update = encodeFunctionCodeBlock(executable->vm(), codeBlock, error);
        if (update && !error.isValid())
            m_cached->addFunctionUpdate(executable, kind, *update);
    }

    // The payload with every update written at its offset (as jsc's shell writes it to disk), or 0 when
    // nothing was added since the last commit. The bytes are malloc'd; weld_jsc_free frees them.
    size_t commit(uint8_t** out)
    {
        *out = nullptr;
        if (!m_cached || !m_cached->hasUpdates() || m_cached->sizeForUpdate() == m_committed)
            return 0;
        auto original = m_cached->span();
        size_t size = m_cached->sizeForUpdate();
        auto* bytes = static_cast<uint8_t*>(malloc(size));
        if (!bytes)
            return 0;
        if (!original.empty())
            memcpy(bytes, original.data(), original.size());
        m_cached->commitUpdates([&](off_t offset, std::span<const uint8_t> data) {
            RELEASE_ASSERT(static_cast<size_t>(offset) + data.size() <= size);
            memcpy(bytes + offset, data.data(), data.size());
        });
        m_committed = size;
        *out = bytes;
        return size;
    }

private:
    WeldCachedProvider(const String& source, const String& url, std::span<const uint8_t> payload, ReleasePayload release, void* context)
        : StringSourceProvider(source, SourceOrigin(URL({ }, url)), SourceTaintedness { }, String(url), TextPosition(), SourceProviderSourceType::Program)
    {
        if (payload.empty()) {
            if (release)
                release(context);
            return;
        }
        // The embedder's bytes (a mapped file) are used in place, and released when the last decoder is done.
        std::span<uint8_t> bytes(const_cast<uint8_t*>(payload.data()), payload.size());
        m_cached = CachedBytecode::create(bytes, [release, context](const void*) {
            if (release)
                release(context);
        }, { });
        m_committed = m_cached->sizeForUpdate();
    }

    mutable RefPtr<CachedBytecode> m_cached;
    size_t m_committed { 0 };
};

}

struct WeldJscScript;

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
