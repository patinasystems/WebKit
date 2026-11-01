// Weld glue (patinasystems/WebKit, API/Weld*): evaluate a script through a
// SourceProvider that serves and collects JSC's bytecode cache, over a C ABI weld-jsc calls.
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
#include <vector>

using namespace JSC;

namespace {

class WeldCachedProvider final : public StringSourceProvider {
public:
    static Ref<WeldCachedProvider> create(const String& source, const String& url, std::span<const uint8_t> cache)
    {
        return adoptRef(*new WeldCachedProvider(source, url, cache));
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

    // The original payload with every update written at its offset (what jsc's shell writes to disk).
    std::vector<uint8_t> commit() const
    {
        std::vector<uint8_t> out(m_original.begin(), m_original.end());
        if (!m_cached)
            return out;
        m_cached->commitUpdates([&](off_t offset, std::span<const uint8_t> data) {
            size_t end = static_cast<size_t>(offset) + data.size();
            if (out.size() < end)
                out.resize(end);
            memcpy(out.data() + offset, data.data(), data.size());
        });
        return out;
    }

private:
    WeldCachedProvider(const String& source, const String& url, std::span<const uint8_t> cache)
        : StringSourceProvider(source, SourceOrigin(URL({ }, url)), SourceTaintedness { }, String(url), TextPosition(), SourceProviderSourceType::Program)
        , m_original(cache.begin(), cache.end())
    {
        if (!cache.empty()) {
            auto copy = MallocSpan<uint8_t, VMMalloc>::malloc(cache.size());
            memcpy(copy.mutableSpan().data(), cache.data(), cache.size());
            m_cached = CachedBytecode::create(WTF::move(copy), { });
        }
    }

    std::vector<uint8_t> m_original;
    mutable RefPtr<CachedBytecode> m_cached;
};

}

extern "C" __attribute__((visibility("default"))) JSValueRef weld_eval_cached(JSContextRef ctx, const char* source, size_t length, const char* url,
    const uint8_t* cache, size_t cacheLength, uint8_t** out, size_t* outLength, JSValueRef* exception)
{
    JSGlobalObject* globalObject = toJS(ctx);
    VM& vm = globalObject->vm();
    JSLockHolder lock(vm);
    auto text = String::fromUTF8(std::span(reinterpret_cast<const char8_t*>(source), length));
    auto provider = WeldCachedProvider::create(text, String::fromUTF8(url), std::span(cache, cacheLength));
    SourceCode code(provider.copyRef());
    NakedPtr<Exception> thrown;
    JSValue result = JSC::evaluate(globalObject, code, JSValue(), thrown);
    if (out) {
        auto bytes = provider->commit();
        *outLength = bytes.size();
        *out = static_cast<uint8_t*>(malloc(bytes.size() ? bytes.size() : 1));
        if (!bytes.empty())
            memcpy(*out, bytes.data(), bytes.size());
    }
    if (thrown) {
        if (exception)
            *exception = toRef(globalObject, thrown->value());
        return nullptr;
    }
    return toRef(globalObject, result);
}

extern "C" __attribute__((visibility("default"))) void weld_cache_free(void* bytes) { free(bytes); }
