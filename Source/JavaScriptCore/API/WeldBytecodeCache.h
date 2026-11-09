// Weld glue (patinasystems/WebKit, API/Weld*): the bytecode cache's SourceProvider and a module graph's payload (WJG1),
// shared by WeldBytecodeCache.cpp (classic scripts and graphs of them) and WeldModuleLoader.cpp (module records).
#pragma once

#include "BytecodeCacheError.h"
#include "CachedBytecode.h"
#include "CachedTypes.h"
#include "SourceCode.h"
#include "SourceProvider.h"
#include "UnlinkedFunctionExecutable.h"
#include "JSCInlines.h"
#include <wtf/ThreadSafeRefCounted.h>
#include <cstdlib>
#include <wtf/text/ASCIIFastPath.h>
#include <wtf/text/ExternalStringImpl.h>
#include <algorithm>
#include <cstring>
#include <limits>

namespace Weld {

using namespace JSC;

using ReleasePayload = void (*)(void*);

class WeldCachedProvider final : public StringSourceProvider {
public:
    static Ref<WeldCachedProvider> create(const String& source, const String& url, std::span<const uint8_t> payload, ReleasePayload release, void* context)
    {
        return adoptRef(*new WeldCachedProvider(source, url, payload, CachePayload::Destructor([release, context](const void*) {
            if (release)
                release(context);
        }), 0, SourceProviderSourceType::Program));
    }

    // A module of a graph: `payload` is a slice of the graph's payload, `done` runs when JSC stops reading it, and
    // `hash` (non-zero) stands for the source's own hash, from the content id the embedder already holds for it.
    // `type` is Module for a module record's source (WeldModuleLoader.cpp).
    static Ref<WeldCachedProvider> create(const String& source, const String& url, std::span<const uint8_t> payload, CachePayload::Destructor&& done, unsigned hash, SourceProviderSourceType type = SourceProviderSourceType::Program)
    {
        return adoptRef(*new WeldCachedProvider(source, url, payload, WTF::move(done), hash, type));
    }

    // The source's hash names it in JSC's code cache keys (SourceCodeKey) and in the payload. A graph's module passes
    // one taken from its content id, so neither the hash nor the check over the payload walks the source again.
    unsigned hash() const final { return m_hash ? m_hash : StringSourceProvider::hash(); }

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
        if (!grew())
            return 0;
        size_t size = this->size();
        auto* bytes = static_cast<uint8_t*>(malloc(size));
        if (!bytes)
            return 0;
        write(std::span(bytes, size));
        *out = bytes;
        return size;
    }

    // Whether something compiled since the payload was read or last written.
    bool grew() const { return m_cached && m_cached->hasUpdates() && m_cached->sizeForUpdate() != m_committed; }
    // The payload's size with every update so far (0: nothing cached).
    size_t size() const { return m_cached ? m_cached->sizeForUpdate() : 0; }
    // Writes the payload with every update so far into `bytes` (size() of them), and counts it as committed.
    void write(std::span<uint8_t> bytes)
    {
        if (!m_cached)
            return;
        auto original = m_cached->span();
        RELEASE_ASSERT(bytes.size() == m_cached->sizeForUpdate() && original.size() <= bytes.size());
        if (!original.empty())
            memcpy(bytes.data(), original.data(), original.size());
        m_cached->commitUpdates([&](off_t offset, std::span<const uint8_t> data) {
            RELEASE_ASSERT(static_cast<size_t>(offset) + data.size() <= bytes.size());
            memcpy(bytes.data() + offset, data.data(), data.size());
        });
        m_committed = bytes.size();
    }

private:
    WeldCachedProvider(const String& source, const String& url, std::span<const uint8_t> payload, CachePayload::Destructor&& done, unsigned hash, SourceProviderSourceType type)
        : StringSourceProvider(source, SourceOrigin(URL({ }, url)), SourceTaintedness { }, String(url), TextPosition(), type)
        , m_hash(hash)
    {
        if (payload.empty()) {
            done(nullptr);
            return;
        }
        // The embedder's bytes (a mapped file) are used in place, and released when the last decoder is done.
        std::span<uint8_t> bytes(const_cast<uint8_t*>(payload.data()), payload.size());
        m_cached = CachedBytecode::create(bytes, WTF::move(done), { });
        m_committed = m_cached->sizeForUpdate();
    }

    mutable RefPtr<CachedBytecode> m_cached;
    size_t m_committed { 0 };
    unsigned m_hash { 0 };
};

}

struct WeldJscScript;
struct WeldJscGraph;

// One module of a graph (weld_jsc_eval_graph). `source` is `length` code units in `encoding`; `hash` is the
// non-zero 24-bit hash that stands for the source (taken from its content id), or 0 to hash the source. With a
// `sourceContext`, JSC keeps the embedder's Latin-1 or UTF-16 characters in place (no copy) and calls the graph's
// `releaseSource(sourceContext)` once it no longer reads them (at once for UTF-8, which is decoded into a copy);
// without one, it copies them.
struct WeldJscModule {
    const void* source;
    size_t length;
    const char* url;
    void* sourceContext;
    uint32_t encoding; // WeldJscEncoding
    uint32_t hash;
};

enum WeldJscEncoding : uint32_t { WeldJscUTF8 = 0, WeldJscLatin1 = 1, WeldJscUTF16 = 2 };

namespace Weld {

using namespace JSC;

// A graph's payload: a header, then one module payload per module, in the graph's order, each at a 16-byte aligned
// offset. Every word is a little-endian u32. An entry's offset is aligned, so its low 4 bits are the module's flags
// (graphASCII: its UTF-8 source is all ASCII, so a hit reads it in place as Latin-1 without scanning it; the payload is
// keyed by the modules' content ids, so a hit's flags are for these very sources).
//   'W' 'J' 'G' '1' | count | count x (offset | flags, length) | payloads
constexpr uint32_t graphMagic = 0x3147'4a57; // "WJG1"
constexpr size_t graphAlignment = 16;
constexpr uint32_t graphASCII = 1;

struct GraphEntry {
    std::span<const uint8_t> payload;
    uint32_t flags { 0 };
};

inline size_t graphHeaderSize(size_t count) { return roundUpToMultipleOf<graphAlignment>(8 + 8 * count); }

inline uint32_t readU32(std::span<const uint8_t> bytes, size_t at)
{
    uint32_t value;
    memcpy(&value, bytes.data() + at, 4);
    return value;
}

// Module `index`'s entry in `graph`: empty, with no flags, when `graph` is not a payload for `count` modules or the entry is out of bounds.
inline GraphEntry graphEntry(std::span<const uint8_t> graph, size_t count, size_t index)
{
    if (graph.size() < graphHeaderSize(count) || readU32(graph, 0) != graphMagic || readU32(graph, 4) != count)
        return { };
    uint32_t word = readU32(graph, 8 + 8 * index);
    size_t offset = word & ~uint32_t(graphAlignment - 1);
    size_t length = readU32(graph, 12 + 8 * index);
    uint32_t flags = word & (graphAlignment - 1);
    if (!length)
        return { { }, offset ? 0 : flags };
    if (offset > graph.size() || length > graph.size() - offset)
        return { };
    return { graph.subspan(offset, length), flags };
}

// The embedder's release for a graph's payload, run once the last module's decoder is done with its slice.
class GraphRelease final : public ThreadSafeRefCounted<GraphRelease> {
public:
    static Ref<GraphRelease> create(ReleasePayload release, void* context) { return adoptRef(*new GraphRelease(release, context)); }
    ~GraphRelease()
    {
        if (m_release)
            m_release(m_context);
    }

private:
    GraphRelease(ReleasePayload release, void* context)
        : m_release(release)
        , m_context(context)
    {
    }
    ReleasePayload m_release;
    void* m_context;
};

// The module's source as JSC keeps it, and whether it is ASCII UTF-8 (for the graph payload's flags). ASCII UTF-8 is
// Latin-1: with a source context it is read in place; `knownASCII` (the payload's flag) skips the scan.
inline std::pair<String, bool> sourceString(const WeldJscModule& module, ReleasePayload releaseSource, bool knownASCII)
{
    void* context = module.sourceContext;
    uint32_t encoding = module.encoding;
    bool ascii = false;
    if (encoding == WeldJscUTF8) {
        ascii = knownASCII || charactersAreAllASCII(std::span(static_cast<const Latin1Character*>(module.source), module.length));
        if (ascii)
            encoding = WeldJscLatin1;
    }
    if (context && releaseSource && module.length <= StringImpl::MaxLength) {
        auto release = [releaseSource](void* context, void*, unsigned) { releaseSource(context); };
        if (encoding == WeldJscLatin1)
            return { String(ExternalStringImpl::create(std::span(static_cast<const Latin1Character*>(module.source), module.length), context, WTF::move(release))), ascii };
        if (encoding == WeldJscUTF16)
            return { String(ExternalStringImpl::create(std::span(static_cast<const char16_t*>(module.source), module.length), context, WTF::move(release))), ascii };
    }
    String text;
    switch (encoding) {
    case WeldJscLatin1:
        text = String(std::span(static_cast<const Latin1Character*>(module.source), module.length));
        break;
    case WeldJscUTF16:
        text = String(std::span(static_cast<const char16_t*>(module.source), module.length));
        break;
    default:
        text = String::fromUTF8(std::span(static_cast<const char8_t*>(module.source), module.length));
    }
    if (context && releaseSource)
        releaseSource(context);
    return { WTF::move(text), ascii };
}

class Graph {
public:
    Vector<Ref<WeldCachedProvider>> modules;
    // Per module: graphASCII when its UTF-8 source is all ASCII (the payload's flags).
    Vector<uint8_t> flags;

    // The graph's payload with what every module compiled so far, or 0 when no module compiled anything new.
    size_t commit(uint8_t** out)
    {
        *out = nullptr;
        if (!std::ranges::any_of(modules, [](auto& module) { return module->grew(); }))
            return 0;
        size_t count = modules.size();
        size_t size = graphHeaderSize(count);
        for (auto& module : modules)
            size = roundUpToMultipleOf<graphAlignment>(size) + module->size();
        if (size > std::numeric_limits<uint32_t>::max())
            return 0;
        auto* bytes = static_cast<uint8_t*>(calloc(size, 1));
        if (!bytes)
            return 0;
        auto put = [&](size_t at, uint32_t value) { memcpy(bytes + at, &value, 4); };
        put(0, graphMagic);
        put(4, count);
        size_t at = graphHeaderSize(count);
        for (size_t i = 0; i < count; ++i) {
            at = roundUpToMultipleOf<graphAlignment>(at);
            size_t length = modules[i]->size();
            put(8 + 8 * i, (length ? at : 0) | flags[i]);
            put(12 + 8 * i, length);
            modules[i]->write(std::span(bytes + at, length));
            at += length;
        }
        *out = bytes;
        return size;
    }
};

}
