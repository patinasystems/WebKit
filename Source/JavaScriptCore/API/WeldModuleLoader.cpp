// Weld glue (patinasystems/WebKit, API/Weld*): JavaScriptCore's own ES module loader for an API global, with the
// module records built from the embedder's analysis (Kitchen lib/Weld/crates/jsc/src/module.rs; weld-module's oxc
// pass makes the analysis), so JavaScriptCore never parses a module to find its imports and exports.
//
// The embedder fetches a module graph (HTML's module map lives in web-front), then hands every module of it to
// weld_jsc_module_provide: its source, the key it is loaded by, its analysis, and the key each of its requests loads.
// Each becomes a JSModuleRecord registered under its key (ModuleRegistryEntry::provideModule), so loading the graph
// never reaches the fetch hook. A record's requests are the keys of the modules they load, so resolving one is the
// identity (the default when moduleLoaderResolve is null). weld_jsc_module_load then loads, links and evaluates the
// graph from its root through the loader: live bindings, cycles, TDZ, import.meta and top-level await are
// JavaScriptCore's. Sources go through WeldCachedProvider, so a graph keeps one bytecode payload (WJG1), as
// weld_jsc_eval_graph's does.
//
// `import()` and `import.meta.resolve` call the functions weld_jsc_module_hooks set on the global (web-front's
// module map: it resolves against the import map and fetches); `import.meta.url` is the module's URL.
#include "config.h"
#include "WeldBytecodeCache.h"

#include "APICast.h"
#include "Identifier.h"
#include "JSAPIGlobalObject.h"
#include "JSCInlines.h"
#include "JSGlobalObjectInlines.h"
#include "JSLock.h"
#include "JSModuleLoader.h"
#include "JSModuleRecord.h"
#include "JSNativeStdFunction.h"
#include "JSPromise.h"
#include "ModuleAnalyzer.h"
#include "ModuleRegistryEntry.h"
#include "Nodes.h"
#include "Parser.h"
#include "ParserError.h"
#include "TopExceptionScope.h"
#include <wtf/text/MakeString.h>
#include <wtf/text/StringBuilder.h>

using namespace JSC;
using namespace Weld;

#define WELD_EXPORT extern "C" __attribute__((visibility("default")))

// UTF-8 characters, not NUL-terminated.
struct WeldJscStr {
    const char* chars;
    size_t length;
};

// One module for weld_jsc_module_provide.
// - `module`: its source and content hash, as for weld_jsc_eval_graph; `module.url` is the module's URL: import.meta.url,
//   the referrer its `import()` resolves against, and what stack traces name.
// - `key`: what the module is registered and loaded by (HTML's module map key: the URL it was requested by).
// - `analysis`: weld-module's analysis of the source (layout below), or null: JavaScriptCore then analyzes the source
//   itself when the module loads (a source oxc could not parse; JavaScriptCore reports the syntax error).
// - `requests`: the key the analysis's request `i` loads, one per request.
struct WeldJscModuleRecordIn {
    WeldJscModule module;
    WeldJscStr key;
    const uint8_t* analysis;
    size_t analysisLength;
    const WeldJscStr* requests;
};

namespace {

// The analysis (weld-module `analysis.rs`, bumped together): little-endian u32 words, then the names, UTF-8.
//   flags | requestCount | importCount | exportCount | starCount
//   importCount x (kind, request, importName, localName)        kind: 0 single, 1 namespace (importName unused)
//   exportCount x (kind, request, exportName, name)             kind: 0 local (name: the local), 1 indirect (name: the
//                                                               imported name; request), 2 namespace (request; name unused)
//   starCount x request
//   names
// A name is two words, (offset, length) into the names; offset defaultName is the `*default*` binding (export default
// of an expression or an anonymous declaration). Requests are numbered in source order, one per import or re-export
// statement, duplicates kept; the record's requested modules are their keys in that order, each once.
constexpr uint32_t flagImportMeta = 1;
constexpr uint32_t flagTopLevelAwait = 2;
constexpr uint32_t defaultName = std::numeric_limits<uint32_t>::max();
constexpr size_t headerWords = 5;
constexpr size_t importWords = 6;
constexpr size_t exportWords = 6;

class Analysis {
public:
    // Null when `bytes` is not a well-formed analysis for `requestCount` requests.
    static std::optional<Analysis> read(std::span<const uint8_t> bytes)
    {
        Analysis a(bytes);
        if (bytes.size() < headerWords * 4 || bytes.size() % 4)
            return std::nullopt;
        a.m_flags = a.word(0);
        a.m_requestCount = a.word(1);
        a.m_importCount = a.word(2);
        a.m_exportCount = a.word(3);
        a.m_starCount = a.word(4);
        uint64_t words = headerWords + uint64_t(a.m_importCount) * importWords + uint64_t(a.m_exportCount) * exportWords + a.m_starCount;
        if (words * 4 > bytes.size())
            return std::nullopt;
        a.m_names = bytes.subspan(words * 4);
        for (uint32_t i = 0; i < a.m_importCount; ++i) {
            size_t at = a.importAt(i);
            if (a.word(at) > 1 || a.word(at + 1) >= a.m_requestCount || !a.nameFits(at + 2) || !a.nameFits(at + 4))
                return std::nullopt;
        }
        for (uint32_t i = 0; i < a.m_exportCount; ++i) {
            size_t at = a.exportAt(i);
            uint32_t kind = a.word(at);
            if (kind > 2 || (kind && a.word(at + 1) >= a.m_requestCount) || !a.nameFits(at + 2) || (kind < 2 && !a.nameFits(at + 4)))
                return std::nullopt;
        }
        for (uint32_t i = 0; i < a.m_starCount; ++i) {
            if (a.word(a.starAt(i)) >= a.m_requestCount)
                return std::nullopt;
        }
        return a;
    }

    uint32_t flags() const { return m_flags; }
    uint32_t requestCount() const { return m_requestCount; }
    uint32_t importCount() const { return m_importCount; }
    uint32_t exportCount() const { return m_exportCount; }
    uint32_t starCount() const { return m_starCount; }
    size_t importAt(uint32_t i) const { return headerWords + size_t(i) * importWords; }
    size_t exportAt(uint32_t i) const { return importAt(m_importCount) + size_t(i) * exportWords; }
    size_t starAt(uint32_t i) const { return exportAt(m_exportCount) + i; }

    uint32_t word(size_t index) const
    {
        uint32_t value;
        memcpy(&value, m_bytes.data() + index * 4, 4);
        return value;
    }

    // The name at words `at`, `at + 1`.
    Identifier name(VM& vm, size_t at) const
    {
        uint32_t offset = word(at);
        if (offset == defaultName)
            return vm.propertyNames->starDefaultPrivateName;
        auto chars = m_names.subspan(offset, word(at + 1));
        return Identifier::fromString(vm, String::fromUTF8(std::span(reinterpret_cast<const char8_t*>(chars.data()), chars.size())));
    }

private:
    explicit Analysis(std::span<const uint8_t> bytes)
        : m_bytes(bytes)
    {
    }

    bool nameFits(size_t at) const
    {
        uint32_t offset = word(at);
        return offset == defaultName || (offset <= m_names.size() && word(at + 1) <= m_names.size() - offset);
    }

    std::span<const uint8_t> m_bytes;
    std::span<const uint8_t> m_names;
    uint32_t m_flags { 0 };
    uint32_t m_requestCount { 0 };
    uint32_t m_importCount { 0 };
    uint32_t m_exportCount { 0 };
    uint32_t m_starCount { 0 };
};

Identifier keyIdentifier(VM& vm, WeldJscStr key)
{
    return Identifier::fromString(vm, String::fromUTF8(std::span(reinterpret_cast<const char8_t*>(key.chars), key.length)));
}

// The module record `analysis` describes, as ModuleAnalyzer would make it from the source. `requests[i]` is what the
// analysis's request `i` names in the record (the key it loads, or for weld_jsc_module_check the specifier itself).
JSModuleRecord* makeRecord(JSGlobalObject* globalObject, JSModuleLoader* loader, const Identifier& key, const SourceCode& source, const Analysis& analysis, const Vector<Identifier>& requests)
{
    VM& vm = globalObject->vm();
    auto* record = JSModuleRecord::create(globalObject, vm, globalObject->moduleRecordStructure(), loader, key, source, (analysis.flags() & flagImportMeta) ? ImportMetaFeature : NoFeatures);
    record->reserveCapacity(analysis.requestCount(), analysis.importCount(), analysis.exportCount());
    // ModuleAnalyzer::appendRequestedModule: each request once, in source order.
    HashSet<RefPtr<UniquedStringImpl>, IdentifierRepHash> seen;
    for (auto& request : requests) {
        if (seen.add(request.impl()).isNewEntry)
            record->appendRequestedModule(request, nullptr);
    }
    constexpr auto javaScript = ScriptFetchParameters::Type::JavaScript;
    for (uint32_t i = 0; i < analysis.importCount(); ++i) {
        size_t at = analysis.importAt(i);
        bool isNamespace = analysis.word(at) == 1;
        record->addImportEntry(JSModuleRecord::ImportEntry {
            isNamespace ? JSModuleRecord::ImportEntryType::Namespace : JSModuleRecord::ImportEntryType::Single,
            AbstractModuleRecord::ModulePhase::Evaluation,
            javaScript,
            requests[analysis.word(at + 1)],
            isNamespace ? vm.propertyNames->starNamespacePrivateName : analysis.name(vm, at + 2),
            analysis.name(vm, at + 4),
        });
    }
    for (uint32_t i = 0; i < analysis.exportCount(); ++i) {
        size_t at = analysis.exportAt(i);
        Identifier exportName = analysis.name(vm, at + 2);
        switch (analysis.word(at)) {
        case 0:
            record->addExportEntry(JSModuleRecord::ExportEntry::createLocal(exportName, analysis.name(vm, at + 4)));
            break;
        case 1:
            record->addExportEntry(JSModuleRecord::ExportEntry::createIndirect(exportName, analysis.name(vm, at + 4), requests[analysis.word(at + 1)], javaScript));
            break;
        default:
            record->addExportEntry(JSModuleRecord::ExportEntry::createNamespace(exportName, requests[analysis.word(at + 1)], javaScript));
            break;
        }
    }
    for (uint32_t i = 0; i < analysis.starCount(); ++i)
        record->addStarExportEntry(requests[analysis.word(analysis.starAt(i))], javaScript);
    record->setHasTLA(analysis.flags() & flagTopLevelAwait);
    return record;
}

// A record's entries as text, each kind sorted, so two records of one source compare equal when ModuleAnalyzer and the
// analysis agree (Bun's BunAnalyzeTranspiledModule.cpp `dumpRecordInfo`, which this follows).
String describe(JSModuleRecord* record)
{
    Vector<String> lines;
    auto sortedInto = [&](ASCIILiteral title, Vector<String>&& entries) {
        std::ranges::sort(entries, [](const String& a, const String& b) { return codePointCompare(a, b) < 0; });
        lines.append(makeString(title, ' ', entries.size()));
        lines.appendVector(WTF::move(entries));
    };
    Vector<String> requests;
    for (auto& request : record->requestedModules())
        requests.append(makeString("  module("_s, request.m_specifier.string(), ')'));
    // Requests keep their order: it is the evaluation order.
    lines.append(makeString("requests "_s, requests.size()));
    lines.appendVector(WTF::move(requests));
    Vector<String> imports;
    for (auto& pair : record->importEntries()) {
        auto& entry = pair.value;
        imports.append(makeString("  import("_s, StringView(entry.importName.impl()), "), local("_s, StringView(entry.localName.impl()), "), module("_s, StringView(entry.moduleRequest.impl()), "), type("_s, static_cast<unsigned>(entry.type), ')'));
    }
    sortedInto("imports"_s, WTF::move(imports));
    Vector<String> exports;
    for (auto& pair : record->exportEntries()) {
        auto& entry = pair.value;
        switch (entry.type) {
        case AbstractModuleRecord::ExportEntry::Type::Local:
            exports.append(makeString("  local export("_s, StringView(entry.exportName.impl()), "), local("_s, StringView(entry.localName.impl()), ')'));
            break;
        case AbstractModuleRecord::ExportEntry::Type::Indirect:
            exports.append(makeString("  indirect export("_s, StringView(entry.exportName.impl()), "), import("_s, StringView(entry.importName.impl()), "), module("_s, StringView(entry.moduleName.impl()), ')'));
            break;
        case AbstractModuleRecord::ExportEntry::Type::Namespace:
            exports.append(makeString("  namespace export("_s, StringView(entry.exportName.impl()), "), module("_s, StringView(entry.moduleName.impl()), ')'));
            break;
        }
    }
    sortedInto("exports"_s, WTF::move(exports));
    Vector<String> stars;
    for (auto& [module, type] : record->starExportEntries())
        stars.append(makeString("  star module("_s, StringView(module.get()), ')'));
    sortedInto("stars"_s, WTF::move(stars));
    lines.append(makeString("import.meta "_s, (record->features() & ImportMetaFeature) ? 1 : 0, ", top-level await "_s, record->hasTLA() ? 1 : 0));
    StringBuilder builder;
    for (auto& line : lines) {
        builder.append(line);
        builder.append('\n');
    }
    return builder.toString();
}

char* mallocUTF8(const String& text)
{
    auto utf8 = text.utf8();
    auto* out = static_cast<char*>(malloc(utf8.length() + 1));
    if (!out)
        return nullptr;
    memcpy(out, utf8.data(), utf8.length());
    out[utf8.length()] = 0;
    return out;
}

}

#if !JSC_OBJC_API_ENABLED

// `import(specifier)` in this global: the hook's result (a promise of the namespace, from any realm) as a promise of
// this one; a throw rejects it. No hook: a TypeError, as a module map with nothing behind it would give.
JSPromise* JSAPIGlobalObject::moduleLoaderImportModule(JSGlobalObject* globalObject, JSModuleLoader*, JSString* specifier, RefPtr<ScriptFetchParameters>, const SourceOrigin& referrer, bool)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    auto* hook = uncheckedDowncast<JSAPIGlobalObject>(globalObject)->weldImportModule();
    if (!hook) {
        scope.release();
        return JSPromise::rejectedPromise(globalObject, createTypeError(globalObject, "import() is not available in this realm"_s));
    }
    auto callData = JSC::getCallData(hook);
    MarkedArgumentBuffer arguments;
    arguments.append(specifier);
    arguments.append(jsString(vm, referrer.string()));
    ASSERT(!arguments.hasOverflowed());
    JSValue result = call(globalObject, hook, callData, jsUndefined(), arguments);
    if (scope.exception()) [[unlikely]] {
        auto* promise = JSPromise::create(vm, globalObject->promiseStructure());
        return promise->rejectWithCaughtException(vm, scope);
    }
    if (auto* promise = dynamicDowncast<JSPromise>(result); promise && promise->globalObject() == globalObject)
        return promise;
    // Another realm's promise, or a thenable: adopted, as `Promise.resolve` would.
    JSPromise* adopted = JSPromise::resolvedPromise(globalObject, result);
    if (scope.exception()) [[unlikely]] {
        auto* promise = JSPromise::create(vm, globalObject->promiseStructure());
        return promise->rejectWithCaughtException(vm, scope);
    }
    return adopted;
}

// import.meta of a module record: `url`, the module's URL, and `resolve(specifier)` when the global has the hook.
JSObject* JSAPIGlobalObject::moduleLoaderCreateImportMetaProperties(JSGlobalObject* globalObject, JSModuleLoader*, JSValue, JSModuleRecord* record, RefPtr<ScriptFetcher>)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    JSObject* meta = constructEmptyObject(vm, globalObject->nullPrototypeObjectStructure());
    String url = record->sourceCode().provider()->sourceURL();
    meta->putDirect(vm, Identifier::fromString(vm, "url"_s), jsString(vm, url));
    if (uncheckedDowncast<JSAPIGlobalObject>(globalObject)->weldResolveModule()) {
        auto* resolve = JSNativeStdFunction::create(vm, globalObject, 1, "resolve"_s, [url](JSGlobalObject* globalObject, CallFrame* callFrame) -> EncodedJSValue {
            VM& vm = globalObject->vm();
            auto scope = DECLARE_THROW_SCOPE(vm);
            auto* hook = uncheckedDowncast<JSAPIGlobalObject>(globalObject)->weldResolveModule();
            if (!hook)
                return throwVMTypeError(globalObject, scope, "import.meta.resolve is not available in this realm"_s);
            auto callData = JSC::getCallData(hook);
            MarkedArgumentBuffer arguments;
            arguments.append(callFrame->argument(0));
            arguments.append(jsString(vm, url));
            ASSERT(!arguments.hasOverflowed());
            RELEASE_AND_RETURN(scope, JSValue::encode(call(globalObject, hook, callData, jsUndefined(), arguments)));
        });
        RETURN_IF_EXCEPTION(scope, meta);
        meta->putDirect(vm, Identifier::fromString(vm, "resolve"_s), resolve);
    }
    return meta;
}

#endif

// Sets the functions `import()` and `import.meta.resolve` call in `ctx`'s global (either may be null, or not callable:
// no hook).
// `importModule(specifier, referrerURL)` returns a promise of the module's namespace; `resolveModule(specifier, url)`
// returns the URL. They run with the API lock held, on the global's thread; the collector never calls them.
WELD_EXPORT void weld_jsc_module_hooks(JSContextRef ctx, JSObjectRef importModule, JSObjectRef resolveModule)
{
    JSGlobalObject* globalObject = toJS(ctx);
    VM& vm = globalObject->vm();
    JSLockHolder lock(vm);
    auto* api = dynamicDowncast<JSAPIGlobalObject>(globalObject);
    RELEASE_ASSERT(api);
    auto callable = [](JSObjectRef function) -> JSObject* {
        JSObject* object = function ? toJS(function) : nullptr;
        return object && object->isCallable() ? object : nullptr;
    };
    api->setWeldModuleHooks(vm, callable(importModule), callable(resolveModule));
}

// Registers `count` modules with `ctx`'s global's module loader, each under its key, so loading them never fetches. A
// module with an analysis becomes its module record now; one without is registered as source, which JavaScriptCore
// analyzes when it loads. A key the loader already holds keeps its module (the source is released at once).
// `payload` (may be empty) is one weld_jsc_graph_commit wrote for these modules (keyed by their content ids): each
// module reads its own slice in place, and `release(context)` runs once JavaScriptCore reads none of it. `releaseSource`
// releases sources as for weld_jsc_eval_graph. With `graph` set, `*graph` receives the graph's handle
// (weld_jsc_graph_commit, weld_jsc_graph_release). Returns how many modules were registered. A malformed analysis is
// the embedder's bug, and stops the process.
WELD_EXPORT size_t weld_jsc_module_provide(JSContextRef ctx, const WeldJscModuleRecordIn* modules, size_t count,
    const uint8_t* payload, size_t payloadLength, ReleasePayload release, void* context, ReleasePayload releaseSource, WeldJscGraph** graph)
{
    JSGlobalObject* globalObject = toJS(ctx);
    VM& vm = globalObject->vm();
    JSLockHolder lock(vm);
    auto scope = DECLARE_TOP_EXCEPTION_SCOPE(vm);
    auto held = GraphRelease::create(release, context);
    JSModuleLoader* loader = globalObject->moduleLoader();
    std::span<const uint8_t> bytes(payload, payloadLength);
    auto* handle = graph ? new Graph : nullptr;
    if (handle) {
        handle->modules.reserveInitialCapacity(count);
        handle->flags.reserveInitialCapacity(count);
    }
    size_t registered = 0;
    constexpr auto javaScript = ScriptFetchParameters::Type::JavaScript;
    for (size_t i = 0; i < count; ++i) {
        auto& in = modules[i];
        Identifier key = keyIdentifier(vm, in.key);
        auto entry = graphEntry(bytes, count, i);
        ModuleRegistryEntry* registryEntry = loader->ensureRegistered(globalObject, key, javaScript);
        if (scope.exception() || registryEntry->status() != ModuleRegistryEntry::Status::New) {
            scope.clearException();
            if (in.module.sourceContext && releaseSource)
                releaseSource(in.module.sourceContext);
            if (handle) {
                // Its slice stays in the graph's payload as it was: the module that holds the key compiled it.
                handle->modules.append(WeldCachedProvider::create(emptyString(), String::fromUTF8(in.module.url), entry.payload,
                    CachePayload::Destructor([held = held.copyRef()](const void*) { }), 0, SourceProviderSourceType::Module));
                handle->flags.append(entry.flags & graphASCII);
            }
            continue;
        }
        auto [text, ascii] = sourceString(in.module, releaseSource, entry.flags & graphASCII);
        auto provider = WeldCachedProvider::create(text, String::fromUTF8(in.module.url), entry.payload,
            CachePayload::Destructor([held = held.copyRef()](const void*) { }), in.module.hash & 0xff'ffff, SourceProviderSourceType::Module);
        SourceCode source(provider.copyRef());
        if (in.analysis) {
            auto analysis = Analysis::read(std::span(in.analysis, in.analysisLength));
            RELEASE_ASSERT_WITH_MESSAGE(analysis, "weld_jsc_module_provide: module %zu's analysis is malformed", i);
            Vector<Identifier> requests(analysis->requestCount(), [&](size_t r) { return keyIdentifier(vm, in.requests[r]); });
            auto* record = makeRecord(globalObject, loader, key, source, *analysis, requests);
            registryEntry->provideModule(vm, record);
        } else
            registryEntry->provideFetch(globalObject, WTF::move(source));
        scope.clearException();
        ++registered;
        if (handle) {
            handle->modules.append(WTF::move(provider));
            handle->flags.append(ascii ? graphASCII : 0);
        }
    }
    if (graph)
        *graph = reinterpret_cast<WeldJscGraph*>(handle);
    return registered;
}

// Loads, links and evaluates the module graph rooted at `key` (its modules provided first) as `import(key)` would, in
// `ctx`'s global. Returns a promise of the root's namespace, settled once the graph has run (top-level await
// included); it rejects with what failed to link or threw. Null only when the promise could not be made.
WELD_EXPORT JSValueRef weld_jsc_module_load(JSContextRef ctx, const char* key, size_t keyLength)
{
    JSGlobalObject* globalObject = toJS(ctx);
    VM& vm = globalObject->vm();
    JSLockHolder lock(vm);
    auto scope = DECLARE_TOP_EXCEPTION_SCOPE(vm);
    JSPromise* promise = globalObject->moduleLoader()->requestImportModule(globalObject, keyIdentifier(vm, { key, keyLength }), Identifier(), nullptr, nullptr);
    if (scope.exception()) [[unlikely]] {
        JSValue error = scope.exception()->value();
        scope.clearException();
        promise = JSPromise::rejectedPromise(globalObject, error);
        if (scope.exception()) {
            scope.clearException();
            return nullptr;
        }
    }
    return toRef(globalObject, promise);
}

// Checks an analysis against ModuleAnalyzer's own analysis of `source` (UTF-8): the record each makes, with request
// `i` named by `requests[i]` (pass the specifiers). Returns null when they agree; else, malloc'd (weld_jsc_free), the two
// records' entries ("analysis:" then "JavaScriptCore:"), or JavaScriptCore's syntax error.
WELD_EXPORT char* weld_jsc_module_check(JSContextRef ctx, const char* source, size_t length, const uint8_t* analysisBytes, size_t analysisLength, const WeldJscStr* requests)
{
    JSGlobalObject* globalObject = toJS(ctx);
    VM& vm = globalObject->vm();
    JSLockHolder lock(vm);
    auto scope = DECLARE_TOP_EXCEPTION_SCOPE(vm);
    String text = String::fromUTF8(std::span(reinterpret_cast<const char8_t*>(source), length));
    auto provider = StringSourceProvider::create(text, SourceOrigin(URL({ }, "weld-check:"_s)), String("weld-check:"_s), SourceTaintedness { }, TextPosition(), SourceProviderSourceType::Module);
    SourceCode code(WTF::move(provider));
    Identifier key = Identifier::fromString(vm, "weld-check:"_s);
    auto analysis = Analysis::read(std::span(analysisBytes, analysisLength));
    if (!analysis)
        return mallocUTF8("the analysis is malformed"_s);
    Vector<Identifier> names(analysis->requestCount(), [&](size_t r) { return keyIdentifier(vm, requests[r]); });
    String ours = describe(makeRecord(globalObject, globalObject->moduleLoader(), key, code, *analysis, names));

    ParserError error;
    std::unique_ptr<ModuleProgramNode> node = parseRootNode<ModuleProgramNode>(vm, code, ImplementationVisibility::Public, JSParserBuiltinMode::NotBuiltin,
        StrictModeLexicallyScopedFeature, JSParserScriptMode::Module, SourceParseMode::ModuleAnalyzeMode, error);
    if (error.isValid())
        return mallocUTF8(makeString("JavaScriptCore: "_s, error.message()));
    ModuleAnalyzer analyzer(globalObject, globalObject->moduleLoader(), key, code, node->features());
    if (scope.exception()) {
        scope.clearException();
        return mallocUTF8("JavaScriptCore: the analyzer threw"_s);
    }
    auto result = analyzer.analyze(*node);
    if (!result)
        return mallocUTF8(makeString("JavaScriptCore: "_s, std::get<1>(result.error())));
    String theirs = describe(result.value());
    if (ours == theirs)
        return nullptr;
    return mallocUTF8(makeString("analysis:\n"_s, ours, "JavaScriptCore:\n"_s, theirs));
}
