// Weld glue (patinasystems/WebKit, API/Weld*): native node wrappers for weld-dom documents, over a C
// ABI weld-jsc and weld-dom call (Kitchen step 5.25 PR1; design in Kitchen's
// shared/notes/work/weld/research/native-dom-core.md §3.3).
//
// - JSWeldNode: a node wrapper cell holding (slot, generation, table): 32 bytes, plus a butterfly
//   only for the properties script (or web-front's JS classes) adds. Its prototype is a web-front
//   class's; `NodeBase` is the native base those classes extend, so `new HTMLDivElement(...)` makes
//   one when web-front armed it for a slot (`arm`), and an ordinary object otherwise.
// - WeldDomTable: one per document, a flat array of Weak handles indexed by slot (8 bytes a slot, a
//   WeakImpl only for touched nodes). Reachability is WebKit's opaque-root model: a marking
//   constraint adds the tree root of every marked wrapper as an opaque root, and the table's
//   WeakHandleOwner keeps a wrapper whose tree root is one. So any reachable wrapper keeps every
//   wrapper of its tree (their expandos and WeakMap entries stay), and the document's wrapper
//   keeps every attached one.
// - DOMJIT getters for the tree reads: parentNode, firstChild, lastChild, previousSibling,
//   nextSibling, parentElement and the element-sibling family. (nodeType and web-front's `_i`,
//   `_g`, `_m` stay JS: a prototype constant and own data properties measure faster than a
//   DOMJIT call.) They load the link
//   straight from weld-dom's shared link pages (Rust-owned memory; WeldDomLinks names them) and the
//   wrapper from the table, inline; a missing wrapper calls web-front's maker.
//
// Rust memory is read only on the mutator thread or while the world is stopped: the weak owner
// answers "not reachable" during concurrent marking, and the final stopped fixpoint answers for
// real. Rust never frees a page the table can name without first clearing its entry.
#include "config.h"

#include "APICast.h"
#include "CCallHelpers.h"
#include "DOMAttributeGetterSetter.h"
#include "DOMJITCallDOMGetterSnippet.h"
#include "DOMJITGetterSetter.h"
#include "FrameTracers.h"
#include "InternalFunction.h"
#include "JITOperations.h"
#include "JSArrayBufferViewInlines.h"
#include "JSCInlines.h"
#include "JSDestructibleObject.h"
#include "JSTypedArrays.h"
#include "ObjectConstructor.h"
#include "SimpleMarkingConstraint.h"
#include "Snippet.h"
#include "SnippetParams.h"
#include "WeakHandleOwner.h"
#include "WeakSetInlines.h"
#include <wtf/FastMalloc.h>

using namespace JSC;

extern "C" {
// weld-dom's link pages as the fork reads them (Rust `links.rs` `PageTable`): `pages[k]` is page
// k's shared copy (`[parent][first][last][prev][next][rslot]`, PAGE words each), or a page of NONE
// links while it has none (released, or the session ended). Rust owns it and keeps its address for
// the document's life. `count` never shrinks, and a wrapper is only made for a slot below it, so
// the inline reads check neither.
struct WeldDomLinks {
    uint32_t* const* pages;
    uint32_t count;
    uint32_t reserved;
};
}

namespace {

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

constexpr unsigned kPageShift = 10;
constexpr unsigned kPage = 1u << kPageShift;
constexpr uint32_t kNone = 0xFFFFFFFFu;
constexpr uint8_t kElementType = 1;
enum Column : unsigned { Parent = 0, First = 1, Last = 2, Prev = 3, Next = 4 };

constexpr JSC::JSType WeldNodeType = static_cast<JSC::JSType>(JSC::LastJSCObjectType + 1);

class JSWeldDocument;

// One document's wrappers, by slot.
class WeldDomTable final : public WeakHandleOwner {
    WTF_DEPRECATED_MAKE_FAST_ALLOCATED(WeldDomTable);
public:
    WeldDomTable(VM& vm, const WeldDomLinks* links, void (*release)(void*), void* releaseContext, uint32_t tag)
        : m_vm(vm)
        , m_links(links)
        , m_release(release)
        , m_releaseContext(releaseContext)
        , m_tag(tag)
    {
    }

    ~WeldDomTable() final;
    void destroyEntries()
    {
        for (uint32_t i = 0; i < m_size; ++i) {
            if (m_entries[i])
                WeakSet::deallocate(m_entries[i]);
        }
        fastFree(m_entries);
        if (m_release)
            m_release(m_releaseContext);
    }

    bool hasSlot(uint32_t slot) const { return slot != kNone && (slot >> kPageShift) < m_links->count; }

    uint32_t link(unsigned column, uint32_t slot) const
    {
        if (!hasSlot(slot))
            return kNone;
        return m_links->pages[slot >> kPageShift][column * kPage + (slot & (kPage - 1))];
    }

    uint8_t type(uint32_t slot) const { return slot < m_typesLength ? m_types[slot] : 0; }

    // The root of `slot`'s tree, through template contents to their template's tree.
    uint32_t root(uint32_t slot) const
    {
        uint32_t n = slot;
        for (unsigned steps = 0; steps < (1u << 26); ++steps) {
            uint32_t p = link(Parent, n);
            if (p == kNone) {
                p = ownerOf(n);
                if (p == kNone)
                    return n;
            }
            n = p;
        }
        return n;
    }

    void* rootKey(uint32_t slot) const
    {
        // Never a pointer: the top bit is set. One tag per table.
        return reinterpret_cast<void*>((uint64_t(1) << 63) | (uint64_t(m_tag) << 32) | root(slot));
    }

    uint32_t ownerOf(uint32_t slot) const
    {
        for (auto& [content, owner] : m_owners) {
            if (content == slot)
                return owner;
        }
        return kNone;
    }

    void setOwner(uint32_t content, uint32_t owner)
    {
        for (size_t i = 0; i < m_owners.size(); ++i) {
            if (m_owners[i].first == content) {
                if (owner == kNone)
                    m_owners.removeAt(i);
                else
                    m_owners[i].second = owner;
                return;
            }
        }
        if (owner != kNone)
            m_owners.append({ content, owner });
    }

    JSObject* get(uint32_t slot) const
    {
        if (slot >= m_size)
            return nullptr;
        WeakImpl* impl = m_entries[slot];
        if (!impl || impl->state() != WeakImpl::Live)
            return nullptr;
        return asObject(impl->jsValue());
    }

    void put(uint32_t slot, JSObject* object)
    {
        if (slot >= m_size) {
            uint32_t size = std::max<uint32_t>(64, m_size);
            while (size <= slot)
                size *= 2;
            auto* entries = static_cast<WeakImpl**>(fastZeroedMalloc(size * sizeof(WeakImpl*)));
            if (m_size)
                memcpy(entries, m_entries, m_size * sizeof(WeakImpl*));
            fastFree(m_entries);
            m_entries = entries;
            m_size = size;
        }
        if (m_entries[slot])
            WeakSet::deallocate(m_entries[slot]);
        m_entries[slot] = WeakSet::allocate(object, this, reinterpret_cast<void*>(static_cast<uintptr_t>(slot)));
    }

    void drop(uint32_t slot, JSObject* only = nullptr)
    {
        if (slot >= m_size || !m_entries[slot])
            return;
        if (only && m_entries[slot]->jsValue() != JSValue(only))
            return;
        WeakSet::deallocate(m_entries[slot]);
        m_entries[slot] = nullptr;
    }

    void setTypes(const uint8_t* types, uint32_t length)
    {
        m_types = types;
        m_typesLength = length;
    }

    uint32_t takeCollected()
    {
        uint32_t n = m_collected;
        m_collected = 0;
        return n;
    }

    // The marking constraint: every marked wrapper's tree root is an opaque root.
    template<typename Visitor>
    void addRoots(Visitor& visitor)
    {
        for (uint32_t i = 0; i < m_size; ++i) {
            WeakImpl* impl = m_entries[i];
            if (!impl || impl->state() != WeakImpl::Live)
                continue;
            if (visitor.isMarked(impl->jsValue().asCell()))
                visitor.addOpaqueRoot(rootKey(i));
        }
    }

    bool isReachableFromOpaqueRoots(JSC::Handle<JSC::Unknown>, void* context, AbstractSlotVisitor& visitor, ASCIILiteral* reason) final
    {
        // Rust's link pages are only read while the world is stopped; the stopped fixpoint answers.
        if (!m_vm.heap.worldIsStopped())
            return false;
        if (reason) [[unlikely]]
            *reason = "Its weld-dom tree is reachable"_s;
        return visitor.containsOpaqueRoot(rootKey(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(context))));
    }

    void finalize(JSC::Handle<JSC::Unknown> handle, void* context) final
    {
        uint32_t slot = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(context));
        if (slot < m_size && m_entries[slot] == WeakImpl::asWeakImpl(handle.slot())) {
            WeakSet::deallocate(m_entries[slot]);
            m_entries[slot] = nullptr;
        }
        ++m_collected;
    }

    static constexpr ptrdiff_t offsetOfEntries() { return OBJECT_OFFSETOF(WeldDomTable, m_entries); }
    static constexpr ptrdiff_t offsetOfSize() { return OBJECT_OFFSETOF(WeldDomTable, m_size); }
    static constexpr ptrdiff_t offsetOfLinks() { return OBJECT_OFFSETOF(WeldDomTable, m_links); }
    static constexpr ptrdiff_t offsetOfTypes() { return OBJECT_OFFSETOF(WeldDomTable, m_types); }
    static constexpr ptrdiff_t offsetOfTypesLength() { return OBJECT_OFFSETOF(WeldDomTable, m_typesLength); }

    VM& m_vm;
    WeakImpl** m_entries { nullptr };
    uint32_t m_size { 0 };
    uint32_t m_collected { 0 };
    const WeldDomLinks* m_links;
    const uint8_t* m_types { nullptr };
    uint32_t m_typesLength { 0 };
    void (*m_release)(void*);
    void* m_releaseContext;
    uint32_t m_tag;
    JSWeldDocument* m_cell { nullptr };
    Vector<std::pair<uint32_t, uint32_t>> m_owners;
};

// The JS face of a table: web-front's mirror holds it; every wrapper keeps it alive.
class JSWeldDocument final : public JSDestructibleObject {
public:
    using Base = JSDestructibleObject;
    static constexpr unsigned StructureFlags = Base::StructureFlags;
    DECLARE_INFO;
    DECLARE_VISIT_CHILDREN;

    template<typename CellType, SubspaceAccess>
    static CompleteSubspace* subspaceFor(VM& vm) { return &vm.destructibleObjectSpace(); }

    static Structure* createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
    {
        return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
    }

    static JSWeldDocument* create(VM& vm, Structure* structure, std::unique_ptr<WeldDomTable> table, Structure* nodeStructure)
    {
        auto* cell = new (NotNull, allocateCell<JSWeldDocument>(vm)) JSWeldDocument(vm, structure, WTF::move(table));
        cell->finishCreation(vm);
        cell->m_nodeStructure.set(vm, cell, nodeStructure);
        cell->m_table->m_cell = cell;
        return cell;
    }

    static void destroy(JSCell* cell) { static_cast<JSWeldDocument*>(cell)->JSWeldDocument::~JSWeldDocument(); }

    WeldDomTable& table() { return *m_table; }

    std::unique_ptr<WeldDomTable> m_table;
    WriteBarrier<JSObject> m_maker;
    WriteBarrier<JSArrayBufferView> m_types;
    WriteBarrier<Structure> m_nodeStructure;

private:
    JSWeldDocument(VM& vm, Structure* structure, std::unique_ptr<WeldDomTable> table)
        : Base(vm, structure)
        , m_table(WTF::move(table))
    {
    }
};

template<typename Visitor>
void JSWeldDocument::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    auto* thisObject = uncheckedDowncast<JSWeldDocument>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
    visitor.append(thisObject->m_maker);
    visitor.append(thisObject->m_types);
    visitor.append(thisObject->m_nodeStructure);
}
DEFINE_VISIT_CHILDREN(JSWeldDocument);
const ClassInfo JSWeldDocument::s_info = { "WeldDocument"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSWeldDocument) };

// A node wrapper.
class JSWeldNode final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;
    static constexpr unsigned StructureFlags = Base::StructureFlags;
    DECLARE_INFO;
    DECLARE_VISIT_CHILDREN;

    template<typename CellType, SubspaceAccess>
    static CompleteSubspace* subspaceFor(VM& vm) { return &vm.cellSpace(); }

    static Structure* createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
    {
        return Structure::create(vm, globalObject, prototype, TypeInfo(WeldNodeType, StructureFlags), info());
    }

    static JSWeldNode* create(VM& vm, Structure* structure, WeldDomTable* table, uint32_t slot, uint32_t gen)
    {
        auto* node = new (NotNull, allocateCell<JSWeldNode>(vm)) JSWeldNode(vm, structure);
        node->finishCreation(vm);
        node->m_slot = slot;
        node->m_gen = gen;
        node->m_table = table;
        return node;
    }

    static Ref<Snippet> checkSubClassSnippet()
    {
        Ref<Snippet> snippet = Snippet::create();
        snippet->setGenerator([=](CCallHelpers& jit, SnippetParams& params) {
            CCallHelpers::JumpList failure;
            failure.append(jit.branchIfNotType(params[0].gpr(), WeldNodeType));
            return failure;
        });
        return snippet;
    }

    static constexpr ptrdiff_t offsetOfSlot() { return OBJECT_OFFSETOF(JSWeldNode, m_slot); }
    static constexpr ptrdiff_t offsetOfGen() { return OBJECT_OFFSETOF(JSWeldNode, m_gen); }
    static constexpr ptrdiff_t offsetOfTable() { return OBJECT_OFFSETOF(JSWeldNode, m_table); }

    uint32_t m_slot { kNone };
    uint32_t m_gen { 0 };
    WeldDomTable* m_table { nullptr };

private:
    JSWeldNode(VM& vm, Structure* structure)
        : Base(vm, structure)
    {
    }
};
static_assert(sizeof(JSWeldNode) == 32, "a node wrapper is 32 bytes (step 5.25 gate)");

template<typename Visitor>
void JSWeldNode::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    auto* thisObject = uncheckedDowncast<JSWeldNode>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
    if (WeldDomTable* table = thisObject->m_table)
        visitor.appendUnbarriered(table->m_cell);
}
DEFINE_VISIT_CHILDREN(JSWeldNode);
const ClassInfo JSWeldNode::s_info = { "WeldNode"_s, &Base::s_info, nullptr, &JSWeldNode::checkSubClassSnippet, CREATE_METHOD_TABLE(JSWeldNode) };

// Per-global: the structures the C ABI makes documents and nodes with.
class JSWeldDomBindings final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;
    static constexpr unsigned StructureFlags = Base::StructureFlags;
    DECLARE_INFO;
    DECLARE_VISIT_CHILDREN;

    template<typename CellType, SubspaceAccess>
    static CompleteSubspace* subspaceFor(VM& vm) { return &vm.cellSpace(); }

    static Structure* createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
    {
        return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
    }

    static JSWeldDomBindings* create(VM& vm, JSGlobalObject* globalObject)
    {
        auto* object = new (NotNull, allocateCell<JSWeldDomBindings>(vm)) JSWeldDomBindings(vm, createStructure(vm, globalObject, globalObject->objectPrototype()));
        object->finishCreation(vm);
        object->m_nodeStructure.set(vm, object, JSWeldNode::createStructure(vm, globalObject, globalObject->objectPrototype()));
        object->m_documentStructure.set(vm, object, JSWeldDocument::createStructure(vm, globalObject, jsNull()));
        return object;
    }

    WriteBarrier<Structure> m_nodeStructure;
    WriteBarrier<Structure> m_documentStructure;

private:
    JSWeldDomBindings(VM& vm, Structure* structure)
        : Base(vm, structure)
    {
    }
};

template<typename Visitor>
void JSWeldDomBindings::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    auto* thisObject = uncheckedDowncast<JSWeldDomBindings>(cell);
    Base::visitChildren(thisObject, visitor);
    visitor.append(thisObject->m_nodeStructure);
    visitor.append(thisObject->m_documentStructure);
}
DEFINE_VISIT_CHILDREN(JSWeldDomBindings);
const ClassInfo JSWeldDomBindings::s_info = { "WeldDomBindings"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSWeldDomBindings) };

// The tables of each VM, for the marking constraint (mutator thread only, or a stopped world).
struct VMTables {
    uint64_t vm;
    Vector<WeldDomTable*> tables;
};
Lock s_tablesLock;
Vector<VMTables>& allTables() WTF_REQUIRES_LOCK(s_tablesLock)
{
    static NeverDestroyed<Vector<VMTables>> tables;
    return tables.get();
}
std::atomic<uint32_t> s_nextTag { 1 };

template<typename Visitor>
void addAllRoots(VM& vm, Visitor& visitor)
{
    if (!vm.heap.worldIsStopped())
        return;
    Locker locker { s_tablesLock };
    for (auto& entry : allTables()) {
        if (entry.vm != vm.identifier().toUInt64())
            continue;
        for (WeldDomTable* table : entry.tables)
            table->addRoots(visitor);
    }
}

void registerTable(VM& vm, WeldDomTable* table)
{
    Locker locker { s_tablesLock };
    uint64_t id = vm.identifier().toUInt64();
    for (auto& entry : allTables()) {
        if (entry.vm == id) {
            entry.tables.append(table);
            return;
        }
    }
    allTables().append({ id, { table } });
    vm.heap.addMarkingConstraint(makeUnique<SimpleMarkingConstraint>(
        "Wd"_s, "Weld DOM trees"_s,
        MAKE_MARKING_CONSTRAINT_EXECUTOR_PAIR(([&vm](auto& visitor) { addAllRoots(vm, visitor); })),
        ConstraintVolatility::GreyedByMarking,
        ConstraintConcurrency::Sequential));
}

void unregisterTable(VM& vm, WeldDomTable* table)
{
    Locker locker { s_tablesLock };
    uint64_t id = vm.identifier().toUInt64();
    for (auto& entry : allTables()) {
        if (entry.vm == id) {
            entry.tables.removeFirst(table);
            return;
        }
    }
}

WeldDomTable::~WeldDomTable()
{
    unregisterTable(m_vm, this);
    destroyEntries();
}

// What `arm` set for the next `NodeBase` construction on this thread.
struct Armed {
    WeldDomTable* table { nullptr };
    uint32_t slot { 0 };
    uint32_t gen { 0 };
};
thread_local Armed s_armed;

// ---------------------------------------------------------------- the wrapper of a slot

// The wrapper of `slot`, made by web-front's maker when none is alive (null for kNone).
JSValue wrapSlot(JSGlobalObject* globalObject, WeldDomTable& table, uint32_t slot)
{
    if (slot == kNone)
        return jsNull();
    if (JSObject* wrapper = table.get(slot))
        return wrapper;
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    JSObject* maker = table.m_cell ? table.m_cell->m_maker.get() : nullptr;
    if (!maker)
        return jsNull();
    auto callData = JSC::getCallData(maker);
    MarkedArgumentBuffer args;
    args.append(jsNumber(slot));
    JSValue made = call(globalObject, maker, callData, jsUndefined(), args);
    RETURN_IF_EXCEPTION(scope, { });
    return made;
}

bool isElement(WeldDomTable& table, uint32_t slot) { return table.type(slot) == kElementType; }

// What each tree getter reads, in C++ (the interpreter's and the slow paths' route).
enum class Getter : uint8_t { ParentNode, FirstChild, LastChild, PreviousSibling, NextSibling, ParentElement, FirstElementChild, LastElementChild, PreviousElementSibling, NextElementSibling };

uint32_t readGetter(WeldDomTable& table, uint32_t slot, Getter getter)
{
    auto elementFrom = [&](uint32_t n, unsigned step) {
        while (n != kNone && !isElement(table, n))
            n = table.link(step, n);
        return n;
    };
    switch (getter) {
    case Getter::ParentNode: return table.link(Parent, slot);
    case Getter::FirstChild: return table.link(First, slot);
    case Getter::LastChild: return table.link(Last, slot);
    case Getter::PreviousSibling: return table.link(Prev, slot);
    case Getter::NextSibling: return table.link(Next, slot);
    case Getter::ParentElement: {
        uint32_t p = table.link(Parent, slot);
        return p != kNone && isElement(table, p) ? p : kNone;
    }
    case Getter::FirstElementChild: return elementFrom(table.link(First, slot), Next);
    case Getter::LastElementChild: return elementFrom(table.link(Last, slot), Prev);
    case Getter::PreviousElementSibling: return elementFrom(table.link(Prev, slot), Prev);
    case Getter::NextElementSibling: return elementFrom(table.link(Next, slot), Next);
    }
    return kNone;
}

template<Getter getter>
OperationReturnType<EncodedJSValue> JIT_OPERATION_ATTRIBUTES operationWeldTree(JSGlobalObject* globalObject, void* pointer)
{
    VM& vm = globalObject->vm();
    CallFrame* callFrame = DECLARE_CALL_FRAME(vm);
    JITOperationPrologueCallFrameTracer tracer(vm, callFrame);
    auto scope = DECLARE_THROW_SCOPE(vm);
    auto* node = static_cast<JSWeldNode*>(pointer);
    JSValue result = wrapSlot(globalObject, *node->m_table, readGetter(*node->m_table, node->m_slot, getter));
    OPERATION_RETURN(scope, JSValue::encode(result));
}

template<Getter getter>
JSC_DEFINE_CUSTOM_GETTER(weldTreeGetter, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    auto* node = dynamicDowncast<JSWeldNode>(JSValue::decode(thisValue));
    if (!node || !node->m_table) [[unlikely]]
        return throwVMTypeError(globalObject, scope, "Not a weld-dom node"_s);
    RELEASE_AND_RETURN(scope, JSValue::encode(wrapSlot(globalObject, *node->m_table, readGetter(*node->m_table, node->m_slot, getter))));
}

#if ENABLE(JIT)
using Jump = CCallHelpers::Jump;
using JumpList = CCallHelpers::JumpList;

// out = column `column` of slot `slot` (links in `links`): `slot` is a live node's or one its
// links name, so its page has an entry (WeldDomLinks). Clobbers `tmp`. 32-bit ops zero the upper
// halves, so indexes are clean.
void emitLink(CCallHelpers& jit, GPRReg links, GPRReg slot, unsigned column, GPRReg out, GPRReg tmp)
{
    jit.urshift32(slot, CCallHelpers::TrustedImm32(kPageShift), tmp);
    jit.loadPtr(CCallHelpers::Address(links, OBJECT_OFFSETOF(WeldDomLinks, pages)), out);
    jit.loadPtr(CCallHelpers::BaseIndex(out, tmp, CCallHelpers::TimesEight), out);
    jit.and32(CCallHelpers::TrustedImm32(kPage - 1), slot, tmp);
    jit.load32(CCallHelpers::BaseIndex(out, tmp, CCallHelpers::TimesFour, column * kPage * 4), out);
}

// result = the live wrapper of slot `slot` (table in `table`), else jump to `fail`.
void emitLookup(CCallHelpers& jit, GPRReg table, GPRReg slot, GPRReg result, JumpList& fail)
{
    fail.append(jit.branch32(CCallHelpers::AboveOrEqual, slot, CCallHelpers::Address(table, WeldDomTable::offsetOfSize())));
    jit.loadPtr(CCallHelpers::Address(table, WeldDomTable::offsetOfEntries()), result);
    jit.loadPtr(CCallHelpers::BaseIndex(result, slot, CCallHelpers::TimesEight), result);
    fail.append(jit.branchTestPtr(CCallHelpers::Zero, result));
    fail.append(jit.branchTestPtr(CCallHelpers::NonZero, CCallHelpers::Address(result, WeakImpl::offsetOfWeakHandleOwner()), CCallHelpers::TrustedImm32(WeakImpl::StateMask)));
    jit.loadPtr(CCallHelpers::Address(result, WeakImpl::offsetOfJSValue()), result);
}

// out = 1 when slot `slot` is an element; jumps to `fail` past the types. Clobbers `tmp`.
void emitIsElement(CCallHelpers& jit, GPRReg table, GPRReg slot, GPRReg tmp, JumpList& fail, JumpList& notElement)
{
    fail.append(jit.branch32(CCallHelpers::AboveOrEqual, slot, CCallHelpers::Address(table, WeldDomTable::offsetOfTypesLength())));
    jit.loadPtr(CCallHelpers::Address(table, WeldDomTable::offsetOfTypes()), tmp);
    jit.load8(CCallHelpers::BaseIndex(tmp, slot, CCallHelpers::TimesOne), tmp);
    notElement.append(jit.branch32(CCallHelpers::NotEqual, tmp, CCallHelpers::TrustedImm32(kElementType)));
}

// Tree reads read JS-written link pages and may run web-front's maker: they read and write
// everything, as a call does. The win is the inline path, not code motion.
constexpr DOMJIT::Effect treeEffect = DOMJIT::Effect::forReadWrite(DOMJIT::HeapRange::top(), DOMJIT::HeapRange::top());

template<Getter getter>
Ref<DOMJIT::CallDOMGetterSnippet> compileTreeGetter()
{
    Ref<DOMJIT::CallDOMGetterSnippet> snippet = DOMJIT::CallDOMGetterSnippet::create();
    snippet->numGPScratchRegisters = 4;
    snippet->requireGlobalObject = true;
    snippet->effect = treeEffect;
    snippet->setGenerator([=](CCallHelpers& jit, SnippetParams& params) {
        GPRReg result = params[0].gpr();
        GPRReg node = params[1].gpr();
        GPRReg globalObject = params[2].gpr();
        GPRReg slot = params.gpScratch(0);
        GPRReg table = params.gpScratch(1);
        GPRReg links = params.gpScratch(2);
        GPRReg target = params.gpScratch(3);

        JumpList slow;
        JumpList nullCases;
        jit.load32(CCallHelpers::Address(node, JSWeldNode::offsetOfSlot()), slot);
        jit.loadPtr(CCallHelpers::Address(node, JSWeldNode::offsetOfTable()), table);
        jit.loadPtr(CCallHelpers::Address(table, WeldDomTable::offsetOfLinks()), links);

        auto column = [](Getter g) -> unsigned {
            switch (g) {
            case Getter::ParentNode:
            case Getter::ParentElement: return Parent;
            case Getter::FirstChild:
            case Getter::FirstElementChild: return First;
            case Getter::LastChild:
            case Getter::LastElementChild: return Last;
            case Getter::PreviousSibling:
            case Getter::PreviousElementSibling: return Prev;
            case Getter::NextSibling:
            case Getter::NextElementSibling: return Next;
            }
            return Parent;
        };
        emitLink(jit, links, slot, column(getter), target, result);

        if constexpr (getter == Getter::ParentElement) {
            nullCases.append(jit.branch32(CCallHelpers::Equal, target, CCallHelpers::TrustedImm32(-1)));
            emitIsElement(jit, table, target, result, slow, nullCases);
        } else if constexpr (getter == Getter::FirstElementChild || getter == Getter::LastElementChild || getter == Getter::PreviousElementSibling || getter == Getter::NextElementSibling) {
            unsigned step = (getter == Getter::LastElementChild || getter == Getter::PreviousElementSibling) ? Prev : Next;
            auto loop = jit.label();
            nullCases.append(jit.branch32(CCallHelpers::Equal, target, CCallHelpers::TrustedImm32(-1)));
            JumpList notElement;
            emitIsElement(jit, table, target, result, slow, notElement);
            Jump found = jit.jump();
            notElement.link(&jit);
            jit.move(target, slot);
            emitLink(jit, links, slot, step, target, result);
            jit.jump().linkTo(loop, &jit);
            found.link(&jit);
        } else
            nullCases.append(jit.branch32(CCallHelpers::Equal, target, CCallHelpers::TrustedImm32(-1)));

        emitLookup(jit, table, target, result, slow);
        Jump done = jit.jump();
        nullCases.link(&jit);
        jit.moveValue(jsNull(), result);
        done.link(&jit);
        params.addSlowPathCall(slow, jit, operationWeldTree<getter>, result, globalObject, node);
        return JumpList();
    });
    return snippet;
}

// `kind`, not `getter`: a non-dependent base's member (GetterSetter::getter) hides a template
// parameter of the same name.
template<Getter kind>
class WeldTreeAttribute final : public DOMJIT::GetterSetter {
public:
    constexpr WeldTreeAttribute()
        : DOMJIT::GetterSetter(weldTreeGetter<kind>, &compileTreeGetter<kind>, SpecHeapTop)
    {
    }
};
const WeldTreeAttribute<Getter::ParentNode> parentNodeAttribute;
const WeldTreeAttribute<Getter::FirstChild> firstChildAttribute;
const WeldTreeAttribute<Getter::LastChild> lastChildAttribute;
const WeldTreeAttribute<Getter::PreviousSibling> previousSiblingAttribute;
const WeldTreeAttribute<Getter::NextSibling> nextSiblingAttribute;
const WeldTreeAttribute<Getter::ParentElement> parentElementAttribute;
const WeldTreeAttribute<Getter::FirstElementChild> firstElementChildAttribute;
const WeldTreeAttribute<Getter::LastElementChild> lastElementChildAttribute;
const WeldTreeAttribute<Getter::PreviousElementSibling> previousElementSiblingAttribute;
const WeldTreeAttribute<Getter::NextElementSibling> nextElementSiblingAttribute;

void defineAttribute(VM& vm, JSGlobalObject* globalObject, JSObject* prototype, ASCIILiteral name, const DOMJIT::GetterSetter* domJIT)
{
    Identifier id = Identifier::fromString(vm, name);
    // Replaces web-front's JS accessor of the same name.
    DeletePropertySlot deleteSlot;
    JSObject::deleteProperty(prototype, globalObject, id, deleteSlot);
    unsigned attributes = PropertyAttribute::CustomAccessor | PropertyAttribute::DontEnum | PropertyAttribute::ReadOnly;
    prototype->putDirectCustomAccessor(vm, id, DOMAttributeGetterSetter::create(vm, domJIT->getter(), nullptr, DOMAttributeAnnotation { JSWeldNode::info(), domJIT }), attributes);
}
#endif

// ---------------------------------------------------------------- functions web-front calls

JSWeldDocument* documentArg(CallFrame* callFrame, unsigned i) { return dynamicDowncast<JSWeldDocument>(callFrame->argument(i)); }
uint32_t slotArg(CallFrame* callFrame, unsigned i)
{
    JSValue value = callFrame->argument(i);
    if (value.isInt32())
        return static_cast<uint32_t>(value.asInt32());
    return value.isNumber() ? JSC::toUInt32(value.asNumber()) : 0;
}

JSC_DEFINE_HOST_FUNCTION(weldNodeBaseCall, (JSGlobalObject* globalObject, CallFrame*))
{
    auto scope = DECLARE_THROW_SCOPE(globalObject->vm());
    return throwVMTypeError(globalObject, scope, "NodeBase is a constructor"_s);
}

// `class EventTarget extends NodeBase`: a node wrapper when armed for a slot, else an ordinary object.
JSC_DEFINE_HOST_FUNCTION(weldNodeBaseConstruct, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    JSObject* newTarget = asObject(callFrame->newTarget());
    Armed armed = s_armed;
    s_armed = { };
    if (armed.table && armed.table->m_cell) {
        Structure* structure = InternalFunction::createSubclassStructure(globalObject, newTarget, armed.table->m_cell->m_nodeStructure.get());
        RETURN_IF_EXCEPTION(scope, { });
        JSWeldNode* node = JSWeldNode::create(vm, structure, armed.table, armed.slot, armed.gen);
        armed.table->put(armed.slot, node);
        return JSValue::encode(node);
    }
    Structure* structure = InternalFunction::createSubclassStructure(globalObject, newTarget, globalObject->objectStructureForObjectConstructor());
    RETURN_IF_EXCEPTION(scope, { });
    return JSValue::encode(constructEmptyObject(vm, structure));
}

// arm(doc, slot, gen): the next NodeBase construction on this thread makes slot's wrapper.
JSC_DEFINE_HOST_FUNCTION(weldArm, (JSGlobalObject*, CallFrame* callFrame))
{
    JSWeldDocument* doc = documentArg(callFrame, 0);
    uint32_t slot = slotArg(callFrame, 1);
    s_armed = doc && doc->table().hasSlot(slot) ? Armed { &doc->table(), slot, slotArg(callFrame, 2) } : Armed { };
    return JSValue::encode(jsUndefined());
}

// install(prototype): the tree getters, on a node class's prototype.
JSC_DEFINE_HOST_FUNCTION(weldInstall, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    JSValue value = callFrame->argument(0);
    if (!value.isObject())
        return throwVMTypeError(globalObject, scope, "install: not a prototype"_s);
    JSObject* prototype = asObject(value);
#if ENABLE(JIT)
    defineAttribute(vm, globalObject, prototype, "parentNode"_s, &parentNodeAttribute);
    defineAttribute(vm, globalObject, prototype, "firstChild"_s, &firstChildAttribute);
    defineAttribute(vm, globalObject, prototype, "lastChild"_s, &lastChildAttribute);
    defineAttribute(vm, globalObject, prototype, "previousSibling"_s, &previousSiblingAttribute);
    defineAttribute(vm, globalObject, prototype, "nextSibling"_s, &nextSiblingAttribute);
    defineAttribute(vm, globalObject, prototype, "parentElement"_s, &parentElementAttribute);
    defineAttribute(vm, globalObject, prototype, "firstElementChild"_s, &firstElementChildAttribute);
    defineAttribute(vm, globalObject, prototype, "lastElementChild"_s, &lastElementChildAttribute);
    defineAttribute(vm, globalObject, prototype, "previousElementSibling"_s, &previousElementSiblingAttribute);
    defineAttribute(vm, globalObject, prototype, "nextElementSibling"_s, &nextElementSiblingAttribute);
#else
    UNUSED_PARAM(prototype);
#endif
    RETURN_IF_EXCEPTION(scope, { });
    return JSValue::encode(jsUndefined());
}

// attach(doc, maker): the function that makes a slot's wrapper when a getter finds none.
JSC_DEFINE_HOST_FUNCTION(weldAttach, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    JSWeldDocument* doc = documentArg(callFrame, 0);
    if (!doc || !callFrame->argument(1).isCallable())
        return JSValue::encode(jsBoolean(false));
    doc->m_maker.set(vm, doc, asObject(callFrame->argument(1)));
    return JSValue::encode(jsBoolean(true));
}

// types(doc, uint8Array): the node types (DOM nodeType codes) by slot; again after each growth.
JSC_DEFINE_HOST_FUNCTION(weldTypes, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    JSWeldDocument* doc = documentArg(callFrame, 0);
    auto* array = dynamicDowncast<JSUint8Array>(callFrame->argument(1));
    if (!doc || !array)
        return JSValue::encode(jsBoolean(false));
    // A view's bytes move once, when a fast typed array is given an ArrayBuffer: do it now.
    array->possiblySharedBuffer();
    doc->m_types.set(vm, doc, array);
    doc->table().setTypes(array->typedVector(), static_cast<uint32_t>(array->length()));
    return JSValue::encode(jsBoolean(true));
}

// get(doc, slot): slot's live wrapper, or undefined.
JSC_DEFINE_HOST_FUNCTION(weldGet, (JSGlobalObject*, CallFrame* callFrame))
{
    JSWeldDocument* doc = documentArg(callFrame, 0);
    JSObject* wrapper = doc ? doc->table().get(slotArg(callFrame, 1)) : nullptr;
    return JSValue::encode(wrapper ? JSValue(wrapper) : jsUndefined());
}

// put(doc, slot, object): slot's wrapper is `object` (a document, or a shim node that joined).
JSC_DEFINE_HOST_FUNCTION(weldPut, (JSGlobalObject*, CallFrame* callFrame))
{
    JSWeldDocument* doc = documentArg(callFrame, 0);
    if (doc && callFrame->argument(2).isObject())
        doc->table().put(slotArg(callFrame, 1), asObject(callFrame->argument(2)));
    return JSValue::encode(jsUndefined());
}

// drop(doc, slot): slot has no wrapper (it moved to another document).
JSC_DEFINE_HOST_FUNCTION(weldDrop, (JSGlobalObject*, CallFrame* callFrame))
{
    if (JSWeldDocument* doc = documentArg(callFrame, 0))
        doc->table().drop(slotArg(callFrame, 1));
    return JSValue::encode(jsUndefined());
}

// rebind(wrapper, doc, slot, gen): the wrapper is now slot's in doc (adoption keeps identity).
JSC_DEFINE_HOST_FUNCTION(weldRebind, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    JSValue value = callFrame->argument(0);
    JSWeldDocument* doc = documentArg(callFrame, 1);
    uint32_t slot = slotArg(callFrame, 2);
    if (!value.isObject() || !doc || !doc->table().hasSlot(slot))
        return JSValue::encode(jsBoolean(false));
    if (auto* node = dynamicDowncast<JSWeldNode>(value)) {
        if (node->m_table)
            node->m_table->drop(node->m_slot, node);
        node->m_table = &doc->table();
        node->m_slot = slot;
        node->m_gen = slotArg(callFrame, 3);
        vm.writeBarrier(node);
    }
    doc->table().put(slot, asObject(value));
    return JSValue::encode(jsBoolean(true));
}

// anyLive(doc, int32Array, count): whether any of the first `count` slots has a live wrapper.
JSC_DEFINE_HOST_FUNCTION(weldAnyLive, (JSGlobalObject*, CallFrame* callFrame))
{
    JSWeldDocument* doc = documentArg(callFrame, 0);
    auto* array = dynamicDowncast<JSInt32Array>(callFrame->argument(1));
    if (!doc || !array)
        return JSValue::encode(jsBoolean(false));
    uint32_t count = std::min<uint32_t>(slotArg(callFrame, 2), static_cast<uint32_t>(array->length()));
    const int32_t* slots = array->typedVector();
    for (uint32_t k = 0; k < count; ++k) {
        if (doc->table().get(static_cast<uint32_t>(slots[k])))
            return JSValue::encode(jsBoolean(true));
    }
    return JSValue::encode(jsBoolean(false));
}

// collected(doc): wrappers collected since the last call.
JSC_DEFINE_HOST_FUNCTION(weldCollected, (JSGlobalObject*, CallFrame* callFrame))
{
    JSWeldDocument* doc = documentArg(callFrame, 0);
    return JSValue::encode(jsNumber(doc ? doc->table().takeCollected() : 0));
}

// owner(doc, content, template): a template's contents fragment roots in its template's tree (-1: none).
JSC_DEFINE_HOST_FUNCTION(weldOwner, (JSGlobalObject*, CallFrame* callFrame))
{
    if (JSWeldDocument* doc = documentArg(callFrame, 0))
        doc->table().setOwner(slotArg(callFrame, 1), slotArg(callFrame, 2));
    return JSValue::encode(jsUndefined());
}

// cellSize(): bytes of a node wrapper without expandos.
JSC_DEFINE_HOST_FUNCTION(weldCellSize, (JSGlobalObject*, CallFrame*))
{
    return JSValue::encode(jsNumber(sizeof(JSWeldNode)));
}

// isNative(object): whether `object` is a node wrapper cell.
JSC_DEFINE_HOST_FUNCTION(weldIsNative, (JSGlobalObject*, CallFrame* callFrame))
{
    return JSValue::encode(jsBoolean(!!dynamicDowncast<JSWeldNode>(callFrame->argument(0))));
}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

} // namespace

// The bindings object web-front reads as `__weldDom.native`: one per global object.
extern "C" __attribute__((visibility("default"))) JSValueRef weld_dom_bindings(JSContextRef ctx)
{
    JSGlobalObject* globalObject = toJS(ctx);
    VM& vm = globalObject->vm();
    JSLockHolder lock(vm);
    auto* bindings = JSWeldDomBindings::create(vm, globalObject);
    auto function = [&](ASCIILiteral name, unsigned length, NativeFunction native) {
        bindings->putDirect(vm, Identifier::fromString(vm, name), JSFunction::create(vm, globalObject, length, name, native, ImplementationVisibility::Public));
    };
    JSFunction* nodeBase = JSFunction::create(vm, globalObject, 0, "NodeBase"_s, weldNodeBaseCall, ImplementationVisibility::Public, NoIntrinsic, weldNodeBaseConstruct);
    JSObject* nodeBasePrototype = constructEmptyObject(globalObject);
    nodeBasePrototype->putDirect(vm, vm.propertyNames->constructor, nodeBase, static_cast<unsigned>(PropertyAttribute::DontEnum));
    nodeBase->putDirect(vm, vm.propertyNames->prototype, nodeBasePrototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
    bindings->putDirect(vm, Identifier::fromString(vm, "NodeBase"_s), nodeBase);
    function("arm"_s, 3, weldArm);
    function("install"_s, 1, weldInstall);
    function("attach"_s, 2, weldAttach);
    function("types"_s, 2, weldTypes);
    function("get"_s, 2, weldGet);
    function("put"_s, 3, weldPut);
    function("drop"_s, 2, weldDrop);
    function("rebind"_s, 4, weldRebind);
    function("anyLive"_s, 3, weldAnyLive);
    function("collected"_s, 1, weldCollected);
    function("owner"_s, 3, weldOwner);
    function("cellSize"_s, 0, weldCellSize);
    function("isNative"_s, 1, weldIsNative);
    return toRef(globalObject, bindings);
}

// A document's wrapper table over weld-dom's link pages `links` (Rust keeps them at that address
// while `release(releaseContext)` has not been called; the table calls it when it dies).
extern "C" __attribute__((visibility("default"))) JSValueRef weld_dom_document(JSContextRef ctx, JSValueRef bindingsRef, const WeldDomLinks* links, void (*release)(void*), void* releaseContext)
{
    JSGlobalObject* globalObject = toJS(ctx);
    VM& vm = globalObject->vm();
    JSLockHolder lock(vm);
    auto* bindings = dynamicDowncast<JSWeldDomBindings>(toJS(globalObject, bindingsRef));
    if (!bindings || !links) {
        if (release)
            release(releaseContext);
        return nullptr;
    }
    auto table = makeUnique<WeldDomTable>(vm, links, release, releaseContext, s_nextTag.fetch_add(1) & 0x7fffffffu);
    WeldDomTable* raw = table.get();
    auto* doc = JSWeldDocument::create(vm, bindings->m_documentStructure.get(), WTF::move(table), bindings->m_nodeStructure.get());
    registerTable(vm, raw);
    return toRef(globalObject, doc);
}
