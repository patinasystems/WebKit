// Weld glue (patinasystems/WebKit, API/Weld*): host classes defined from a member spec, over a C ABI weld-jsc calls
// (Kitchen lib/Weld/crates/jsc/src/host_class.rs; header there and in this file's `extern "C"` block).
//
// For the classes Rust implements for script: CanvasRenderingContext2D, Path2D, CanvasGradient, CanvasPattern, and the
// binding-table generator's DOM attribute classes. An instance is a JSC cell carrying (table, slot, gen): Rust owns the
// object at `slot`; `gen` lets Rust tell a stale cell from the slot's current one.
//
// - JSWeldHostObject: an instance. JSDestructibleObject plus its class's data (refcounted), its triple, and an owner
//   value it keeps alive (a 2D context keeps its <canvas> element's wrapper). Its destructor (the sweeper, on the
//   mutator thread) queues the triple on its class's data; Rust drains the queue on its own schedule
//   (weld_host_take_collected) and frees the slot. Rust is never called from the collector.
// - Members: JSNativeStdFunctions, so they are JSFunctions (the JITs call them directly, no C API callback shim). The
//   stub checks `this`, passes the arguments in place (the call frame's own slots when there are at least `arity`) and
//   calls the member's Rust fn pointer. On 64-bit a JSValueRef is the encoded JSValue, so nothing is converted.
// - JSWeldHostClass: one per spec and realm. It holds the instance structure, the prototype, the constructor, the
//   string cache and, for classes that ask, a flat weak table by slot (weld_host_lookup).
// - JSWeldHostConstructor: the class's constructor function (InternalFunction): `new Path2D()` calls the spec's
//   `construct`, which allocates the slot in Rust.
// - weld_host_bytes: a typed array's or ArrayBuffer's bytes in place.
//
// 64-bit only (CPU(ADDRESS64)), as weld-jsc is.
#include "config.h"

#include "APICast.h"
#include "CallFrameInlines.h"
#include "Error.h"
#include "GetterSetter.h"
#include "InternalFunction.h"
#include "JSArrayBuffer.h"
#include "JSArrayBufferViewInlines.h"
#include "JSCInlines.h"
#include "JSDestructibleObject.h"
#include "JSNativeStdFunction.h"
#include "ObjectConstructor.h"
#include "Weak.h"
#include "WeakInlines.h"
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/text/CString.h>
#include <wtf/text/MakeString.h>

#if !CPU(ADDRESS64)
#error "WeldHostClass passes JSValueRef as the encoded JSValue (64-bit only)"
#endif

using namespace JSC;

extern "C" {

typedef struct WeldHostThis {
    void* table;
    uint32_t slot;
    uint32_t gen;
} WeldHostThis;

typedef struct WeldHostCall {
    JSContextRef ctx;
    JSValueRef cls;
    JSValueRef thisValue;
    WeldHostThis self;
    const JSValueRef* args;
    uint32_t argc;
    uint32_t reserved;
} WeldHostCall;

typedef JSValueRef (*WeldHostFn)(const WeldHostCall*);
typedef int (*WeldHostConstruct)(const WeldHostCall*, WeldHostThis*, JSValueRef* owner);

enum { WeldHostMethod = 0, WeldHostGetter = 1, WeldHostSetter = 2, WeldHostConstant = 3 };
enum { WeldHostEnumerable = 1 << 0, WeldHostPure = 1 << 1 };
enum { WeldHostClassLookup = 1 << 0 };
enum { WeldHostTypeError = 0, WeldHostRangeError = 1, WeldHostDOMException = 2, WeldHostError = 3 };

typedef struct WeldHostMember {
    const char* name;
    uint8_t kind;
    uint8_t arity;
    uint16_t flags;
    uint32_t reserved;
    WeldHostFn fn;
    JSValueRef constant;
} WeldHostMember;

typedef struct WeldHostClassSpec {
    const char* name;
    const WeldHostMember* members;
    uint32_t count;
    uint32_t flags;
    WeldHostConstruct construct;
    uint32_t constructArity;
    uint32_t reserved;
    JSValueRef parent;
} WeldHostClassSpec;

typedef struct WeldStringSpan {
    const void* chars;
    uint32_t length;
    uint32_t latin1;
} WeldStringSpan;

}

namespace {

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

// The non-GC part of a class: what an instance's destructor and the member stubs need, alive as long as any of them.
struct WeldHostData final : ThreadSafeRefCounted<WeldHostData> {
    String name;
    RefPtr<WeldHostData> parent;
    Lock lock;
    Vector<WeldHostThis> collected WTF_GUARDED_BY_LOCK(lock);

    bool isOrInherits(const WeldHostData* other) const
    {
        for (const WeldHostData* data = this; data; data = data->parent.get()) {
            if (data == other)
                return true;
        }
        return false;
    }
};

class JSWeldHostClass;

class JSWeldHostObject final : public JSDestructibleObject {
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

    static JSWeldHostObject* create(VM& vm, Structure* structure, JSWeldHostClass* cls, WeldHostData* data, WeldHostThis self, JSValue owner)
    {
        auto* object = new (NotNull, allocateCell<JSWeldHostObject>(vm)) JSWeldHostObject(vm, structure, data, self);
        object->finishCreation(vm);
        object->m_class.setMayBeNull(vm, object, cls);
        if (owner && !owner.isUndefinedOrNull())
            object->m_owner.set(vm, object, owner);
        return object;
    }

    static void destroy(JSCell* cell) { static_cast<JSWeldHostObject*>(cell)->JSWeldHostObject::~JSWeldHostObject(); }

    ~JSWeldHostObject()
    {
        {
            Locker locker { m_data->lock };
            m_data->collected.append(m_self);
        }
        m_data->deref();
    }

    WeldHostData* data() const { return m_data; }
    const WeldHostThis& self() const { return m_self; }
    JSWeldHostClass* hostClass() const { return m_class.get(); }

private:
    JSWeldHostObject(VM& vm, Structure* structure, WeldHostData* data, WeldHostThis self)
        : Base(vm, structure)
        , m_data(data)
        , m_self(self)
    {
        m_data->ref();
    }

    WeldHostData* m_data;
    WeldHostThis m_self;
    WriteBarrier<JSWeldHostClass> m_class;
    WriteBarrier<Unknown> m_owner;
};

// One spec in one realm.
class JSWeldHostClass final : public JSDestructibleObject {
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

    static JSWeldHostClass* create(VM& vm, JSGlobalObject* globalObject, Ref<WeldHostData>&& data, WeldHostConstruct construct, bool lookup)
    {
        // The structure first: allocating it can collect, and a collection must not find this cell before its
        // constructor gives it a structure (C++ allocates the cell before it evaluates the constructor's arguments).
        Structure* structure = createStructure(vm, globalObject, jsNull());
        auto* cls = new (NotNull, allocateCell<JSWeldHostClass>(vm)) JSWeldHostClass(vm, structure, WTF::move(data), construct, lookup);
        cls->finishCreation(vm);
        return cls;
    }

    static void destroy(JSCell* cell) { static_cast<JSWeldHostClass*>(cell)->JSWeldHostClass::~JSWeldHostClass(); }

    WeldHostData& data() { return m_data.get(); }
    WeldHostConstruct construct() const { return m_construct; }
    bool lookup() const { return m_lookup; }

    // The cached string for `key`, else null.
    JSString* cachedString(uint64_t key)
    {
        auto it = m_strings.find(key);
        return it == m_strings.end() ? nullptr : it->value.get();
    }
    void cacheString(VM& vm, uint64_t key, JSString* string)
    {
        Locker locker { cellLock() };
        // Bounded: a cache that outgrows its use starts over (Rust clears it on its own boundaries as well).
        if (m_strings.size() >= maxCachedStrings)
            m_strings.clear();
        m_strings.add(key, WriteBarrier<JSString>(vm, this, string));
    }
    void clearStrings()
    {
        Locker locker { cellLock() };
        m_strings.clear();
    }

    JSObject* liveInstance(uint32_t slot, uint32_t gen)
    {
        if (slot >= m_instances.size())
            return nullptr;
        JSObject* live = m_instances[slot].get();
        if (!live)
            return nullptr;
        auto* object = uncheckedDowncast<JSWeldHostObject>(live);
        return object->self().gen == gen ? object : nullptr;
    }
    void recordInstance(VM& vm, uint32_t slot, JSWeldHostObject* object)
    {
        if (slot >= m_instances.size()) {
            size_t size = std::max<size_t>(slot + 1, m_instances.size() * 2);
            m_instances.grow(std::max<size_t>(size, 16));
        }
        m_instances[slot] = Weak<JSObject>(vm, object);
    }

    static constexpr unsigned maxCachedStrings = 4096;

    WriteBarrier<Structure> m_instanceStructure;
    WriteBarrier<JSObject> m_prototype;
    WriteBarrier<JSObject> m_constructor;
    WriteBarrier<JSWeldHostClass> m_parent;

private:
    JSWeldHostClass(VM& vm, Structure* structure, Ref<WeldHostData>&& data, WeldHostConstruct construct, bool lookup)
        : Base(vm, structure)
        , m_data(WTF::move(data))
        , m_construct(construct)
        , m_lookup(lookup)
    {
    }

    Ref<WeldHostData> m_data;
    WeldHostConstruct m_construct;
    bool m_lookup;
    HashMap<uint64_t, WriteBarrier<JSString>, WTF::IntHash<uint64_t>, WTF::UnsignedWithZeroKeyHashTraits<uint64_t>> m_strings;
    Vector<Weak<JSObject>> m_instances;
};

JSC_DECLARE_HOST_FUNCTION(weldHostConstructorCall);
JSC_DECLARE_HOST_FUNCTION(weldHostConstructorConstruct);

// The constructor: `new Class(...)` calls the spec's `construct`; a plain call is a TypeError.
class JSWeldHostConstructor final : public InternalFunction {
public:
    using Base = InternalFunction;
    static constexpr unsigned StructureFlags = Base::StructureFlags;
    DECLARE_INFO;
    DECLARE_VISIT_CHILDREN;

    template<typename CellType, SubspaceAccess>
    static CompleteSubspace* subspaceFor(VM& vm) { return &vm.cellSpace(); }

    static Structure* createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
    {
        return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
    }

    static JSWeldHostConstructor* create(VM& vm, JSGlobalObject* globalObject, JSValue functionPrototype, JSWeldHostClass* cls, unsigned length, const String& name)
    {
        Structure* structure = createStructure(vm, globalObject, functionPrototype);
        auto* function = new (NotNull, allocateCell<JSWeldHostConstructor>(vm)) JSWeldHostConstructor(vm, structure);
        function->finishCreation(vm, length, name);
        function->m_class.set(vm, function, cls);
        return function;
    }

    JSWeldHostClass* hostClass() const { return m_class.get(); }

private:
    JSWeldHostConstructor(VM& vm, Structure* structure)
        : Base(vm, structure, weldHostConstructorCall, weldHostConstructorConstruct)
    {
    }

    WriteBarrier<JSWeldHostClass> m_class;
};

template<typename Visitor>
void JSWeldHostObject::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    auto* thisObject = uncheckedDowncast<JSWeldHostObject>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
    visitor.append(thisObject->m_class);
    visitor.append(thisObject->m_owner);
}
DEFINE_VISIT_CHILDREN(JSWeldHostObject);
const ClassInfo JSWeldHostObject::s_info = { "WeldHostObject"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSWeldHostObject) };

template<typename Visitor>
void JSWeldHostClass::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    auto* thisObject = uncheckedDowncast<JSWeldHostClass>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
    visitor.append(thisObject->m_instanceStructure);
    visitor.append(thisObject->m_prototype);
    visitor.append(thisObject->m_constructor);
    visitor.append(thisObject->m_parent);
    Locker locker { thisObject->cellLock() };
    for (auto& entry : thisObject->m_strings)
        visitor.append(entry.value);
}
DEFINE_VISIT_CHILDREN(JSWeldHostClass);
const ClassInfo JSWeldHostClass::s_info = { "WeldHostClass"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSWeldHostClass) };

template<typename Visitor>
void JSWeldHostConstructor::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    auto* thisObject = uncheckedDowncast<JSWeldHostConstructor>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
    visitor.append(thisObject->m_class);
}
DEFINE_VISIT_CHILDREN(JSWeldHostConstructor);
const ClassInfo JSWeldHostConstructor::s_info = { "Function"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSWeldHostConstructor) };

// Arguments as the Rust entry reads them: at least `arity` values, missing trailing ones undefined. The call frame's
// own slots when it has enough (no copy), else a copy padded with undefined.
class Arguments {
public:
    Arguments(CallFrame* callFrame, unsigned arity)
    {
        size_t count = callFrame->argumentCount();
        if (count >= arity) {
            m_values = reinterpret_cast<const JSValueRef*>(callFrame->addressOfArgumentsStart());
            m_count = static_cast<uint32_t>(count);
            return;
        }
        JSValueRef* values = arity <= inlineCount ? m_inline.data() : (m_heap.resize(arity), m_heap.data());
        for (size_t i = 0; i < arity; ++i)
            values[i] = toRefWithoutGlobalObject(i < count ? callFrame->uncheckedArgument(i) : jsUndefined());
        m_values = values;
        m_count = arity;
    }

    const JSValueRef* values() const { return m_values; }
    uint32_t count() const { return m_count; }

private:
    static constexpr unsigned inlineCount = 16;
    std::array<JSValueRef, inlineCount> m_inline;
    Vector<JSValueRef> m_heap;
    const JSValueRef* m_values { nullptr };
    uint32_t m_count { 0 };
};

enum class MemberKind : uint8_t { Method, Getter, Setter };

// A member's stub: checks `this`, calls `fn` with the arguments in place, returns its result (null: it threw).
NativeStdFunction memberStub(Ref<WeldHostData>&& data, WeldHostFn fn, unsigned arity, MemberKind kind, String name)
{
    return [data = WTF::move(data), fn, arity, kind, name = WTF::move(name)](JSGlobalObject* globalObject, CallFrame* callFrame) -> EncodedJSValue {
        VM& vm = globalObject->vm();
        auto scope = DECLARE_THROW_SCOPE(vm);
        JSValue thisValue = callFrame->thisValue();
        auto* self = dynamicDowncast<JSWeldHostObject>(thisValue);
        if (!self || !self->data()->isOrInherits(data.ptr())) [[unlikely]]
            return throwVMTypeError(globalObject, scope, makeString("Can only call "_s, data->name, '.', name, " on instances of "_s, data->name));
        Arguments args(callFrame, kind == MemberKind::Setter ? 1 : arity);
        WeldHostCall call {
            toRef(globalObject),
            toRefWithoutGlobalObject(JSValue(self->hostClass())),
            toRefWithoutGlobalObject(thisValue),
            self->self(),
            args.values(),
            args.count(),
            0,
        };
        JSValueRef result = fn(&call);
        RETURN_IF_EXCEPTION(scope, { });
        ensureStillAliveHere(self);
        if (kind == MemberKind::Setter || !result)
            return JSValue::encode(jsUndefined());
        return JSValue::encode(toJS(result));
    };
}

Identifier memberIdentifier(VM& vm, const char* name)
{
    if (!strcmp(name, "@@iterator"))
        return vm.propertyNames->iteratorSymbol;
    if (!strcmp(name, "@@asyncIterator"))
        return vm.propertyNames->asyncIteratorSymbol;
    if (!strcmp(name, "@@toStringTag"))
        return vm.propertyNames->toStringTagSymbol;
    return Identifier::fromString(vm, String::fromUTF8(name));
}

JSC_DEFINE_HOST_FUNCTION(weldHostConstructorCall, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    auto scope = DECLARE_THROW_SCOPE(globalObject->vm());
    auto* function = uncheckedDowncast<JSWeldHostConstructor>(callFrame->jsCallee());
    return throwVMTypeError(globalObject, scope, makeString("Failed to construct '"_s, function->hostClass()->data().name, "': Please use the 'new' operator"_s));
}

JSC_DEFINE_HOST_FUNCTION(weldHostConstructorConstruct, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    auto* function = uncheckedDowncast<JSWeldHostConstructor>(callFrame->jsCallee());
    JSWeldHostClass* cls = function->hostClass();
    WeldHostConstruct construct = cls->construct();
    if (!construct)
        return throwVMTypeError(globalObject, scope, "Illegal constructor"_s);
    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* structure = newTarget == function ? cls->m_instanceStructure.get() : InternalFunction::createSubclassStructure(globalObject, newTarget, cls->m_instanceStructure.get());
    RETURN_IF_EXCEPTION(scope, { });
    Arguments args(callFrame, 0);
    WeldHostCall call { toRef(globalObject), toRefWithoutGlobalObject(JSValue(cls)), nullptr, { }, args.values(), args.count(), 0 };
    WeldHostThis self { };
    JSValueRef owner = nullptr;
    int made = construct(&call, &self, &owner);
    RETURN_IF_EXCEPTION(scope, { });
    if (!made)
        return throwVMTypeError(globalObject, scope, makeString("Failed to construct '"_s, cls->data().name, '\''));
    auto* object = JSWeldHostObject::create(vm, structure, cls, &cls->data(), self, owner ? toJS(owner) : JSValue());
    if (cls->lookup())
        cls->recordInstance(vm, self.slot, object);
    ensureStillAliveHere(cls);
    return JSValue::encode(object);
}

JSWeldHostClass* classArg(JSValueRef value)
{
    return value ? dynamicDowncast<JSWeldHostClass>(toJS(value)) : nullptr;
}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

} // namespace

#define WELD_EXPORT extern "C" __attribute__((visibility("default")))

// Defines `spec` in ctx's realm and returns its class handle (weld_host_constructor gives the constructor).
WELD_EXPORT JSValueRef weld_host_class(JSContextRef ctx, const WeldHostClassSpec* spec)
{
    JSGlobalObject* globalObject = toJS(ctx);
    VM& vm = globalObject->vm();
    JSLockHolder lock(vm);
    if (!spec || !spec->name)
        return nullptr;
    JSWeldHostClass* parent = classArg(spec->parent);
    if (spec->parent && !parent)
        return nullptr;

    Ref<WeldHostData> data = adoptRef(*new WeldHostData);
    data->name = String::fromUTF8(spec->name);
    if (parent)
        data->parent = &parent->data();
    String name = data->name;
    auto* cls = JSWeldHostClass::create(vm, globalObject, data.copyRef(), spec->construct, spec->flags & WeldHostClassLookup);
    if (parent)
        cls->m_parent.set(vm, cls, parent);

    JSObject* prototype = constructEmptyObject(globalObject, parent ? parent->m_prototype.get() : globalObject->objectPrototype());
    cls->m_prototype.set(vm, cls, prototype);
    cls->m_instanceStructure.set(vm, cls, JSWeldHostObject::createStructure(vm, globalObject, prototype));
    JSValue functionPrototype = parent ? JSValue(parent->m_constructor.get()) : JSValue(globalObject->functionPrototype());
    auto* constructor = JSWeldHostConstructor::create(vm, globalObject, functionPrototype, cls, spec->constructArity, name);
    cls->m_constructor.set(vm, cls, constructor);
    constructor->putDirect(vm, vm.propertyNames->prototype, prototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
    prototype->putDirect(vm, vm.propertyNames->constructor, constructor, static_cast<unsigned>(PropertyAttribute::DontEnum));
    prototype->putDirect(vm, vm.propertyNames->toStringTagSymbol, jsString(vm, name), PropertyAttribute::DontEnum | PropertyAttribute::ReadOnly);

    // Getters first so a setter of the same name joins its accessor.
    HashMap<String, std::pair<JSObject*, JSObject*>> accessors;
    Vector<std::pair<Identifier, unsigned>> accessorOrder;
    MarkedArgumentBuffer keepAlive;
    for (uint32_t i = 0; i < spec->count; ++i) {
        const WeldHostMember& member = spec->members[i];
        if (!member.name)
            continue;
        Identifier id = memberIdentifier(vm, member.name);
        unsigned enumerable = (member.flags & WeldHostEnumerable) ? 0 : static_cast<unsigned>(PropertyAttribute::DontEnum);
        switch (member.kind) {
        case WeldHostMethod: {
            if (!member.fn)
                break;
            auto* function = JSNativeStdFunction::create(vm, globalObject, member.arity, id.isSymbol() ? makeString('[', String(id.impl()), ']') : id.string(), memberStub(data.copyRef(), member.fn, member.arity, MemberKind::Method, id.string()));
            prototype->putDirect(vm, id, function, enumerable);
            break;
        }
        case WeldHostGetter:
        case WeldHostSetter: {
            if (!member.fn)
                break;
            bool getter = member.kind == WeldHostGetter;
            auto* function = JSNativeStdFunction::create(vm, globalObject, getter ? 0 : 1, makeString(getter ? "get "_s : "set "_s, id.string()), memberStub(data.copyRef(), member.fn, 0, getter ? MemberKind::Getter : MemberKind::Setter, id.string()));
            keepAlive.append(function);
            auto result = accessors.add(id.string(), std::pair<JSObject*, JSObject*> { nullptr, nullptr });
            if (result.isNewEntry)
                accessorOrder.append({ id, enumerable });
            (getter ? result.iterator->value.first : result.iterator->value.second) = function;
            break;
        }
        case WeldHostConstant: {
            JSValue value = member.constant ? toJS(member.constant) : jsUndefined();
            if (value.isCell())
                break; // numbers, booleans and null only: the spec outlives no cell
            unsigned attributes = enumerable | PropertyAttribute::ReadOnly | PropertyAttribute::DontDelete;
            prototype->putDirect(vm, id, value, attributes);
            constructor->putDirect(vm, id, value, attributes);
            break;
        }
        default:
            break;
        }
    }
    for (auto& [id, enumerable] : accessorOrder) {
        auto pair = accessors.get(id.string());
        auto* getterSetter = GetterSetter::create(vm, globalObject, pair.first ? JSValue(pair.first) : jsUndefined(), pair.second ? JSValue(pair.second) : jsUndefined());
        prototype->putDirectAccessor(globalObject, id, getterSetter, PropertyAttribute::Accessor | enumerable);
    }
    return toRef(globalObject, cls);
}

// The class's constructor function (put it where the interface goes).
WELD_EXPORT JSObjectRef weld_host_constructor(JSContextRef ctx, JSValueRef clsRef)
{
    JSGlobalObject* globalObject = toJS(ctx);
    JSLockHolder lock(globalObject->vm());
    JSWeldHostClass* cls = classArg(clsRef);
    return cls ? toRef(cls->m_constructor.get()) : nullptr;
}

// The class's prototype object.
WELD_EXPORT JSObjectRef weld_host_prototype(JSContextRef ctx, JSValueRef clsRef)
{
    JSGlobalObject* globalObject = toJS(ctx);
    JSLockHolder lock(globalObject->vm());
    JSWeldHostClass* cls = classArg(clsRef);
    return cls ? toRef(cls->m_prototype.get()) : nullptr;
}

// A new instance of `cls` for `self`; it keeps `owner` (when not null or undefined) alive.
WELD_EXPORT JSObjectRef weld_host_new(JSContextRef ctx, JSValueRef clsRef, WeldHostThis self, JSValueRef owner)
{
    JSGlobalObject* globalObject = toJS(ctx);
    VM& vm = globalObject->vm();
    JSLockHolder lock(vm);
    JSWeldHostClass* cls = classArg(clsRef);
    if (!cls)
        return nullptr;
    auto* object = JSWeldHostObject::create(vm, cls->m_instanceStructure.get(), cls, &cls->data(), self, owner ? toJS(owner) : JSValue());
    if (cls->lookup())
        cls->recordInstance(vm, self.slot, object);
    return toRef(object);
}

// The live instance of a lookup class (WeldHostClassLookup) for (slot, gen), else null.
WELD_EXPORT JSObjectRef weld_host_lookup(JSContextRef ctx, JSValueRef clsRef, uint32_t slot, uint32_t gen)
{
    JSGlobalObject* globalObject = toJS(ctx);
    JSLockHolder lock(globalObject->vm());
    JSWeldHostClass* cls = classArg(clsRef);
    return cls ? toRef(cls->liveInstance(slot, gen)) : nullptr;
}

// `value`'s triple when it is an instance of `cls` or of a class inheriting from it.
WELD_EXPORT int weld_host_this(JSContextRef ctx, JSValueRef clsRef, JSValueRef value, WeldHostThis* out)
{
    JSGlobalObject* globalObject = toJS(ctx);
    JSLockHolder lock(globalObject->vm());
    JSWeldHostClass* cls = classArg(clsRef);
    auto* object = value ? dynamicDowncast<JSWeldHostObject>(toJS(value)) : nullptr;
    if (!cls || !object || !object->data()->isOrInherits(&cls->data()))
        return 0;
    if (out)
        *out = object->self();
    return 1;
}

// Moves up to `capacity` collected instances' triples of `cls` (its own instances, not subclasses') to `out`.
WELD_EXPORT size_t weld_host_take_collected(JSContextRef ctx, JSValueRef clsRef, WeldHostThis* out, size_t capacity)
{
    JSGlobalObject* globalObject = toJS(ctx);
    JSLockHolder lock(globalObject->vm());
    JSWeldHostClass* cls = classArg(clsRef);
    if (!cls || !out)
        return 0;
    WeldHostData& data = cls->data();
    Locker locker { data.lock };
    size_t count = std::min(capacity, data.collected.size());
    size_t start = data.collected.size() - count;
    for (size_t i = 0; i < count; ++i)
        out[i] = data.collected[start + i];
    data.collected.shrink(start);
    return count;
}

// Throws `kind` with `message` (UTF-8). DOMException: `message` is "Name: text"; it becomes an Error whose `name` is
// Name. Returns null, the value a member returns to throw.
WELD_EXPORT JSValueRef weld_host_throw(JSContextRef ctx, int kind, const char* message, size_t length)
{
    JSGlobalObject* globalObject = toJS(ctx);
    VM& vm = globalObject->vm();
    JSLockHolder lock(vm);
    auto scope = DECLARE_THROW_SCOPE(vm);
    String text = String::fromUTF8(std::span { reinterpret_cast<const char8_t*>(message), length });
    JSObject* error = nullptr;
    switch (kind) {
    case WeldHostTypeError:
        error = createTypeError(globalObject, text);
        break;
    case WeldHostRangeError:
        error = createRangeError(globalObject, text);
        break;
    case WeldHostDOMException: {
        size_t colon = text.find(':');
        String name = colon == notFound ? "Error"_s : text.left(colon);
        String rest = colon == notFound ? text : text.substring(colon + 1);
        if (rest.startsWith(' '))
            rest = rest.substring(1);
        error = createError(globalObject, rest);
        error->putDirect(vm, vm.propertyNames->name, jsString(vm, name), static_cast<unsigned>(PropertyAttribute::DontEnum));
        break;
    }
    default:
        error = createError(globalObject, text);
        break;
    }
    throwException(globalObject, scope, error);
    return nullptr;
}

// `value` as its characters in place (a rope is resolved once). Valid while `value` is (a member's argument: until it
// returns). 0 when `value` is not a string.
WELD_EXPORT int weld_host_string_span(JSContextRef ctx, JSValueRef value, WeldStringSpan* out)
{
    JSGlobalObject* globalObject = toJS(ctx);
    VM& vm = globalObject->vm();
    JSLockHolder lock(vm);
    JSValue v = value ? toJS(value) : JSValue();
    if (!v || !v.isString() || !out)
        return 0;
    auto scope = DECLARE_TOP_EXCEPTION_SCOPE(vm);
    // A copy of the String (a reference, not the characters): the JSString keeps the same storage, resolved.
    String string = v.toWTFString(globalObject);
    if (scope.exception()) [[unlikely]] {
        scope.clearException();
        return 0;
    }
    StringImpl* impl = string.impl();
    if (!impl) {
        *out = { "", 0, 1 };
        return 1;
    }
    if (impl->is8Bit())
        *out = { impl->span8().data(), impl->length(), 1 };
    else
        *out = { impl->span16().data(), impl->length(), 0 };
    return 1;
}

// ToString(value); null with the exception pending when it throws.
WELD_EXPORT JSValueRef weld_host_to_string(JSContextRef ctx, JSValueRef value)
{
    JSGlobalObject* globalObject = toJS(ctx);
    VM& vm = globalObject->vm();
    JSLockHolder lock(vm);
    auto scope = DECLARE_THROW_SCOPE(vm);
    JSString* string = (value ? toJS(value) : jsUndefined()).toString(globalObject);
    RETURN_IF_EXCEPTION(scope, nullptr);
    return toRef(globalObject, string);
}

// The cached string of `cls` for `key`; on a miss, makes it from `latin1` and caches it.
WELD_EXPORT JSValueRef weld_host_cached_string(JSContextRef ctx, JSValueRef clsRef, uint64_t key, const uint8_t* latin1, size_t length)
{
    JSGlobalObject* globalObject = toJS(ctx);
    VM& vm = globalObject->vm();
    JSLockHolder lock(vm);
    JSWeldHostClass* cls = classArg(clsRef);
    if (!cls)
        return nullptr;
    if (JSString* string = cls->cachedString(key))
        return toRef(globalObject, string);
    JSString* string = jsString(vm, String(std::span { latin1, length }));
    cls->cacheString(vm, key, string);
    return toRef(globalObject, string);
}

WELD_EXPORT void weld_host_string_cache_clear(JSContextRef ctx, JSValueRef clsRef)
{
    JSGlobalObject* globalObject = toJS(ctx);
    JSLockHolder lock(globalObject->vm());
    if (JSWeldHostClass* cls = classArg(clsRef))
        cls->clearStrings();
}

// A new string from Latin-1 bytes or UTF-16 units (copied).
WELD_EXPORT JSValueRef weld_host_string(JSContextRef ctx, const uint8_t* latin1, size_t length)
{
    JSGlobalObject* globalObject = toJS(ctx);
    VM& vm = globalObject->vm();
    JSLockHolder lock(vm);
    return toRef(globalObject, jsString(vm, String(std::span { latin1, length })));
}

WELD_EXPORT JSValueRef weld_host_string16(JSContextRef ctx, const uint16_t* chars, size_t length)
{
    JSGlobalObject* globalObject = toJS(ctx);
    VM& vm = globalObject->vm();
    JSLockHolder lock(vm);
    return toRef(globalObject, jsString(vm, String(std::span { reinterpret_cast<const char16_t*>(chars), length })));
}

// The bytes of a typed array, DataView or ArrayBuffer in place: valid until script next runs (it may detach or
// resize the buffer). 0 when `value` is none of these or is detached.
WELD_EXPORT int weld_host_bytes(JSContextRef ctx, JSValueRef value, uint8_t** ptr, size_t* length)
{
    JSGlobalObject* globalObject = toJS(ctx);
    JSLockHolder lock(globalObject->vm());
    JSValue v = value ? toJS(value) : JSValue();
    if (!v || !v.isCell())
        return 0;
    if (auto* view = dynamicDowncast<JSArrayBufferView>(v.asCell())) {
        if (view->isDetached() || view->isOutOfBounds())
            return 0;
        *ptr = static_cast<uint8_t*>(view->vector());
        *length = view->byteLength();
        return 1;
    }
    if (auto* buffer = dynamicDowncast<JSArrayBuffer>(v.asCell())) {
        ArrayBuffer* impl = buffer->impl();
        if (!impl || impl->isDetached())
            return 0;
        *ptr = static_cast<uint8_t*>(impl->data());
        *length = impl->byteLength();
        return 1;
    }
    return 0;
}
