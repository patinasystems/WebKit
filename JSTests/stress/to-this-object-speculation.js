//@ requireOptions("--useDollarVM=true")
function shouldBe(actual, expected, what) {
    if (actual !== expected)
        throw new Error(what + ": " + String(actual) + " !== " + String(expected));
}

function strictThis() { "use strict"; return this; }
function sloppyThis() { return this; }
noInline(strictThis);
noInline(sloppyThis);

// Receivers of several object types, most of them SpecObjectOther: an object prediction that is
// neither a final object nor a string object, so ToThis folds with ObjectUse and a scope check.
// $vm's DOMJIT node has an embedder JSType (LastJSCObjectType + 1), as JSWeldNode has.
const embedder = typeof $vm === "undefined" ? [] : [$vm.createDOMJITNodeObject(), $vm.createDOMJITNodeObject()];
const objects = [...embedder, new Error("e"), new Boolean(true), new Number(1), new ArrayBuffer(8), new WeakRef({}), new Map, new Date(0), [], () => 0, { }, new (class { })];
for (let i = 0; i < 1e5; ++i) {
    for (const object of objects) {
        shouldBe(strictThis.call(object), object, "strict");
        shouldBe(sloppyThis.call(object), object, "sloppy");
    }
}

// globalThis is a JSGlobalProxy, not a JSScope: its ToThis is the identity, so it folds too, in
// both modes, alone and among other objects.
for (let i = 0; i < 1e5; ++i) {
    shouldBe(strictThis.call(globalThis), globalThis, "strict global proxy");
    shouldBe(sloppyThis.call(globalThis), globalThis, "sloppy global proxy");
    const object = objects[i % objects.length];
    shouldBe(strictThis.call(i & 1 ? globalThis : object), i & 1 ? globalThis : object, "strict mixed");
    shouldBe(sloppyThis.call(i & 1 ? globalThis : object), i & 1 ? globalThis : object, "sloppy mixed");
}
function globalProxyStrict() { "use strict"; return (function () { return this; }).call(globalThis); }
function globalProxySloppy() { return (function () { return this; })(); }
for (let i = 0; i < 1e5; ++i) {
    shouldBe(globalProxyStrict(), globalThis, "strict call on globalThis");
    shouldBe(globalProxySloppy(), globalThis, "sloppy unqualified call");
}

// Non-objects fail the check and take the generic conversion.
shouldBe(strictThis.call(5), 5, "strict number");
shouldBe(strictThis.call(undefined), undefined, "strict undefined");
shouldBe(strictThis.call("s"), "s", "strict string");
shouldBe(typeof sloppyThis.call(5), "object", "sloppy number");
shouldBe(sloppyThis.call(undefined), globalThis, "sloppy undefined");
shouldBe(sloppyThis.call(null), globalThis, "sloppy null");
for (let i = 0; i < 1e5; ++i) {
    for (const object of objects) {
        shouldBe(strictThis.call(object), object, "strict after exits");
        shouldBe(sloppyThis.call(object), object, "sloppy after exits");
    }
    if (!(i & 1023)) {
        shouldBe(strictThis.call(i), i, "strict number after exits");
        shouldBe(typeof sloppyThis.call(i), "object", "sloppy number after exits");
        shouldBe(sloppyThis.call(undefined), globalThis, "sloppy undefined after exits");
    }
}
