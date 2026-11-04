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
const objects = [new Error("e"), new Boolean(true), new Number(1), new ArrayBuffer(8), new WeakRef({}), new Map, new Date(0), [], () => 0, { }, new (class { })];
for (let i = 0; i < 1e5; ++i) {
    for (const object of objects) {
        shouldBe(strictThis.call(object), object, "strict");
        shouldBe(sloppyThis.call(object), object, "sloppy");
    }
}

// Non-objects fail the check and take the generic conversion.
shouldBe(strictThis.call(5), 5, "strict number");
shouldBe(strictThis.call(undefined), undefined, "strict undefined");
shouldBe(strictThis.call("s"), "s", "strict string");
shouldBe(typeof sloppyThis.call(5), "object", "sloppy number");
shouldBe(sloppyThis.call(undefined), globalThis, "sloppy undefined");
shouldBe(sloppyThis.call(null), globalThis, "sloppy null");
for (let i = 0; i < 1e5; ++i) {
    for (const object of objects)
        shouldBe(strictThis.call(object), object, "strict after exits");
}
