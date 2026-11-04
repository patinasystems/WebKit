// A captured parameter that every scope writes with the same value is folded to that value
// (SymbolTableEntry::inferredValue()); any other write must undo that everywhere.
function shouldBe(actual, expected, message) {
    if (actual !== expected && !(actual !== actual && expected !== expected))
        throw new Error(message + ": expected " + expected + " but got " + actual);
}

// 1. Many scopes with the same value, then one with another.
function make(W) {
    function get(x) { return (x + W) % W; }
    return function step(x) { return get(x) + W; };
}
var fns = [];
for (var i = 0; i < 2000; ++i)
    fns.push(make(32));
for (var k = 0; k < 200; ++k) {
    for (var i = 0; i < fns.length; ++i)
        shouldBe(fns[i](i), (i % 32) + 32, "same value");
}
var other = make(10);
for (var k = 0; k < 2000; ++k) {
    shouldBe(other(k), (k % 10) + 10, "other value");
    shouldBe(fns[k % fns.length](k), (k % 32) + 32, "same value after other");
}

// 2. Reassigned in its own function, to the same value and to another.
function makeReassign(W, v) {
    W = v;
    return function () { return W; };
}
var r = [];
for (var i = 0; i < 2000; ++i)
    r.push(makeReassign(7, 7));
for (var k = 0; k < 100; ++k) {
    for (var i = 0; i < r.length; ++i)
        shouldBe(r[i](), 7, "reassigned same");
}
var r2 = makeReassign(7, 8);
for (var k = 0; k < 2000; ++k) {
    shouldBe(r2(), 8, "reassigned other");
    shouldBe(r[k % r.length](), 7, "reassigned same after other");
}

// 3. Written from an inner function.
function makeInner(W) {
    return { get: function () { return W; }, set: function (v) { W = v; } };
}
var o = [];
for (var i = 0; i < 2000; ++i)
    o.push(makeInner(3));
for (var k = 0; k < 100; ++k) {
    for (var i = 0; i < o.length; ++i)
        shouldBe(o[i].get(), 3, "inner before");
}
o[5].set(4);
for (var k = 0; k < 1000; ++k) {
    shouldBe(o[5].get(), 4, "inner after");
    shouldBe(o[6].get(), 3, "inner other");
}

// 4. -0 and 0, NaN, doubles, booleans, undefined.
function makeAny(W) { return function () { return W; }; }
var values = [0, -0, NaN, 1.5, true, undefined, null, "s"];
for (var v = 0; v < values.length; ++v) {
    var f = [];
    for (var i = 0; i < 1000; ++i)
        f.push(makeAny(values[v]));
    for (var k = 0; k < 100; ++k) {
        for (var i = 0; i < f.length; ++i)
            shouldBe(Object.is(f[i](), values[v]), true, "value " + v);
    }
    var g = makeAny(values[(v + 1) % values.length]);
    for (var k = 0; k < 1000; ++k)
        shouldBe(Object.is(g(), values[(v + 1) % values.length]), true, "next value " + v);
}

// 5. Mapped arguments (sloppy, `arguments` used) alias the parameter.
function makeArgs(W) {
    var f = function () { return W; };
    arguments[0] = W + 1;
    return f;
}
var mapped = (function (W) { arguments[0] = W + 1; return W; })(9); // 10 when sloppy, 9 when strict
for (var i = 0; i < 3000; ++i)
    shouldBe(makeArgs(9)(), mapped, "mapped arguments");

// 6. The outer function hot enough to be compiled, then called with another value.
function makeHot(W) { return function () { return W * 2; }; }
var h = [];
for (var i = 0; i < 100000; ++i) {
    var fn = makeHot(5);
    if (i % 100 === 0)
        h.push(fn);
}
for (var k = 0; k < 100; ++k) {
    for (var i = 0; i < h.length; ++i)
        shouldBe(h[i](), 10, "hot");
}
var hot2 = makeHot(6);
for (var k = 0; k < 1000; ++k) {
    shouldBe(hot2(), 12, "hot other");
    shouldBe(h[k % h.length](), 10, "hot same after other");
}
