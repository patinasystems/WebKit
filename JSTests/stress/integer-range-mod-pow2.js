// x % 2^k becomes x & (2^k - 1) only when x is proven non-negative; the result keeps its range.
function shouldBe(actual, expected, label) {
    if (!Object.is(actual, expected))
        throw new Error(label + ": expected " + expected + " but got " + actual);
}

function wrap(g, x, y) {
    var n = 0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++)
            n += g[((y + dy + 32) % 32) * 32 + ((x + dx + 32) % 32)];
    }
    return n;
}
noInline(wrap);

function signedMod(a) {
    var r = 0;
    for (var i = 0; i < 8; i++)
        r += (a + i) % 4;
    return r;
}
noInline(signedMod);

function negativeZero(a) {
    return (a % 8) * -1;
}
noInline(negativeZero);

var g = new Int32Array(1024);
for (var i = 0; i < 1024; i++)
    g[i] = i;

// JS % by its definition (sign of the dividend), so the reference shares no code with wrap's %.
function jsMod(a, b) { return a - b * Math.trunc(a / b); }
function wrapRef(x, y) {
    var n = 0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++)
            n += g[jsMod(y + dy + 32, 32) * 32 + jsMod(x + dx + 32, 32)];
    }
    return n;
}

for (var iter = 0; iter < testLoopCount; iter++) {
    for (var y = 0; y < 32; y += 7) {
        for (var x = 0; x < 32; x += 5)
            shouldBe(wrap(g, x, y), wrapRef(x, y), "wrap " + x + "," + y);
    }
    shouldBe(signedMod(iter & 15), 12, "signedMod");
    shouldBe(signedMod(-9), -12, "signedMod negative");
    shouldBe(negativeZero(16), -0, "negativeZero");
}
// Inputs outside the speculated range must still compute JS semantics.
shouldBe(wrap(g, -40, -3), wrapRef(-40, -3), "wrap negative");
