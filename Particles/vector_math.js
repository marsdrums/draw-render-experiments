function mat4Identity() {
    return new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ]);
}

function lookAt(camPos, at, up) {
    const ex = camPos[0], ey = camPos[1], ez = camPos[2];
    const ax = at[0], ay = at[1], az = at[2];
    const ux = up[0], uy = up[1], uz = up[2];

    let fx = ex - ax, fy = ey - ay, fz = ez - az;
    let fl = Math.hypot(fx, fy, fz);
    fx /= fl; fy /= fl; fz /= fl;

    let rx = uy * fz - uz * fy;
    let ry = uz * fx - ux * fz;
    let rz = ux * fy - uy * fx;
    let rl = Math.hypot(rx, ry, rz);
    rx /= rl; ry /= rl; rz /= rl;

    const tx = fy * rz - fz * ry;
    const ty = fz * rx - fx * rz;
    const tz = fx * ry - fy * rx;

    return new Float32Array([
        rx, tx, fx, 0,
        ry, ty, fy, 0,
        rz, tz, fz, 0,
        -(rx * ex + ry * ey + rz * ez),
        -(tx * ex + ty * ey + tz * ez),
        -(fx * ex + fy * ey + fz * ez),
        1
    ]);
}

function perspective(fovyDeg, aspect, near, far) {
    const fovyRad = fovyDeg * Math.PI / 180;
    const f = 1.0 / Math.tan(fovyRad / 2);
    const nf = 1 / (near - far);

    return new Float32Array([
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, far * nf, -1,
        0, 0, near * far * nf, 0
    ]);
}

function ortho(left, right, bottom, top, near, far) {
    const lr = 1 / (left - right);
    const bt = 1 / (bottom - top);
    const nf = 1 / (near - far);

    return new Float32Array([
        -2 * lr, 0, 0, 0,
        0, -2 * bt, 0, 0,
        0, 0, nf, 0,
        (left + right) * lr, (top + bottom) * bt, near * nf, 1
    ]);
}

function normalMatrix4FromMat4(m) {
    const a00 = m[0], a01 = m[4], a02 = m[8];
    const a10 = m[1], a11 = m[5], a12 = m[9];
    const a20 = m[2], a21 = m[6], a22 = m[10];

    const b01 = a22 * a11 - a12 * a21;
    const b11 = -a22 * a10 + a12 * a20;
    const b21 = a21 * a10 - a11 * a20;

    let det = a00 * b01 + a01 * b11 + a02 * b21;
    if (!det) return null;
    det = 1.0 / det;

    return new Float32Array([
        b01 * det, (-a22 * a01 + a02 * a21) * det, (a12 * a01 - a02 * a11) * det, 0,
        b11 * det, (a22 * a00 - a02 * a20) * det, (-a12 * a00 + a02 * a10) * det, 0,
        b21 * det, (-a21 * a00 + a01 * a20) * det, (a11 * a00 - a01 * a10) * det, 0,
        0, 0, 0, 1
    ]);
}

function mulMat4(a, b) {
    const out = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
        for (let r = 0; r < 4; r++) {
            out[c * 4 + r] =
                a[0 * 4 + r] * b[c * 4 + 0] +
                a[1 * 4 + r] * b[c * 4 + 1] +
                a[2 * 4 + r] * b[c * 4 + 2] +
                a[3 * 4 + r] * b[c * 4 + 3];
        }
    }
    return out;
}

function normalizeVec3(v) {
    const len = Math.hypot(v[0], v[1], v[2]);
    if (len === 0) return new Float32Array([0, 0, 0]);
    const inv = 1 / len;
    return new Float32Array([v[0] * inv, v[1] * inv, v[2] * inv]);
}

function mat4FromTranslation(t) {
    return new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        t[0], t[1], t[2], 1
    ]);
}

function mat4FromUniformScale(s) {
    return new Float32Array([
        s[0], 0, 0, 0,
        0, s[1], 0, 0,
        0, 0, s[2], 0,
        0, 0, 0, 1
    ]);
}

function mat4FromAxisAngle(axis, angle) {
    const x = axis[0], y = axis[1], z = axis[2];
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    const t = 1 - c;

    return new Float32Array([
        t * x * x + c, t * x * y + s * z, t * x * z - s * y, 0,
        t * x * y - s * z, t * y * y + c, t * y * z + s * x, 0,
        t * x * z + s * y, t * y * z - s * x, t * z * z + c, 0,
        0, 0, 0, 1
    ]);
}

exports.mat4Identity = mat4Identity;
exports.lookAt = lookAt;
exports.perspective = perspective;
exports.ortho = ortho;
exports.normalMatrix4FromMat4 = normalMatrix4FromMat4;
exports.mulMat4 = mulMat4;
exports.normalizeVec3 = normalizeVec3;
exports.mat4FromTranslation = mat4FromTranslation;
exports.mat4FromUniformScale = mat4FromUniformScale;
exports.mat4FromAxisAngle = mat4FromAxisAngle;
