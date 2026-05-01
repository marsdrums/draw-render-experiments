autowhatch = 1; inlets = 1; outlets = 1;

const m = require("Patcher://vector_math.js");

let TIME = 0;
let VIEWPORT = [1920, 1080];
let RATIO = VIEWPORT[0] / VIEWPORT[1];
let pos, at, farClip, nearClip, lensAngle;

const RADIX_BITS = 8;
const RADIX_SIZE = 1 << RADIX_BITS;   // 256
const SORT_LOCAL_SIZE = 256;

let PARTICLE_COUNT = 0;   // requested count
let ALLOC_COUNT = 1;      // storage/dispatch-safe count
let SORT_BLOCKS = 1;

let proxy_camera = new JitterObject("jit.proxy");

// particle buffer: Pos(vec3), rad(float), col(vec4);
let buff_particles = new JitterObject("jit.gpu.buffer");

let buff_quad = new JitterObject("jit.gpu.buffer");
let mat_quad = new JitterMatrix(4, "float32", 4);
mat_quad.setcell(0, "val", [-1, -1, 0, 0]);
mat_quad.setcell(1, "val", [+1, -1, 0, 0]);
mat_quad.setcell(2, "val", [-1, +1, 0, 0]);
mat_quad.setcell(3, "val", [+1, +1, 0, 0]);
buff_quad.jit_matrix(mat_quad.name);

let buff_inside_counter = new JitterObject("jit.gpu.buffer"); // indirect draw args / visible count
buff_inside_counter.bytecount = 4 * 4;

let buff_inside = new JitterObject("jit.gpu.buffer"); // per-particle visibility flag
let buff_visible_indices = new JitterObject("jit.gpu.buffer");

// radix-sort temp buffers
let buff_sort_keys = new JitterObject("jit.gpu.buffer");
let buff_sort_keys_tmp = new JitterObject("jit.gpu.buffer");
let buff_visible_indices_tmp = new JitterObject("jit.gpu.buffer");

let buff_radix_hist = new JitterObject("jit.gpu.buffer");
let buff_radix_offsets = new JitterObject("jit.gpu.buffer");
let buff_radix_digit_base = new JitterObject("jit.gpu.buffer");

let comp_reset_inside_counter = new JitterObject("jit.gpu.compute");
comp_reset_inside_counter.file = "comp_reset_inside_counter.comp";
comp_reset_inside_counter.workgroups = [1, 1, 1];
comp_reset_inside_counter.bind("buff_inside_counter", buff_inside_counter.name);

let comp_generate_position = new JitterObject("jit.gpu.compute");
comp_generate_position.file = "comp_generate_position.comp";
comp_generate_position.bind("buff_particles", buff_particles.name);

let comp_transform_position = new JitterObject("jit.gpu.compute");
comp_transform_position.file = "comp_transform_position.comp";
comp_transform_position.bind("buff_particles", buff_particles.name);

// compute frustum culling
let comp_inside_frustum = new JitterObject("jit.gpu.compute");
comp_inside_frustum.file = "comp_inside_frustum.comp";
comp_inside_frustum.bind("buff_particles", buff_particles.name);
comp_inside_frustum.bind("buff_inside", buff_inside.name);

// compact the visible particles
let comp_compact_inside = new JitterObject("jit.gpu.compute");
comp_compact_inside.file = "comp_compact_inside.comp";
comp_compact_inside.bind("buff_inside", buff_inside.name);
comp_compact_inside.bind("buff_visible_indices", buff_visible_indices.name);
comp_compact_inside.bind("buff_inside_counter", buff_inside_counter.name);

// build radix keys from visible particles' view-space depth
let comp_build_sort_keys = new JitterObject("jit.gpu.compute");
comp_build_sort_keys.file = "comp_build_sort_keys.comp";
comp_build_sort_keys.bind("buff_particles", buff_particles.name);
comp_build_sort_keys.bind("buff_visible_indices", buff_visible_indices.name);
comp_build_sort_keys.bind("buff_sort_keys", buff_sort_keys.name);
comp_build_sort_keys.bind("buff_inside_counter", buff_inside_counter.name);

let comp_radix_clear = new JitterObject("jit.gpu.compute");
comp_radix_clear.file = "comp_radix_clear.comp";
comp_radix_clear.bind("buff_radix_hist", buff_radix_hist.name);
comp_radix_clear.bind("buff_radix_offsets", buff_radix_offsets.name);
comp_radix_clear.bind("buff_radix_digit_base", buff_radix_digit_base.name);

let comp_radix_histogram = new JitterObject("jit.gpu.compute");
comp_radix_histogram.file = "comp_radix_histogram.comp";
comp_radix_histogram.bind("buff_sort_keys", buff_sort_keys.name);
comp_radix_histogram.bind("buff_radix_hist", buff_radix_hist.name);
comp_radix_histogram.bind("buff_inside_counter", buff_inside_counter.name);

let comp_radix_scan = new JitterObject("jit.gpu.compute");
comp_radix_scan.file = "comp_radix_scan.comp";
comp_radix_scan.bind("buff_radix_hist", buff_radix_hist.name);
comp_radix_scan.bind("buff_radix_offsets", buff_radix_offsets.name);
comp_radix_scan.bind("buff_radix_digit_base", buff_radix_digit_base.name);

let comp_radix_scatter = new JitterObject("jit.gpu.compute");
comp_radix_scatter.file = "comp_radix_scatter.comp";
comp_radix_scatter.bind("buff_sort_keys_in", buff_sort_keys.name);
comp_radix_scatter.bind("buff_sort_keys_out", buff_sort_keys_tmp.name);
comp_radix_scatter.bind("buff_indices_in", buff_visible_indices.name);
comp_radix_scatter.bind("buff_indices_out", buff_visible_indices_tmp.name);
comp_radix_scatter.bind("buff_radix_offsets", buff_radix_offsets.name);
comp_radix_scatter.bind("buff_radix_digit_base", buff_radix_digit_base.name);
comp_radix_scatter.bind("buff_inside_counter", buff_inside_counter.name);

let img_color_target = new JitterObject("jit.gpu.image");
img_color_target.dim = [VIEWPORT[0], VIEWPORT[1]];
img_color_target.format = "rgba32_float";

let img_depth_target = new JitterObject("jit.gpu.image");
img_depth_target.dim = [VIEWPORT[0], VIEWPORT[1]];
img_depth_target.format = "d32_float";

let draw_particles = new JitterObject("jit.gpu.draw");
draw_particles.shader = "draw_particles.rend";
draw_particles.vb0 = buff_quad.name;
draw_particles.buff_particles = buff_particles.name;
draw_particles.buff_visible_indices = buff_visible_indices.name;
draw_particles.indirect = buff_inside_counter.name;
draw_particles.buff_inside_counter = buff_inside_counter.name;
draw_particles.topology = "trianglestrip";
draw_particles.elemcount = 4;
draw_particles.blendenable = true;
draw_particles.depth_write = false;

let render_particles = new JitterObject("jit.gpu.render");
render_particles.colorattachments = 1;
render_particles.depth = false;
render_particles.colorimg0 = img_color_target.name;
render_particles.clearcolor0 = [0, 0, 0, 1];
render_particles.depthimg = img_depth_target.name;
render_particles.msaa = 1;

count(1000000);

function camera_name(name) {
    proxy_camera.name = name;
}

// VECTOR MATH
function normalizeVec3(v) {
    const len = Math.hypot(v[0], v[1], v[2]);
    if (len === 0) return new Float32Array([0, 0, 0]);
    const inv = 1 / len;
    return new Float32Array([v[0] * inv, v[1] * inv, v[2] * inv]);
}
function mulVec3Float(a, b) { return [a[0] * b, a[1] * b, a[2] * b]; }
function sumVec3(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function subVec3(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }

function count(N) {

    PARTICLE_COUNT = Math.max(0, N | 0);
    ALLOC_COUNT = Math.max(1, PARTICLE_COUNT);
    SORT_BLOCKS = Math.max(1, Math.ceil(ALLOC_COUNT / SORT_LOCAL_SIZE));

    let SIZE_OF_FLOAT = 4;
    let SIZE_OF_UINT = 4;

    // set workgroups for existing passes
    comp_generate_position.workgroups = [Math.max(1, Math.ceil(ALLOC_COUNT / 256)), 1, 1];
    comp_transform_position.workgroups = [Math.max(1, Math.ceil(ALLOC_COUNT / 256)), 1, 1];
    comp_inside_frustum.workgroups = [Math.max(1, Math.ceil(ALLOC_COUNT / 256)), 1, 1];
    comp_compact_inside.workgroups = [Math.max(1, Math.ceil(ALLOC_COUNT / 256)), 1, 1];

    // set workgroups for radix sort
    comp_build_sort_keys.workgroups = [SORT_BLOCKS, 1, 1];
    comp_radix_clear.workgroups = [Math.max(1, Math.ceil((SORT_BLOCKS * RADIX_SIZE) / 256)), 1, 1];
    comp_radix_histogram.workgroups = [SORT_BLOCKS, 1, 1];
    comp_radix_scan.workgroups = [RADIX_SIZE, 1, 1];
    comp_radix_scatter.workgroups = [SORT_BLOCKS, 1, 1];

    // allocate dependencies
    buff_particles.bytecount = ALLOC_COUNT * SIZE_OF_FLOAT * 8;
    buff_inside.bytecount = ALLOC_COUNT * SIZE_OF_UINT;
    buff_visible_indices.bytecount = ALLOC_COUNT * SIZE_OF_UINT;

    buff_sort_keys.bytecount = ALLOC_COUNT * SIZE_OF_UINT;
    buff_sort_keys_tmp.bytecount = ALLOC_COUNT * SIZE_OF_UINT;
    buff_visible_indices_tmp.bytecount = ALLOC_COUNT * SIZE_OF_UINT;

    buff_radix_hist.bytecount = SORT_BLOCKS * RADIX_SIZE * SIZE_OF_UINT;
    buff_radix_offsets.bytecount = SORT_BLOCKS * RADIX_SIZE * SIZE_OF_UINT;
    buff_radix_digit_base.bytecount = RADIX_SIZE * SIZE_OF_UINT;

    // set params
    comp_generate_position.param("COUNT", PARTICLE_COUNT);
    comp_generate_position.param("SQRT_COUNT", Math.ceil(Math.sqrt(Math.max(1, PARTICLE_COUNT))));

    comp_transform_position.param("COUNT", PARTICLE_COUNT);
    comp_inside_frustum.param("COUNT", PARTICLE_COUNT);
    comp_compact_inside.param("COUNT", PARTICLE_COUNT);

    comp_build_sort_keys.param("COUNT", PARTICLE_COUNT);

    comp_radix_clear.param("BLOCK_COUNT", SORT_BLOCKS);

    comp_radix_histogram.param("COUNT", PARTICLE_COUNT);
    comp_radix_histogram.param("BLOCK_COUNT", SORT_BLOCKS);

    comp_radix_scan.param("BLOCK_COUNT", SORT_BLOCKS);

    comp_radix_scatter.param("COUNT", PARTICLE_COUNT);
    comp_radix_scatter.param("BLOCK_COUNT", SORT_BLOCKS);

    // set number of instances (actual upper bound; indirect draw uses visible count)
    draw_particles.instancecount = PARTICLE_COUNT;
}

function time(T) {
    TIME = T;
}

function calc_matrices() {

    pos = proxy_camera.send("getposition");
    at = proxy_camera.send("getlookat");
    farClip = proxy_camera.send("getfar_clip");
    nearClip = proxy_camera.send("getnear_clip");
    lensAngle = proxy_camera.send("getlens_angle");
    let up = [0, 1, 0];

    return {
        V: m.lookAt(pos, at, up),
        P: m.perspective(lensAngle, RATIO, nearClip, farClip)
    };
}

function calc_matrices_for_debug() {

    pos = [3, 3, 3];
    at = [0, 0, 0];
    farClip = proxy_camera.send("getfar_clip");
    nearClip = proxy_camera.send("getnear_clip");
    lensAngle = proxy_camera.send("getlens_angle");
    let up = [0, 1, 0];

    return {
        V: m.lookAt(pos, at, up),
        P: m.perspective(lensAngle, RATIO, nearClip, farClip)
    };
}

function calc_frustum() {
    const fovRad = lensAngle * Math.PI / 180.0;

    const halfVSide = farClip * Math.tan(fovRad * 0.5);
    const halfHSide = halfVSide * RATIO;

    const front = normalizeVec3(subVec3(at, pos));
    const right = normalizeVec3(cross(front, [0, 1, 0]));
    const up = normalizeVec3(cross(right, front));

    const frontMultNear = mulVec3Float(front, nearClip);
    const frontMultFar = mulVec3Float(front, farClip);

    comp_inside_frustum.param("Npos", sumVec3(pos, frontMultNear));
    comp_inside_frustum.param("Ndir", front);
    comp_inside_frustum.param("Fpos", sumVec3(pos, frontMultFar));
    comp_inside_frustum.param("Fdir", [-front[0], -front[1], -front[2]]);
    comp_inside_frustum.param("Rpos", pos);
    comp_inside_frustum.param("Rdir", normalizeVec3(cross(subVec3(frontMultFar, mulVec3Float(right, halfHSide)), up)));
    comp_inside_frustum.param("Lpos", pos);
    comp_inside_frustum.param("Ldir", normalizeVec3(cross(up, sumVec3(frontMultFar, mulVec3Float(right, halfHSide)))));
    comp_inside_frustum.param("Tpos", pos);
    comp_inside_frustum.param("Tdir", normalizeVec3(cross(right, subVec3(frontMultFar, mulVec3Float(up, halfVSide)))));
    comp_inside_frustum.param("Bpos", pos);
    comp_inside_frustum.param("Bdir", normalizeVec3(cross(sumVec3(frontMultFar, mulVec3Float(up, halfVSide)), right)));
}

function radix_sort_visible_front_to_back() {

    // input starts in main buffers; output starts in temp buffers
    let keyIn = buff_sort_keys;
    let keyOut = buff_sort_keys_tmp;
    let idxIn = buff_visible_indices;
    let idxOut = buff_visible_indices_tmp;

    for (let pass = 0; pass < 4; pass++) {
        let shift = pass * 8;

        comp_radix_clear.bang();

        comp_radix_histogram.bind("buff_sort_keys", keyIn.name);
        comp_radix_histogram.param("SHIFT", shift);
        comp_radix_histogram.bang();

        comp_radix_scan.bang();

        comp_radix_scatter.bind("buff_sort_keys_in", keyIn.name);
        comp_radix_scatter.bind("buff_sort_keys_out", keyOut.name);
        comp_radix_scatter.bind("buff_indices_in", idxIn.name);
        comp_radix_scatter.bind("buff_indices_out", idxOut.name);
        comp_radix_scatter.param("SHIFT", shift);
        comp_radix_scatter.bang();

        let tmpKey = keyIn; keyIn = keyOut; keyOut = tmpKey;
        let tmpIdx = idxIn; idxIn = idxOut; idxOut = tmpIdx;
    }

    // 4 passes is even, so final output is back in buff_visible_indices.
    // Explicitly restore bindings so the next frame always starts from the compacted buffer.
    draw_particles.buff_visible_indices = buff_visible_indices.name;
    comp_build_sort_keys.bind("buff_visible_indices", buff_visible_indices.name);
}

function bang() {

    comp_generate_position.param("TIME", TIME);
    comp_generate_position.param("SCALE", 0.03);
    comp_generate_position.bang();

    let transfrom = calc_matrices();
    calc_frustum();

    comp_reset_inside_counter.bang();
    comp_inside_frustum.bang();
    comp_compact_inside.bang();

    // Remove the comment for the debug view
    // transfrom = calc_matrices_for_debug();

    // transform positions into view space first
    comp_transform_position.param("V", transfrom.V);
    comp_transform_position.bang();

    // build depth keys from currently visible particles and sort them front-to-back
    //comp_build_sort_keys.param("NEAR_CLIP", nearClip);
    //comp_build_sort_keys.param("FAR_CLIP", farClip);
    //comp_build_sort_keys.bang();

    //radix_sort_visible_front_to_back();

    draw_particles.param("P", transfrom.P);
    render_particles.jit_gpu_draw(draw_particles.name);
    render_particles.bang();

    outlet(0, "source", img_color_target.name);
    outlet(0, "bang");
}
