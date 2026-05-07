autowhatch = 1; inlets = 1; outlets = 1;

const m = require("Patcher://vector_math.js");

let TIME = 0;
let VIEWPORT = [1920, 1080];
let RATIO = VIEWPORT[0] / VIEWPORT[1];
let RADIUS = 0.003;
let ALPHA = 0.1;
let pos, at, farClip, nearClip, lensAngle, viewDir;
let sliceSize;
let particleCount;
let sliceCount = 256;
let scale = 0.0006;
let shadowMapSize = 1024;

// One-pass radix/counting sort settings.
// The sort key range is measured on the GPU every frame, so particle positions
// are no longer assumed to live in any fixed range such as [-1, 1].
const RADIX_BITS = 16;
const RADIX_BINS = 1 << RADIX_BITS;      // 65536 depth buckets
const RADIX_BLOCK_SIZE = 256;            // scan block size
const RADIX_BLOCK_COUNT = RADIX_BINS / RADIX_BLOCK_SIZE;

let lightDir = normalizeVec3([1,1,1]);
let halfVector, flipped;

let proxy_camera = new JitterObject("jit.proxy");

// particle buffer: Pos(vec3), rad(float), col(vec4), key(float), id(uint)
// The particle buffer remains in original/generated order. Sorting now produces
// a compact uint permutation buffer instead of moving full Particle structs.
let buff_particles = new JitterObject("jit.gpu.buffer");
let buff_sorted_indices = new JitterObject("jit.gpu.buffer");

// Per-particle data prepared once per slice and reused by the 4 quad vertices.
// Layout matches DrawParticle in the draw/prepare shaders: 3 vec4 = 48 bytes.
let buff_particle_draw = new JitterObject("jit.gpu.buffer");

let buff_quad = new JitterObject("jit.gpu.buffer");
let mat_quad = new JitterMatrix(4, "float32", 4);
mat_quad.setcell(0, "val", [-1, -1, 0, 0]);
mat_quad.setcell(1, "val", [+1, -1, 0, 0]);
mat_quad.setcell(2, "val", [-1, +1, 0, 0]);
mat_quad.setcell(3, "val", [+1, +1, 0, 0]);
buff_quad.jit_matrix(mat_quad.name);

let comp_generate_position = new JitterObject("jit.gpu.compute");
comp_generate_position.file = "comp_generate_position.comp";
comp_generate_position.bind("buff_particles", buff_particles.name);
comp_generate_position.param("RADIUS", RADIUS);

// -----------------------------------------------------------------------------
// Fast GPU radix/counting sort pipeline for arbitrary particle counts.
//
// This version sorts only particle indices. buff_particles stays in generated
// order, while buff_sorted_indices contains the current sorted permutation.
//
// Passes per frame:
//  1. clear histogram and scatter cursors
//  2. measure the actual min/max floating-point sort key
//  3. build 16-bit depth-key histogram
//  4. prefix-scan each 256-bin histogram block
//  5. prefix-scan the 256 block totals
//  6. add block offsets to per-bin offsets
//  7. scatter uint particle indices into buff_sorted_indices
// -----------------------------------------------------------------------------
let buff_radix_histogram = new JitterObject("jit.gpu.buffer");
let buff_radix_cursor = new JitterObject("jit.gpu.buffer");
let buff_radix_bin_offsets = new JitterObject("jit.gpu.buffer");
let buff_radix_block_sums = new JitterObject("jit.gpu.buffer");
let buff_radix_block_offsets = new JitterObject("jit.gpu.buffer");
let buff_sort_range = new JitterObject("jit.gpu.buffer"); // uint min/max of ordered float particle keys

let comp_radix_clear = new JitterObject("jit.gpu.compute");
comp_radix_clear.shader = "comp_radix_clear.comp";
comp_radix_clear.workgroups = [RADIX_BINS / 256, 1, 1];
comp_radix_clear.bind("buff_radix_histogram", buff_radix_histogram.name);
comp_radix_clear.bind("buff_radix_cursor", buff_radix_cursor.name);
comp_radix_clear.bind("buff_sort_range", buff_sort_range.name);

let comp_radix_find_key_range = new JitterObject("jit.gpu.compute");
comp_radix_find_key_range.shader = "comp_radix_find_key_range.comp";
comp_radix_find_key_range.bind("buff_particles", buff_particles.name);
comp_radix_find_key_range.bind("buff_sort_range", buff_sort_range.name);

let comp_radix_build_histogram = new JitterObject("jit.gpu.compute");
comp_radix_build_histogram.shader = "comp_radix_build_histogram.comp";
comp_radix_build_histogram.bind("buff_particles", buff_particles.name);
comp_radix_build_histogram.bind("buff_radix_histogram", buff_radix_histogram.name);
comp_radix_build_histogram.bind("buff_sort_range", buff_sort_range.name);

let comp_radix_scan_bins = new JitterObject("jit.gpu.compute");
comp_radix_scan_bins.shader = "comp_radix_scan_bins.comp";
comp_radix_scan_bins.workgroups = [RADIX_BLOCK_COUNT, 1, 1];
comp_radix_scan_bins.bind("buff_radix_histogram", buff_radix_histogram.name);
comp_radix_scan_bins.bind("buff_radix_bin_offsets", buff_radix_bin_offsets.name);
comp_radix_scan_bins.bind("buff_radix_block_sums", buff_radix_block_sums.name);

let comp_radix_scan_block_sums = new JitterObject("jit.gpu.compute");
comp_radix_scan_block_sums.shader = "comp_radix_scan_block_sums.comp";
comp_radix_scan_block_sums.workgroups = [1, 1, 1];
comp_radix_scan_block_sums.bind("buff_radix_block_sums", buff_radix_block_sums.name);
comp_radix_scan_block_sums.bind("buff_radix_block_offsets", buff_radix_block_offsets.name);

let comp_radix_add_block_offsets = new JitterObject("jit.gpu.compute");
comp_radix_add_block_offsets.shader = "comp_radix_add_block_offsets.comp";
comp_radix_add_block_offsets.workgroups = [RADIX_BLOCK_COUNT, 1, 1];
comp_radix_add_block_offsets.bind("buff_radix_bin_offsets", buff_radix_bin_offsets.name);
comp_radix_add_block_offsets.bind("buff_radix_block_offsets", buff_radix_block_offsets.name);

let comp_radix_scatter_particles = new JitterObject("jit.gpu.compute");
comp_radix_scatter_particles.shader = "comp_radix_scatter_particles.comp";
comp_radix_scatter_particles.bind("buff_particles", buff_particles.name);
comp_radix_scatter_particles.bind("buff_sorted_indices", buff_sorted_indices.name);
comp_radix_scatter_particles.bind("buff_radix_bin_offsets", buff_radix_bin_offsets.name);
comp_radix_scatter_particles.bind("buff_radix_cursor", buff_radix_cursor.name);
comp_radix_scatter_particles.bind("buff_sort_range", buff_sort_range.name);

let img_color_target = new JitterObject("jit.gpu.image");
img_color_target.dim = [VIEWPORT[0], VIEWPORT[1]];
img_color_target.format = "rgba32_float";

let img_density_map = new JitterObject("jit.gpu.image");
// 256-slice occupancy map stored as eight horizontal r32ui tiles.
// Tile 0 stores slices    0..31, tile 1 stores   32..63,
// tile 2 stores slices   64..95, tile 3 stores   96..127,
// tile 4 stores slices 128..159, tile 5 stores 160..191,
// tile 6 stores slices 192..223, tile 7 stores 224..255.
img_density_map.dim = [shadowMapSize * 8, shadowMapSize];
img_density_map.format = "r32_uint";

let comp_clear_color_target = new JitterObject("jit.gpu.compute");
comp_clear_color_target.shader = "comp_clear_color_target.comp";
comp_clear_color_target.bind("img_color_target", img_color_target.name);
comp_clear_color_target.workgroups = [Math.ceil(VIEWPORT[0] / 16), Math.ceil(VIEWPORT[1] / 16), 1];

let comp_clear_shadow_map = new JitterObject("jit.gpu.compute");
comp_clear_shadow_map.shader = "comp_clear_shadow_map.comp";
comp_clear_shadow_map.bind("img_density_map", img_density_map.name);
comp_clear_shadow_map.workgroups = [Math.ceil((shadowMapSize * 8) / 16), Math.ceil(shadowMapSize / 16), 1];

let comp_prepare_eye_particles = new JitterObject("jit.gpu.compute");
comp_prepare_eye_particles.shader = "comp_prepare_eye_particles.comp";
comp_prepare_eye_particles.bind("buff_particles", buff_particles.name);
comp_prepare_eye_particles.bind("buff_sorted_indices", buff_sorted_indices.name);
comp_prepare_eye_particles.bind("img_density_map", img_density_map.name);
comp_prepare_eye_particles.bind("buff_particle_draw", buff_particle_draw.name);

let comp_prepare_shadow_particles = new JitterObject("jit.gpu.compute");
comp_prepare_shadow_particles.shader = "comp_prepare_shadow_particles.comp";
comp_prepare_shadow_particles.bind("buff_particles", buff_particles.name);
comp_prepare_shadow_particles.bind("buff_sorted_indices", buff_sorted_indices.name);
comp_prepare_shadow_particles.bind("buff_particle_draw", buff_particle_draw.name);

// shadow pass

let comp_render_shadow = new JitterObject("jit.gpu.compute");
comp_render_shadow.shader = "comp_render_shadow.comp";
comp_render_shadow.bind("buff_particle_draw", buff_particle_draw.name);
comp_render_shadow.bind("img_density_map", img_density_map.name);
comp_render_shadow.param("ALPHA", ALPHA);

// main pass

let draw_particles = new JitterObject("jit.gpu.draw");
draw_particles.shader = "draw_particles.rend";
draw_particles.vb0 = buff_quad.name;
draw_particles.buff_particle_draw = buff_particle_draw.name;
draw_particles.topology = "trianglestrip";
draw_particles.elemcount = 4;
draw_particles.blendenable = true;
draw_particles.depth_write = false;
draw_particles.param("ALPHA", ALPHA);
draw_particles.blendcolorsrc = "inv_dst_alpha";
draw_particles.blendcolordst = "one";
draw_particles.blendalphasrc = "inv_dst_alpha";
draw_particles.blendalphadst = "one";

let render_particles = new JitterObject("jit.gpu.render");
render_particles.colorattachments = 1;
render_particles.depth = false;
render_particles.colorimg0 = img_color_target.name;
render_particles.colorloadop0 = "load";

let comp_composite_background = new JitterObject("jit.gpu.compute");
comp_composite_background.shader = "comp_composite_background.comp";
comp_composite_background.workgroups = [Math.ceil(VIEWPORT[0] / 16), Math.ceil(VIEWPORT[1] / 16), 1];
comp_composite_background.bind("img_color_target", img_color_target.name);

count(1000000);

function camera_name(name) {
    proxy_camera.name = name;
}

function setalpha(x){
    ALPHA = x;
    draw_particles.param("ALPHA", ALPHA);
    comp_render_shadow.param("ALPHA", ALPHA);
}

function setradius(x){
    RADIUS = x;
    comp_generate_position.param("RADIUS", RADIUS);
}

function setslice_count(x){
    sliceCount = Math.max(1, Math.min(256, x));
    count(particleCount);
}

function setscale(x){
    scale = x;
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
function dot(a, b){ return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }

function light_direction(){
    lightDir = normalizeVec3([arguments[0], arguments[1], arguments[2]]);
}

function compute_half_vector(){

    // The paper's prose describes the branch as positive vs. negative dot product.
    // Its pseudocode says > 1.0, but normalized vectors cannot exceed 1.0.

    if(dot(viewDir, lightDir) > 0.0) {
        halfVector = normalizeVec3(sumVec3(mulVec3Float(viewDir, -1.0), mulVec3Float(lightDir, +1.0)));
        halfVector = mulVec3Float(halfVector, -1.0);
        flipped = false;
    } else {
        halfVector = normalizeVec3(sumVec3(mulVec3Float(viewDir, -1.0), mulVec3Float(lightDir, +1.0)));

        flipped = true;
    }
}



function count(N) {

    particleCount = Math.max(0, Math.floor(N));

    const SIZE_OF_UINT = 4;
    const PARTICLE_STRIDE_BYTES = 48;

    const particleBytes = Math.max(1, particleCount) * PARTICLE_STRIDE_BYTES;

    comp_generate_position.workgroups = [Math.max(1, Math.ceil(particleCount / 256)), 1, 1];

    buff_particles.bytecount = particleBytes;
    buff_sorted_indices.bytecount = Math.max(1, particleCount) * SIZE_OF_UINT;
    buff_particle_draw.bytecount = Math.max(1, particleCount) * 48;

    buff_radix_histogram.bytecount = RADIX_BINS * SIZE_OF_UINT;
    buff_radix_cursor.bytecount = RADIX_BINS * SIZE_OF_UINT;
    buff_radix_bin_offsets.bytecount = RADIX_BINS * SIZE_OF_UINT;
    buff_radix_block_sums.bytecount = RADIX_BLOCK_COUNT * SIZE_OF_UINT;
    buff_radix_block_offsets.bytecount = RADIX_BLOCK_COUNT * SIZE_OF_UINT;
    buff_sort_range.bytecount = 2 * SIZE_OF_UINT;

    comp_generate_position.param("COUNT", particleCount);
    comp_generate_position.param("SQRT_COUNT", Math.ceil(Math.sqrt(Math.max(1, particleCount))));

    comp_radix_find_key_range.workgroups = [Math.max(1, Math.ceil(particleCount / 256)), 1, 1];
    comp_radix_find_key_range.param("COUNT", particleCount);

    comp_radix_build_histogram.workgroups = [Math.max(1, Math.ceil(particleCount / 256)), 1, 1];
    comp_radix_build_histogram.param("COUNT", particleCount);

    comp_radix_scatter_particles.workgroups = [Math.max(1, Math.ceil(particleCount / 256)), 1, 1];
    comp_radix_scatter_particles.param("COUNT", particleCount);

    sliceSize = Math.ceil(particleCount / sliceCount);

    comp_render_shadow.param("COUNT", particleCount);
    comp_render_shadow.param("SLICE_SIZE", sliceSize);
    comp_render_shadow.workgroups = [Math.ceil(particleCount / 256), 1, 1];

    comp_prepare_eye_particles.param("pc.SLICE_SIZE", sliceSize);
    comp_prepare_shadow_particles.param("pc.SLICE_SIZE", sliceSize);

    comp_prepare_eye_particles.workgroups = [Math.max(1, Math.ceil(particleCount / 256)), 1, 1];
    comp_prepare_shadow_particles.workgroups = [Math.max(1, Math.ceil(particleCount / 256)), 1, 1];

    draw_particles.instancecount = particleCount;
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
    viewDir = proxy_camera.send("getdirection");

    let up = [0, 1, 0];

    let lightPos = new Float32Array([-lightDir[0] * 3, -lightDir[1] * 3, -lightDir[2] * 3]);


    let matrices = {
        V: m.lookAt(pos, at, up),
        P: m.perspective(lensAngle, RATIO, nearClip, farClip),
        ligV: m.lookAt(lightPos, [0,0,0], up),
        ligP: m.ortho(-1.5, 1.5, -1.5, 1.5, 0.1, 30)
    };

    matrices.VP = m.mulMat4(matrices.P, matrices.V);
    matrices.ligVP = m.mulMat4(matrices.ligP, matrices.ligV);
    return matrices;
}

function sort_particles(){

    comp_radix_clear.bang();

    // Measure the actual min/max key for this frame before quantizing to bins.
    comp_radix_find_key_range.bang();

    comp_radix_build_histogram.bang();
    comp_radix_scan_bins.bang();
    comp_radix_scan_block_sums.bang();
    comp_radix_add_block_offsets.bang();

    // Scatter only uint particle indices. buff_particles is not reordered.
    comp_radix_scatter_particles.bang();
}

function clear_color_attachments(){

    comp_clear_color_target.bang();
    comp_clear_shadow_map.bang();
}

function prepare_eye_particles(matrices) {
    comp_prepare_eye_particles.param("VP", matrices.VP);
    comp_prepare_eye_particles.param("ligVP", matrices.ligVP);
    comp_prepare_eye_particles.param("lightDir", lightDir);
    comp_prepare_eye_particles.param("ALPHA", ALPHA);
    comp_prepare_eye_particles.param("pc.COUNT", particleCount);
    comp_prepare_eye_particles.param("pc.SLICE_SIZE", sliceSize);
    comp_prepare_eye_particles.param("pc.FLIPPED", flipped ? 1 : 0);
    comp_prepare_eye_particles.bang();
}

function prepare_shadow_particles(matrices) {
    comp_prepare_shadow_particles.param("VP", matrices.ligVP);
    comp_prepare_shadow_particles.param("ALPHA", ALPHA);
    comp_prepare_shadow_particles.param("pc.COUNT", particleCount);
    comp_prepare_shadow_particles.param("pc.FLIPPED", flipped ? 1 : 0);
    comp_prepare_shadow_particles.bang();
}

function bang() {

    let transfrom = calc_matrices();

    compute_half_vector();

    comp_generate_position.param("TIME", TIME*0.1);
    comp_generate_position.param("SCALE", scale);
    comp_generate_position.param("HALF_VECTOR", halfVector);
    comp_generate_position.bang();

    sort_particles();

    clear_color_attachments();

    prepare_shadow_particles(transfrom);
    comp_render_shadow.param("P", transfrom.ligP);
    comp_render_shadow.param("FLIPPED", flipped ? 1 : 0);
    comp_render_shadow.bang();

    prepare_eye_particles(transfrom);
    draw_particles.param("V", transfrom.V);
    draw_particles.param("P", transfrom.P);
    draw_particles.param("ligV", transfrom.ligV);
    draw_particles.param("ligP", transfrom.ligP);
    draw_particles.param("lightDir", lightDir);
    render_particles.jit_gpu_draw(draw_particles.name);
    render_particles.bang();

    //composite background
    comp_composite_background.param("background", [0.2, 0.25, 0.4, 0.0]);
    comp_composite_background.bang();

    outlet(0, "source", img_color_target.name);
    outlet(0, "bang");
}
