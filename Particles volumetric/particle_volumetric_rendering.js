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
let sliceCount = 32;

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

let buff_slice = new JitterObject("jit.gpu.buffer");
buff_slice.bytecount = 4 * 4;

let comp_set_slice = new JitterObject("jit.gpu.compute");
comp_set_slice.shader = "comp_set_slice.comp";
comp_set_slice.workgroups = [1,1,1];
comp_set_slice.bind("buff_slice", buff_slice.name);

let buff_inside_counter = new JitterObject("jit.gpu.buffer"); // indirect draw args / visible count
buff_inside_counter.bytecount = 4 * 4;

let buff_inside = new JitterObject("jit.gpu.buffer"); // per-particle visibility flag, kept for compatibility
let buff_visible_indices = new JitterObject("jit.gpu.buffer"); // kept for compatibility

let comp_reset_inside_counter = new JitterObject("jit.gpu.compute");
comp_reset_inside_counter.file = "comp_reset_inside_counter.comp";
comp_reset_inside_counter.workgroups = [1, 1, 1];
comp_reset_inside_counter.bind("buff_inside_counter", buff_inside_counter.name);

let comp_generate_position = new JitterObject("jit.gpu.compute");
comp_generate_position.file = "comp_generate_position.comp";
comp_generate_position.bind("buff_particles", buff_particles.name);
comp_generate_position.param("RADIUS", RADIUS);

// -----------------------------------------------------------------------------
// Fast GPU radix/counting sort pipeline for arbitrary particle counts.
// Replaces the previous O(n log^2 n) bitonic merge path.
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

let shadowMapSize = 512;
const SHADOW_BLUR_LOCAL_SIZE = 128;

let img_shadow_map = new JitterObject("jit.gpu.image");
img_shadow_map.dim = [shadowMapSize, shadowMapSize];
img_shadow_map.format = "r32_float";

let img_shadow_map_prev = new JitterObject("jit.gpu.image");
img_shadow_map_prev.dim = [shadowMapSize, shadowMapSize];
img_shadow_map_prev.format = "r32_float";

let img_shadow_map_next = new JitterObject("jit.gpu.image");
img_shadow_map_next.dim = [shadowMapSize, shadowMapSize];
img_shadow_map_next.format = "r32_float";

// Ping-pong target for the separable shadow-map Gaussian blur.
// Horizontal blur writes here; vertical blur writes back into img_shadow_map.
let img_shadow_map_tmp = new JitterObject("jit.gpu.image");
img_shadow_map_tmp.dim = [shadowMapSize, shadowMapSize];
img_shadow_map_tmp.format = "r32_float";

let comp_clear_color_target = new JitterObject("jit.gpu.compute");
comp_clear_color_target.shader = "comp_clear_color_target.comp";
comp_clear_color_target.bind("img_color_target", img_color_target.name);
comp_clear_color_target.workgroups = [Math.ceil(VIEWPORT[0] / 16), Math.ceil(VIEWPORT[1] / 16), 1];

let comp_clear_shadow_map = new JitterObject("jit.gpu.compute");
comp_clear_shadow_map.shader = "comp_clear_shadow_map.comp";
comp_clear_shadow_map.bind("img_shadow_map", img_shadow_map.name);
comp_clear_shadow_map.workgroups = [Math.ceil(shadowMapSize / 16), Math.ceil(shadowMapSize / 16), 1];

// -----------------------------------------------------------------------------
// High-performance separable Gaussian blur for the accumulated shadow map.
// Each pass uses shared memory with halo pixels, so each source texel is loaded
// once per tile instead of once per tap. The result of blur_shadow_map() is back
// in img_shadow_map, so the existing particle shader keeps sampling the same name.
// -----------------------------------------------------------------------------
let comp_shadow_blur_h = new JitterObject("jit.gpu.compute");
comp_shadow_blur_h.shader = "comp_shadow_blur_h.comp";
comp_shadow_blur_h.bind("img_src", img_shadow_map.name);
comp_shadow_blur_h.bind("img_dst", img_shadow_map_tmp.name);
comp_shadow_blur_h.workgroups = [Math.ceil(shadowMapSize / SHADOW_BLUR_LOCAL_SIZE), shadowMapSize, 1];

let comp_shadow_blur_v = new JitterObject("jit.gpu.compute");
comp_shadow_blur_v.shader = "comp_shadow_blur_v.comp";
comp_shadow_blur_v.bind("img_src", img_shadow_map_tmp.name);
comp_shadow_blur_v.bind("img_dst", img_shadow_map.name);
comp_shadow_blur_v.workgroups = [shadowMapSize, Math.ceil(shadowMapSize / SHADOW_BLUR_LOCAL_SIZE), 1];

// -----------------------------------------------------------------------------
// Per-slice particle preparation.
//  - transform particle center to clip space
//  - sample previous eye/shadow slice for opacity culling
//  - sample the light buffer for eye-pass shadowing
// The vertex shaders then only add a cheap projected x/y billboard offset.
// -----------------------------------------------------------------------------
let comp_prepare_eye_particles = new JitterObject("jit.gpu.compute");
comp_prepare_eye_particles.shader = "comp_prepare_eye_particles.comp";
comp_prepare_eye_particles.bind("buff_particles", buff_particles.name);
comp_prepare_eye_particles.bind("buff_sorted_indices", buff_sorted_indices.name);
comp_prepare_eye_particles.bind("img_shadow_map", img_shadow_map.name);
comp_prepare_eye_particles.bind("img_prev_slice", img_color_target.name);
comp_prepare_eye_particles.bind("buff_particle_draw", buff_particle_draw.name);

let comp_prepare_shadow_particles = new JitterObject("jit.gpu.compute");
comp_prepare_shadow_particles.shader = "comp_prepare_shadow_particles.comp";
comp_prepare_shadow_particles.bind("buff_particles", buff_particles.name);
comp_prepare_shadow_particles.bind("buff_sorted_indices", buff_sorted_indices.name);
comp_prepare_shadow_particles.bind("img_shadow_map_prev", img_shadow_map_prev.name);
comp_prepare_shadow_particles.bind("buff_particle_draw", buff_particle_draw.name);

// shadow pass

let draw_shadow = new JitterObject("jit.gpu.draw");
draw_shadow.shader = "draw_shadow.rend";
draw_shadow.vb0 = buff_quad.name;
draw_shadow.buff_particle_draw = buff_particle_draw.name;
draw_shadow.indirect = buff_slice.name;
draw_shadow.topology = "trianglestrip";
draw_shadow.elemcount = 4;
draw_shadow.blendenable = true;
draw_shadow.depth_write = false;
draw_shadow.param("ALPHA", ALPHA);

let render_shadow = new JitterObject("jit.gpu.render");
render_shadow.colorattachments = 1;
render_shadow.depth = false;
render_shadow.depthimg = img_shadow_map.name;
render_shadow.colorimg0 = img_shadow_map.name;
render_shadow.clearcolor0 = [0, 0, 0, 0];
render_shadow.colorloadop0 = "load";

// main pass

let draw_particles = new JitterObject("jit.gpu.draw");
draw_particles.shader = "draw_particles.rend";
draw_particles.vb0 = buff_quad.name;
draw_particles.buff_particle_draw = buff_particle_draw.name;
draw_particles.indirect = buff_slice.name;
draw_particles.topology = "trianglestrip";
draw_particles.elemcount = 4;
draw_particles.blendenable = true;
draw_particles.depth_write = false;
draw_particles.param("ALPHA", ALPHA);

let render_particles = new JitterObject("jit.gpu.render");
render_particles.colorattachments = 1;
render_particles.depth = false;
render_particles.colorimg0 = img_color_target.name;
render_particles.colorloadop0 = "load";

let comp_composite_background = new JitterObject("jit.gpu.compute");
comp_composite_background.shader = "comp_composite_background.comp";
comp_composite_background.workgroups = [Math.ceil(VIEWPORT[0] / 16), Math.ceil(VIEWPORT[1] / 16), 1];
comp_composite_background.bind("img_color_target", img_color_target.name);

count(100000);

function camera_name(name) {
    proxy_camera.name = name;
}

function setalpha(x){
    ALPHA = x;
    draw_shadow.param("ALPHA", ALPHA);
    draw_particles.param("ALPHA", ALPHA);
}

function setradius(x){
    RADIUS = x;
    comp_generate_position.param("RADIUS", RADIUS);
}

function setslice_count(x){
    sliceCount = x;
    count(particleCount);
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

    buff_inside.bytecount = Math.max(1, particleCount) * SIZE_OF_UINT;
    buff_visible_indices.bytecount = Math.max(1, particleCount) * SIZE_OF_UINT;

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

    draw_particles.instancecount = Math.ceil(particleCount / sliceCount);
    draw_shadow.instancecount = Math.ceil(particleCount / sliceCount);

    sliceSize = Math.ceil(particleCount / sliceCount);
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

    return {
        V: m.lookAt(pos, at, up),
        P: m.perspective(lensAngle, RATIO, nearClip, farClip),
        ligV: m.lookAt(lightPos, [0,0,0], up),
        ligP: m.ortho(-1.5, 1.5, -1.5, 1.5, 0.1, 30)
    };
}

function sort_particles(){
    if (particleCount <= 0) return;

    comp_radix_clear.bang();

    // Measure the actual min/max key for this frame before quantizing to bins.
    // This removes the old fixed [-1.8, +1.8] sort range limitation.
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

function blur_shadow_map(){
    comp_shadow_blur_h.bang();
    comp_shadow_blur_v.bang();
}

function prepare_eye_particles(firstInstance, count, matrices, sliceProgress) {
    comp_prepare_eye_particles.workgroups = [Math.max(1, Math.ceil(count / 256)), 1, 1];
    comp_prepare_eye_particles.param("V", matrices.V);
    comp_prepare_eye_particles.param("P", matrices.P);
    comp_prepare_eye_particles.param("ligV", matrices.ligV);
    comp_prepare_eye_particles.param("ligP", matrices.ligP);
    comp_prepare_eye_particles.param("lightDir", lightDir);
    comp_prepare_eye_particles.param("ALPHA", ALPHA);
    comp_prepare_eye_particles.param("test", sliceProgress);
    comp_prepare_eye_particles.param("pc.FIRST_INSTANCE", firstInstance);
    comp_prepare_eye_particles.param("pc.COUNT", count);
    comp_prepare_eye_particles.param("pc.TOTAL_COUNT", particleCount);
    comp_prepare_eye_particles.bang();
}

function prepare_shadow_particles(firstInstance, count, matrices) {
    comp_prepare_shadow_particles.workgroups = [Math.max(1, Math.ceil(count / 256)), 1, 1];
    comp_prepare_shadow_particles.param("V", matrices.ligV);
    comp_prepare_shadow_particles.param("P", matrices.ligP);
    comp_prepare_shadow_particles.param("ALPHA", ALPHA);
    comp_prepare_shadow_particles.param("pc.FIRST_INSTANCE", firstInstance);
    comp_prepare_shadow_particles.param("pc.COUNT", count);
    comp_prepare_shadow_particles.param("pc.TOTAL_COUNT", particleCount);
    comp_prepare_shadow_particles.bang();
}

function bang() {

    let transfrom = calc_matrices();
    compute_half_vector();

    draw_particles.blendcolorsrc = "inv_dst_alpha";
    draw_particles.blendcolordst = "one";
    draw_particles.blendalphasrc = "inv_dst_alpha";
    draw_particles.blendalphadst = "one";

    comp_generate_position.param("TIME", TIME*0.1);
    comp_generate_position.param("SCALE", 0.0002);
    comp_generate_position.param("HALF_VECTOR", halfVector);
    comp_generate_position.bang();

    sort_particles();

    draw_particles.param("V", transfrom.V);
    draw_particles.param("P", transfrom.P);
    draw_particles.param("ligV", transfrom.ligV);
    draw_particles.param("ligP", transfrom.ligP);
    draw_particles.param("lightDir", lightDir);
    draw_shadow.param("V", transfrom.ligV);
    draw_shadow.param("P", transfrom.ligP);

    clear_color_attachments();

    // Sliced rendering. buff_sorted_indices is ascending in half-vector depth;
    for (let si = 0; si < sliceCount; si++) {
        let i = flipped ? (sliceCount - 1 - si) : si;
        let offset = sliceSize * i;

        let count = Math.min(particleCount, offset + sliceSize) - offset;
        if (count <= 0) continue;

        comp_set_slice.param("pc.instanceCount", count);
        comp_set_slice.param("pc.firstInstance", offset);
        comp_set_slice.bang();

        // Eye pass: prepare center-based opacity/shadow once per particle, then
        // draw the 4 quad vertices using the cached per-particle data.
        prepare_eye_particles(offset, count, transfrom, si / sliceCount);
        draw_particles.param("test", si / sliceCount);
        render_particles.jit_gpu_draw(draw_particles.name);
        render_particles.bang();

        if(si < sliceCount - 1){ //skip the last iteration
            // Shadow pass: same idea, but in light clip space.
            prepare_shadow_particles(offset, count, transfrom);
            render_shadow.jit_gpu_draw(draw_shadow.name);
            render_shadow.bang();

            // Blur the accumulated light buffer before the next eye pass samples it (every 2 slices).
            blur_shadow_map();          
        }
    }

    //composite background
    comp_composite_background.param("background", [0.2, 0.25, 0.4, 0.0]);
    comp_composite_background.bang();

    outlet(0, "source", img_color_target.name);
    outlet(0, "bang");
}
