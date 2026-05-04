autowhatch = 1; inlets = 1; outlets = 1;

const m = require("Patcher://vector_math.js");

let TIME = 0;
let VIEWPORT = [1920, 1080];
let RATIO = VIEWPORT[0] / VIEWPORT[1];
let RADIUS = 0.01;
let ALPHA = 0.05;
let pos, at, farClip, nearClip, lensAngle, viewDir;
let particleCount = 0;
let sliceCount = 64;       // variable; call slice_count(N) or bins(N)
let radius_multiplier = 1.0;

let lightDir = normalizeVec3([0.3, 1, -0.2]);
let halfVector, flipped;

let proxy_camera = new JitterObject("jit.proxy");

// Particle: vec3 pos, float rad, vec4 col, float key, uint id. std430 stride = 48 bytes.
let buff_particles = new JitterObject("jit.gpu.buffer");

let buff_quad = new JitterObject("jit.gpu.buffer");
let mat_quad = new JitterMatrix(4, "float32", 4);
mat_quad.setcell(0, "val", [-1, -1, 0, 0]);
mat_quad.setcell(1, "val", [+1, -1, 0, 0]);
mat_quad.setcell(2, "val", [-1, +1, 0, 0]);
mat_quad.setcell(3, "val", [+1, +1, 0, 0]);
buff_quad.jit_matrix(mat_quad.name);

// Current indirect draw args: vertexCount, instanceCount, firstVertex, firstInstance.
let buff_slice = new JitterObject("jit.gpu.buffer");
buff_slice.bytecount = 4 * 4;

// Slice/bucket buffers.
// binCounts[i]       = particle count in bin i
// binOffsets[i]      = exclusive prefix-sum start of bin i in bucketIndices
// binWriteCounts[i]  = temporary atomic cursor used while filling bucketIndices
// bucketIndices[k]   = particle index to render for logical instance k
let buff_bin_counts = new JitterObject("jit.gpu.buffer");
let buff_bin_offsets = new JitterObject("jit.gpu.buffer");
let buff_bin_write_counts = new JitterObject("jit.gpu.buffer");
let buff_bucket_indices = new JitterObject("jit.gpu.buffer");

let comp_generate_position = new JitterObject("jit.gpu.compute");
comp_generate_position.file = "comp_generate_position.comp";
comp_generate_position.bind("buff_particles", buff_particles.name);
comp_generate_position.param("RADIUS", RADIUS);

let comp_reset_bins = new JitterObject("jit.gpu.compute");
comp_reset_bins.file = "comp_reset_bins.comp";
comp_reset_bins.bind("buff_bin_counts", buff_bin_counts.name);
comp_reset_bins.bind("buff_bin_offsets", buff_bin_offsets.name);
comp_reset_bins.bind("buff_bin_write_counts", buff_bin_write_counts.name);

let comp_count_bins = new JitterObject("jit.gpu.compute");
comp_count_bins.file = "comp_count_bins.comp";
comp_count_bins.bind("buff_particles", buff_particles.name);
comp_count_bins.bind("buff_bin_counts", buff_bin_counts.name);

let comp_prefix_bins = new JitterObject("jit.gpu.compute");
comp_prefix_bins.file = "comp_prefix_bins.comp";
comp_prefix_bins.workgroups = [1, 1, 1];
comp_prefix_bins.bind("buff_bin_counts", buff_bin_counts.name);
comp_prefix_bins.bind("buff_bin_offsets", buff_bin_offsets.name);
comp_prefix_bins.bind("buff_bin_write_counts", buff_bin_write_counts.name);

let comp_fill_bins = new JitterObject("jit.gpu.compute");
comp_fill_bins.file = "comp_fill_bins.comp";
comp_fill_bins.bind("buff_particles", buff_particles.name);
comp_fill_bins.bind("buff_bin_offsets", buff_bin_offsets.name);
comp_fill_bins.bind("buff_bin_write_counts", buff_bin_write_counts.name);
comp_fill_bins.bind("buff_bucket_indices", buff_bucket_indices.name);

let comp_set_bin_slice = new JitterObject("jit.gpu.compute");
comp_set_bin_slice.file = "comp_set_bin_slice.comp";
comp_set_bin_slice.workgroups = [1, 1, 1];
comp_set_bin_slice.bind("buff_slice", buff_slice.name);
comp_set_bin_slice.bind("buff_bin_counts", buff_bin_counts.name);
comp_set_bin_slice.bind("buff_bin_offsets", buff_bin_offsets.name);

let img_color_target = new JitterObject("jit.gpu.image");
img_color_target.dim = [VIEWPORT[0], VIEWPORT[1]];
img_color_target.format = "rgba32_float";

let shadowMapSize = 1024;
let img_shadow_map = new JitterObject("jit.gpu.image");
img_shadow_map.dim = [shadowMapSize, shadowMapSize];
img_shadow_map.format = "r32_float";

// Ping-pong target for separable shadow blur. Do not blur img_shadow_map in-place.
let img_shadow_tmp = new JitterObject("jit.gpu.image");
img_shadow_tmp.dim = [shadowMapSize, shadowMapSize];
img_shadow_tmp.format = "r32_float";

let comp_clear_color_target = new JitterObject("jit.gpu.compute");
comp_clear_color_target.file = "comp_clear_color_target.comp";
comp_clear_color_target.bind("img_color_target", img_color_target.name);
comp_clear_color_target.workgroups = [Math.ceil(VIEWPORT[0] / 16), Math.ceil(VIEWPORT[1] / 16), 1];

let comp_clear_shadow_map = new JitterObject("jit.gpu.compute");
comp_clear_shadow_map.file = "comp_clear_shadow_map.comp";
comp_clear_shadow_map.bind("img_shadow_map", img_shadow_map.name);
comp_clear_shadow_map.workgroups = [Math.ceil(shadowMapSize / 16), Math.ceil(shadowMapSize / 16), 1];

// Shadow pass.
let draw_shadow = new JitterObject("jit.gpu.draw");
draw_shadow.shader = "draw_shadow_bucketed.rend";
draw_shadow.vb0 = buff_quad.name;
draw_shadow.buff_particles = buff_particles.name;
draw_shadow.buff_bucket_indices = buff_bucket_indices.name;
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

// Eye pass.
let draw_particles = new JitterObject("jit.gpu.draw");
draw_particles.shader = "draw_particles_bucketed.rend";
draw_particles.vb0 = buff_quad.name;
draw_particles.buff_particles = buff_particles.name;
draw_particles.buff_bucket_indices = buff_bucket_indices.name;
draw_particles.indirect = buff_slice.name;
draw_particles.img_shadow_map = img_shadow_map.name;
draw_particles.img_prev_slice = img_color_target.name;
draw_particles.topology = "trianglestrip";
draw_particles.elemcount = 4;
draw_particles.blendenable = true;
draw_particles.depth_write = false;
draw_particles.param("ALPHA", ALPHA);

let render_particles = new JitterObject("jit.gpu.render");
render_particles.colorattachments = 1;
render_particles.depth = false;
render_particles.colorimg0 = img_color_target.name;
render_particles.msaa = 1;
render_particles.colorloadop0 = "load";

// Separable Gaussian blur for the light/shadow buffer.
// H: img_shadow_map -> img_shadow_tmp
// V: img_shadow_tmp -> img_shadow_map
let comp_blur_shadow_h = new JitterObject("jit.gpu.compute");
comp_blur_shadow_h.file = "comp_blur_shadow_h.comp";
comp_blur_shadow_h.bind("img_shadow_src", img_shadow_map.name);
comp_blur_shadow_h.bind("img_shadow_dst", img_shadow_tmp.name);
comp_blur_shadow_h.workgroups = [Math.ceil(shadowMapSize / 16), Math.ceil(shadowMapSize / 16), 1];

let comp_blur_shadow_v = new JitterObject("jit.gpu.compute");
comp_blur_shadow_v.file = "comp_blur_shadow_v.comp";
comp_blur_shadow_v.bind("img_shadow_src", img_shadow_tmp.name);
comp_blur_shadow_v.bind("img_shadow_dst", img_shadow_map.name);
comp_blur_shadow_v.workgroups = [Math.ceil(shadowMapSize / 16), Math.ceil(shadowMapSize / 16), 1];

count(100000);

function camera_name(name) {
    proxy_camera.name = name;
}

function normalizeVec3(v) {
    const len = Math.hypot(v[0], v[1], v[2]);
    if (len === 0) return new Float32Array([0, 0, 0]);
    const inv = 1 / len;
    return new Float32Array([v[0] * inv, v[1] * inv, v[2] * inv]);
}
function mulVec3Float(a, b) { return [a[0] * b, a[1] * b, a[2] * b]; }
function sumVec3(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

function light_direction() {
    lightDir = normalizeVec3([arguments[0], arguments[1], arguments[2]]);
}

function compute_half_vector() {
    if (dot(viewDir, lightDir) > 0.0) {
        halfVector = normalizeVec3(sumVec3(viewDir, lightDir));
        flipped = true;
    } else {
        halfVector = normalizeVec3(sumVec3(mulVec3Float(viewDir, -1.0), mulVec3Float(lightDir, -1.0)));
        flipped = false;
    }
}

function configure_bucket_buffers() {
    const SIZE_OF_UINT = 4;

    buff_bin_counts.bytecount = sliceCount * SIZE_OF_UINT;
    buff_bin_offsets.bytecount = sliceCount * SIZE_OF_UINT;
    buff_bin_write_counts.bytecount = sliceCount * SIZE_OF_UINT;

    comp_reset_bins.workgroups = [Math.ceil(sliceCount / 256), 1, 1];
    comp_reset_bins.param("BIN_COUNT", sliceCount);
    comp_count_bins.param("BIN_COUNT", sliceCount);
    comp_prefix_bins.param("BIN_COUNT", sliceCount);
    comp_fill_bins.param("BIN_COUNT", sliceCount);
    comp_set_bin_slice.param("BIN_COUNT", sliceCount);
}

function count(N) {
    particleCount = Math.max(1, Math.floor(N));

    const SIZE_OF_UINT = 4;
    const PARTICLE_STRIDE = 48;

    buff_particles.bytecount = particleCount * PARTICLE_STRIDE;
    buff_bucket_indices.bytecount = particleCount * SIZE_OF_UINT;

    comp_generate_position.workgroups = [Math.ceil(particleCount / 256), 1, 1];
    comp_generate_position.param("COUNT", particleCount);
    comp_generate_position.param("SQRT_COUNT", Math.ceil(Math.sqrt(particleCount)));

    comp_count_bins.workgroups = [Math.ceil(particleCount / 256), 1, 1];
    comp_count_bins.param("COUNT", particleCount);

    comp_fill_bins.workgroups = [Math.ceil(particleCount / 256), 1, 1];
    comp_fill_bins.param("COUNT", particleCount);

    // Non-indirect fallback only. The real count comes from buff_slice.
    draw_particles.instancecount = particleCount;
    draw_shadow.instancecount = particleCount;

    configure_bucket_buffers();
}

// Variable number of bins/slices. Use this from Max: slice_count 32, slice_count 64, etc.
function slice_count(N) {
    sliceCount = Math.max(1, Math.floor(N));
    configure_bucket_buffers();
}

// Alias, because "bins 64" reads nicely in the patcher.
function bins(N) {
    slice_count(N);
}

function blur_shadow_map() {

    comp_blur_shadow_h.bang();
    comp_blur_shadow_v.bang();
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
        ligV: m.lookAt(lightPos, at, up),
        ligP: m.ortho(-5, 5, -5, 5, 0.1, 30)
    };
}

function update_bucket_key_range() {
    // Particle positions are generated roughly in [-1,1]^3 plus jitter.
    // The exact projection range along halfVector is the box extent projected onto halfVector.
    // Add radius/jitter padding so particles near the edge do not clamp too aggressively.
    let boxExtent = 1.0 + RADIUS + 0.02;
    let projectedExtent = boxExtent * (Math.abs(halfVector[0]) + Math.abs(halfVector[1]) + Math.abs(halfVector[2]));
    projectedExtent = Math.max(projectedExtent, 0.0001);

    let keyMin = -projectedExtent;
    let keyInvRange = 1.0 / (2.0 * projectedExtent);

    comp_count_bins.param("KEY_MIN", keyMin);
    comp_count_bins.param("KEY_INV_RANGE", keyInvRange);
    comp_fill_bins.param("KEY_MIN", keyMin);
    comp_fill_bins.param("KEY_INV_RANGE", keyInvRange);
}

function build_slice_buckets() {
    comp_reset_bins.bang();
    comp_count_bins.bang();
    comp_prefix_bins.bang();
    comp_fill_bins.bang();
}

function clear_color_attachments() {
    comp_clear_color_target.bang();
    comp_clear_shadow_map.bang();
}

function bang() {
    let transform = calc_matrices();
    compute_half_vector();

    comp_generate_position.param("TIME", TIME * 0.1);
    comp_generate_position.param("SCALE", 0.001);
    comp_generate_position.param("HALF_VECTOR", halfVector);
    comp_generate_position.param("FLIPPED", flipped);
    comp_generate_position.bang();

    update_bucket_key_range();
    build_slice_buckets();

    draw_particles.param("V", transform.V);
    draw_particles.param("P", transform.P);
    draw_particles.param("ligV", transform.ligV);
    draw_particles.param("ligP", transform.ligP);
    draw_particles.param("lightDir", lightDir);
    draw_shadow.param("V", transform.ligV);
    draw_shadow.param("P", transform.ligP);
    draw_shadow.param("radius_multiplier", radius_multiplier);

    clear_color_attachments();

    // Draw bins back-to-front/front-to-back depending on the half-angle branch.
    // Each iteration writes buff_slice from the GPU-side binCounts/binOffsets.
    for (let si = 0; si < sliceCount; si++) {
        let binIndex = flipped ? (sliceCount - 1 - si) : si;

        comp_set_bin_slice.param("BIN_INDEX", binIndex);
        comp_set_bin_slice.bang();

        render_particles.jit_gpu_draw(draw_particles.name);
        render_particles.bang();

        render_shadow.jit_gpu_draw(draw_shadow.name);
        render_shadow.bang();

        // Diffuse the updated light buffer before the next eye slice samples it.
        // This matches the paper's scattering approximation, but without illegal
        // read/write feedback on the same image.
        //blur_shadow_map();
    }

    outlet(0, "source", img_color_target.name);
    outlet(0, "bang");
}
