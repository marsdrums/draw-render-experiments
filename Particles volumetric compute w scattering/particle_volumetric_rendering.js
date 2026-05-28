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
let sliceCount = 512;
let scale = 0.0006;
let shadowMapSize = 512;
let shadowBlurScale = 0.0005;
let ambientOcclusionStrength = 1.0;
let ambientOcclusionRadius = 0.1;
// Ambient occlusion is cached on a cubic probe grid over the particle AABB.
// The sample count now controls the quality of each cached probe, not a per-particle loop.
let ambientOcclusionCacheResolution = 24;
let ambientOcclusionCacheSampleCount = 16;
// Temporal AO cache feedback: 0.0 = no history, 1.0 = freeze previous cache.
let ambientOcclusionCacheHistoryWeight = 0.96;
let ambientOcclusionCacheHistoryValid = 0;
let ambientOcclusionCacheWriteIndex = 0;
//let ambientLight = [0.2, 0.25, 0.4];
//let ambientLight = [0.8, 0.8, 0.8];
let ambientLight = [0.007, 0.007, 0.007];
//let lightColor = [4.0, 3.0, 2.0];
//let lightColor = [0.0, 0.0, 0.0];
let lightColor = [2.0, 1.5, 1.0];
let lightFlareStrength = 0.20;
let lightFlareCoreRadius = 0.135;
let lightFlareHaloRadius = 0.28;
let lightFlareHaloStrength = 0.25;
// World-space padding added around the GPU-fitted light-space AABB.
let lightFitPadding = 0.3;
let motionBlurStrength = 1.0;
let motionBlurMaxStretch = 32.0;
let previousGeneratedTime = null;
let display_probes = false;

let DENSITY_WORD_COUNT, DENSITY_BITS_PER_WORD, DENSITY_SLICE_COUNT;

// One-pass radix/counting sort settings.
// The sort key range is measured on the GPU every frame, so particle positions
// are no longer assumed to live in any fixed range such as [-1, 1].
const RADIX_BITS = 16;
const RADIX_BINS = 1 << RADIX_BITS;      // 65536 depth buckets
const RADIX_BLOCK_SIZE = 256;            // scan block size
const RADIX_BLOCK_COUNT = RADIX_BINS / RADIX_BLOCK_SIZE;

let lightDir = normalizeVec3([-1,-1,-1]);

let proxy_camera = new JitterObject("jit.proxy");

// particle buffer: pos(vec3), rad(float), col(vec4), key(float), id(uint), padding, prevPos(vec3), padding
// The particle buffer remains in original/generated order. Sorting now produces
// a compact uint permutation buffer instead of moving full Particle structs.
let buff_particles = new JitterObject("jit.gpu.buffer");
let buff_sorted_indices = new JitterObject("jit.gpu.buffer");

// Per-particle data prepared once per slice and reused by the 4 quad vertices.
// Layout matches DrawParticle in the draw/prepare shaders: 4 vec4 = 64 bytes.
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
let buff_sort_range = new JitterObject("jit.gpu.buffer"); // uint ordered-float camera key min/max only
// Precomputed projection of the particle AABB onto lightDir: vec4(lightMin, lightMax, invRange, unused).
let buff_light_slice_range = new JitterObject("jit.gpu.buffer");
// GPU-fitted light-space projection matrices: mat4 lightP + mat4 lightVP.
let buff_light_matrices = new JitterObject("jit.gpu.buffer");
// Decoded particle AABB: vec4(min.xyz, unused), vec4(max.xyz, unused).
let buff_particle_aabb = new JitterObject("jit.gpu.buffer");
// Particle AABB retained from the frame that generated the filtered irradiance history.
let buff_previous_particle_aabb = new JitterObject("jit.gpu.buffer");
// Ping-pong reduction buffers for the atomics-free particle AABB reduction.
// Each element is AabbRecord { vec4 bmin; vec4 bmax; } = 32 bytes.
let buff_aabb_reduce_a = new JitterObject("jit.gpu.buffer");
let buff_aabb_reduce_b = new JitterObject("jit.gpu.buffer");
let aabbReductionInitialRecordCount = 1;
let aabbReductionFinalInA = 1;
// Raw current-frame irradiance / ambient-visibility cache generated once per probe.
// Stored as a cubic 3D image so the filter pass can access local neighborhoods directly.
let img_ambient_occlusion_cache_raw = new JitterObject("jit.gpu.image");
img_ambient_occlusion_cache_raw.format = "rgba32_float";
// Double-depth filtered irradiance cache. The front and back halves alternate
// between history and the current filtered output while the binding stays fixed.
// The eye pass samples this as a sampler3D, using hardware trilinear interpolation.
let img_ambient_occlusion_cache = new JitterObject("jit.gpu.image");
img_ambient_occlusion_cache.format = "rgba32_float";

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

// Atomics-free AABB reduction. First pass reduces particle positions into
// one AabbRecord per workgroup; subsequent passes reduce those records with ping-pong buffers.
let comp_reduce_particle_aabb_from_particles = new JitterObject("jit.gpu.compute");
comp_reduce_particle_aabb_from_particles.shader = "comp_reduce_particle_aabb_from_particles.comp";
comp_reduce_particle_aabb_from_particles.bind("buff_particles", buff_particles.name);
comp_reduce_particle_aabb_from_particles.bind("buff_aabb_reduce_a", buff_aabb_reduce_a.name);

let comp_reduce_particle_aabb_records = new JitterObject("jit.gpu.compute");
comp_reduce_particle_aabb_records.shader = "comp_reduce_particle_aabb_records.comp";
comp_reduce_particle_aabb_records.bind("buff_aabb_reduce_a", buff_aabb_reduce_a.name);
comp_reduce_particle_aabb_records.bind("buff_aabb_reduce_b", buff_aabb_reduce_b.name);

let comp_compute_light_slice_range = new JitterObject("jit.gpu.compute");
comp_compute_light_slice_range.shader = "comp_compute_light_slice_range.comp";
comp_compute_light_slice_range.workgroups = [1, 1, 1];
comp_compute_light_slice_range.bind("buff_aabb_reduce_a", buff_aabb_reduce_a.name);
comp_compute_light_slice_range.bind("buff_aabb_reduce_b", buff_aabb_reduce_b.name);
comp_compute_light_slice_range.bind("buff_light_slice_range", buff_light_slice_range.name);
comp_compute_light_slice_range.bind("buff_particle_aabb", buff_particle_aabb.name);
comp_compute_light_slice_range.bind("buff_light_matrices", buff_light_matrices.name);

// Retain the current AABB after the irradiance filter has consumed the previous one.
// The next frame uses this stored box to reproject temporal cache history in world space.
let comp_store_previous_particle_aabb = new JitterObject("jit.gpu.compute");
comp_store_previous_particle_aabb.shader = "comp_store_previous_particle_aabb.comp";
comp_store_previous_particle_aabb.workgroups = [1, 1, 1];
comp_store_previous_particle_aabb.bind("buff_particle_aabb", buff_particle_aabb.name);
comp_store_previous_particle_aabb.bind("buff_previous_particle_aabb", buff_previous_particle_aabb.name);

let img_color_target = new JitterObject("jit.gpu.image");
img_color_target.dim = [VIEWPORT[0], VIEWPORT[1]];
img_color_target.format = "rgba32_float";

let img_density_map = new JitterObject("jit.gpu.image");
// 256-slice occupancy map stored as eight horizontal r32ui tiles.
// Tile 0 stores slices    0..31, tile 1 stores   32..63,
// tile 2 stores slices   64..95, tile 3 stores   96..127,
// tile 4 stores slices 128..159, tile 5 stores 160..191,
// tile 6 stores slices 192..223, tile 7 stores 224..255.
img_density_map.dim = [shadowMapSize * Math.ceil(sliceCount / 32), shadowMapSize];
img_density_map.format = "r32_uint";

// Same tiling as img_density_map. For each logical pixel and density word,
// stores the number of occupied bits in all previous words. This lets the eye
// pass answer "how many blockers are before this slice?" with one prefix load
// plus one current-word load instead of scanning all previous words per tap.
let img_density_prefix_count_map = new JitterObject("jit.gpu.image");
img_density_prefix_count_map.dim = [shadowMapSize * Math.ceil(sliceCount / 32), shadowMapSize];
img_density_prefix_count_map.format = "r32_uint";

// Closest-to-light occupied slice per logical shadow-map pixel. This is derived
// from img_density_map after the shadow occupancy pass and used to choose a
// scattering blur radius based on distance from the first blocker.
let img_first_blocked_slice_map = new JitterObject("jit.gpu.image");
img_first_blocked_slice_map.dim = [shadowMapSize, shadowMapSize];
img_first_blocked_slice_map.format = "r32_uint";

let comp_clear_color_target = new JitterObject("jit.gpu.compute");
comp_clear_color_target.shader = "comp_clear_color_target.comp";
comp_clear_color_target.bind("img_color_target", img_color_target.name);
comp_clear_color_target.workgroups = [Math.ceil(VIEWPORT[0] / 16), Math.ceil(VIEWPORT[1] / 16), 1];

let comp_clear_shadow_map = new JitterObject("jit.gpu.compute");
comp_clear_shadow_map.shader = "comp_clear_shadow_map.comp";
comp_clear_shadow_map.bind("img_density_map", img_density_map.name);
comp_clear_shadow_map.workgroups = [Math.ceil((shadowMapSize * Math.ceil(sliceCount / 32)) / 16), Math.ceil(shadowMapSize / 16), 1];

let comp_build_density_metadata = new JitterObject("jit.gpu.compute");
comp_build_density_metadata.shader = "comp_build_density_metadata.comp";
comp_build_density_metadata.bind("img_density_map", img_density_map.name);
comp_build_density_metadata.bind("img_density_prefix_count_map", img_density_prefix_count_map.name);
comp_build_density_metadata.bind("img_first_blocked_slice_map", img_first_blocked_slice_map.name);
comp_build_density_metadata.workgroups = [Math.ceil(shadowMapSize / 16), Math.ceil(shadowMapSize / 16), 1];

// Build a reusable ambient-visibility cache on a cubic probe grid over the particle AABB.
// Each probe does the expensive 3D density-map occupancy sampling once; particles later
// trilinearly interpolate this cache in comp_prepare_eye_particles.comp.
let comp_build_ambient_occlusion_cache = new JitterObject("jit.gpu.compute");
comp_build_ambient_occlusion_cache.shader = "comp_build_ambient_occlusion_cache.comp";
comp_build_ambient_occlusion_cache.bind("buff_particle_aabb", buff_particle_aabb.name);
comp_build_ambient_occlusion_cache.bind("buff_light_slice_range", buff_light_slice_range.name);
comp_build_ambient_occlusion_cache.bind("img_density_map", img_density_map.name);
comp_build_ambient_occlusion_cache.bind("img_ambient_occlusion_cache_raw", img_ambient_occlusion_cache_raw.name);
comp_build_ambient_occlusion_cache.bind("img_env", "img_env");
comp_build_ambient_occlusion_cache.bind("buff_light_matrices", buff_light_matrices.name);

// Clamp previous filtered AO against the current raw AO neighborhood, then blend.
let comp_filter_ambient_occlusion_cache = new JitterObject("jit.gpu.compute");
comp_filter_ambient_occlusion_cache.shader = "comp_filter_ambient_occlusion_cache.comp";
comp_filter_ambient_occlusion_cache.bind("img_ambient_occlusion_cache_raw", img_ambient_occlusion_cache_raw.name);
comp_filter_ambient_occlusion_cache.bind("img_ambient_occlusion_cache", img_ambient_occlusion_cache.name);
// Same 3D cache bound as a sampler so history can be fetched trilinearly at a reprojected world-space position.
comp_filter_ambient_occlusion_cache.bind("tex_ambient_occlusion_cache_history", img_ambient_occlusion_cache.name);
comp_filter_ambient_occlusion_cache.bind("buff_particle_aabb", buff_particle_aabb.name);
comp_filter_ambient_occlusion_cache.bind("buff_previous_particle_aabb", buff_previous_particle_aabb.name);

let comp_prepare_eye_particles = new JitterObject("jit.gpu.compute");
comp_prepare_eye_particles.shader = "comp_prepare_eye_particles.comp";
comp_prepare_eye_particles.bind("buff_particles", buff_particles.name);
comp_prepare_eye_particles.bind("buff_sorted_indices", buff_sorted_indices.name);
comp_prepare_eye_particles.bind("img_density_map", img_density_map.name);
comp_prepare_eye_particles.bind("img_density_prefix_count_map", img_density_prefix_count_map.name);
comp_prepare_eye_particles.bind("img_first_blocked_slice_map", img_first_blocked_slice_map.name);
comp_prepare_eye_particles.bind("buff_particle_draw", buff_particle_draw.name);
comp_prepare_eye_particles.bind("buff_light_slice_range", buff_light_slice_range.name);
comp_prepare_eye_particles.bind("img_ambient_occlusion_cache", img_ambient_occlusion_cache.name);
comp_prepare_eye_particles.bind("buff_particle_aabb", buff_particle_aabb.name);
comp_prepare_eye_particles.bind("buff_light_matrices", buff_light_matrices.name);
comp_prepare_eye_particles.param("MOTION_BLUR_STRENGTH", motionBlurStrength);
comp_prepare_eye_particles.param("MOTION_BLUR_MAX_STRETCH", motionBlurMaxStretch);
comp_prepare_eye_particles.param("ASPECT", RATIO);
comp_prepare_eye_particles.param("AMBIENT_OCCLUSION_CACHE_RESOLUTION", ambientOcclusionCacheResolution);
comp_prepare_eye_particles.param("AMBIENT_LIGHT", ambientLight);
comp_prepare_eye_particles.param("LIGHT_COLOR", [lightColor[0], lightColor[1], lightColor[2], 0.0]);

// shadow pass
// The old comp_prepare_shadow_particles pass only generated a temporary shadow
// draw record which comp_render_shadow immediately consumed. The merged shadow
// pass now reads particles/sorted indices directly and performs that setup inline.
let comp_render_shadow = new JitterObject("jit.gpu.compute");
comp_render_shadow.shader = "comp_render_shadow.comp";
comp_render_shadow.bind("buff_particles", buff_particles.name);
comp_render_shadow.bind("buff_sorted_indices", buff_sorted_indices.name);
comp_render_shadow.bind("buff_light_slice_range", buff_light_slice_range.name);
comp_render_shadow.bind("img_density_map", img_density_map.name);
comp_render_shadow.bind("buff_light_matrices", buff_light_matrices.name);
comp_render_shadow.param("ALPHA", ALPHA);
comp_render_shadow.param("DENSITY_WORD_COUNT", Math.ceil(sliceCount / 32));

// main pass

let draw_probes = new JitterObject("jit.gpu.draw");
draw_probes.shader = "draw_probes.rend";
draw_probes.vb0 = buff_quad.name;
draw_probes.topology = "trianglestrip";
draw_probes.elemcount = 4;
draw_probes.instancecount = ambientOcclusionCacheResolution*ambientOcclusionCacheResolution*ambientOcclusionCacheResolution;
draw_probes.buff_particle_aabb = buff_particle_aabb.name;

let draw_particles = new JitterObject("jit.gpu.draw");
draw_particles.shader = "draw_particles.rend";
draw_particles.vb0 = buff_quad.name;
draw_particles.buff_particle_draw = buff_particle_draw.name;
draw_particles.topology = "trianglestrip";
draw_particles.elemcount = 4;
draw_particles.blendenable = true;
draw_particles.depth_write = false;
draw_particles.param("ALPHA", ALPHA);
draw_particles.param("INV_ASPECT", 1.0 / Math.max(RATIO, 1e-6));
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
comp_composite_background.bind("img_env", "img_env_full_res");
comp_composite_background.param("background", [ambientLight[0], ambientLight[1], ambientLight[2], 0.0]);
comp_composite_background.param("flareCenterVisibility", [0.5, 0.5, 0.0, 0.0]);
comp_composite_background.param("flareColorStrength", [lightColor[0], lightColor[1], lightColor[2], lightFlareStrength]);
comp_composite_background.param("flareShape", [lightFlareCoreRadius, lightFlareHaloRadius, RATIO, lightFlareHaloStrength]);
comp_composite_background.param("cameraForwardTanHalfFov", [0.0, 0.0, -1.0, Math.tan((45.0 * Math.PI / 180.0) * 0.5)]);
comp_composite_background.param("cameraRightAspect", [1.0, 0.0, 0.0, RATIO]);
comp_composite_background.param("cameraUp", [0.0, 1.0, 0.0, 0.0]);

function debug_probes(x){
	display_probes = x;
}

function update_ambient_occlusion_cache_offsets_for_frame() {
    const writeZOffset = ambientOcclusionCacheWriteIndex * ambientOcclusionCacheResolution;
    const readZOffset = (1 - ambientOcclusionCacheWriteIndex) * ambientOcclusionCacheResolution;

    // The filter pass reads history from one half and writes the new filtered
    // cache to the other half. The eye pass samples the just-written half.
    comp_filter_ambient_occlusion_cache.param("AMBIENT_OCCLUSION_CACHE_READ_Z_OFFSET", readZOffset);
    comp_filter_ambient_occlusion_cache.param("AMBIENT_OCCLUSION_CACHE_WRITE_Z_OFFSET", writeZOffset);
    comp_prepare_eye_particles.param("AMBIENT_OCCLUSION_CACHE_READ_Z_OFFSET", writeZOffset);
}

function reset_ambient_occlusion_cache_history() {
    ambientOcclusionCacheHistoryValid = 0;
    ambientOcclusionCacheWriteIndex = 0;
    update_ambient_occlusion_cache_offsets_for_frame();
}

function resize_ambient_occlusion_cache() {
    ambientOcclusionCacheResolution = Math.max(1, Math.floor(ambientOcclusionCacheResolution));
    const probeCount = ambientOcclusionCacheResolution * ambientOcclusionCacheResolution * ambientOcclusionCacheResolution;
    img_ambient_occlusion_cache_raw.dim = [ambientOcclusionCacheResolution, ambientOcclusionCacheResolution, ambientOcclusionCacheResolution];
    // Two Z-halves in one 3D texture: previous filtered history and current filtered output.
    img_ambient_occlusion_cache.dim = [ambientOcclusionCacheResolution, ambientOcclusionCacheResolution, ambientOcclusionCacheResolution * 2];
    comp_build_ambient_occlusion_cache.workgroups = [Math.max(1, Math.ceil(probeCount / 256)), 1, 1];
    comp_filter_ambient_occlusion_cache.workgroups = [Math.max(1, Math.ceil(probeCount / 256)), 1, 1];
    comp_build_ambient_occlusion_cache.param("AMBIENT_OCCLUSION_CACHE_RESOLUTION", ambientOcclusionCacheResolution);
    comp_filter_ambient_occlusion_cache.param("AMBIENT_OCCLUSION_CACHE_RESOLUTION", ambientOcclusionCacheResolution);
    comp_prepare_eye_particles.param("AMBIENT_OCCLUSION_CACHE_RESOLUTION", ambientOcclusionCacheResolution);
    reset_ambient_occlusion_cache_history();
}

resize_ambient_occlusion_cache();
count(1000000);
setslice_count(256);

function camera_name(name) {
    proxy_camera.name = name;
}

function setalpha(x){
    ALPHA = x;
    shadowBlurScale = 0.0005 * (1 - ALPHA);
    draw_particles.param("ALPHA", ALPHA);
    comp_render_shadow.param("ALPHA", ALPHA);
    comp_build_ambient_occlusion_cache.param("ALPHA", ALPHA);
}

function setradius(x){
    RADIUS = x;
    comp_generate_position.param("RADIUS", RADIUS);
}

function setslice_count(x){
    sliceCount = Math.max(1, x);
    DENSITY_WORD_COUNT = Math.ceil(sliceCount / 32);
    DENSITY_BITS_PER_WORD = 32;
    DENSITY_SLICE_COUNT = DENSITY_WORD_COUNT * DENSITY_BITS_PER_WORD;
    img_density_map.dim = [shadowMapSize * DENSITY_WORD_COUNT, shadowMapSize];
    img_density_prefix_count_map.dim = [shadowMapSize * DENSITY_WORD_COUNT, shadowMapSize];
    img_first_blocked_slice_map.dim = [shadowMapSize, shadowMapSize];
    comp_clear_shadow_map.workgroups = [Math.ceil((shadowMapSize * DENSITY_WORD_COUNT) / 16), Math.ceil(shadowMapSize / 16), 1];
    comp_build_density_metadata.workgroups = [Math.ceil(shadowMapSize / 16), Math.ceil(shadowMapSize / 16), 1];
    comp_build_density_metadata.param("DENSITY_WORD_COUNT", DENSITY_WORD_COUNT);
    comp_build_density_metadata.param("DENSITY_BITS_PER_WORD", 32);
    comp_build_density_metadata.param("DENSITY_SLICE_COUNT", DENSITY_SLICE_COUNT);
    comp_render_shadow.param("DENSITY_WORD_COUNT", DENSITY_WORD_COUNT);
    comp_render_shadow.param("DENSITY_BITS_PER_WORD", 32);
    comp_render_shadow.param("DENSITY_SLICE_COUNT", DENSITY_SLICE_COUNT);
    comp_prepare_eye_particles.param("DENSITY_WORD_COUNT", DENSITY_WORD_COUNT);
    comp_prepare_eye_particles.param("DENSITY_BITS_PER_WORD", 32);
    comp_prepare_eye_particles.param("DENSITY_SLICE_COUNT", DENSITY_SLICE_COUNT);
    comp_prepare_eye_particles.param("SHADOW_BLUR_SCALE", shadowBlurScale);
    comp_prepare_eye_particles.param("AMBIENT_OCCLUSION_CACHE_RESOLUTION", ambientOcclusionCacheResolution);
    comp_prepare_eye_particles.param("AMBIENT_LIGHT", ambientLight);
    comp_build_ambient_occlusion_cache.param("DENSITY_WORD_COUNT", DENSITY_WORD_COUNT);
    comp_build_ambient_occlusion_cache.param("DENSITY_BITS_PER_WORD", 32);
    comp_build_ambient_occlusion_cache.param("DENSITY_SLICE_COUNT", DENSITY_SLICE_COUNT);
    comp_build_ambient_occlusion_cache.param("AMBIENT_OCCLUSION_STRENGTH", ambientOcclusionStrength);
    comp_build_ambient_occlusion_cache.param("AMBIENT_OCCLUSION_RADIUS", ambientOcclusionRadius);
    comp_build_ambient_occlusion_cache.param("AMBIENT_OCCLUSION_CACHE_SAMPLE_COUNT", ambientOcclusionCacheSampleCount);
    count(particleCount);
}

function setscale(x){
    scale = x;
}

function setshadow_blur_scale(x){
    shadowBlurScale = Math.max(0, x);
    comp_prepare_eye_particles.param("SHADOW_BLUR_SCALE", shadowBlurScale);
}

function setambient_occlusion_strength(x){
    ambientOcclusionStrength = Math.max(0, x);
    comp_build_ambient_occlusion_cache.param("AMBIENT_OCCLUSION_STRENGTH", ambientOcclusionStrength);
}

function setambient_occlusion_radius(x){
    ambientOcclusionRadius = Math.max(0, x);
    comp_build_ambient_occlusion_cache.param("AMBIENT_OCCLUSION_RADIUS", ambientOcclusionRadius);
}

function setambient_occlusion_samples(x){
    // Backward-compatible message name: this now controls the number of 3D
    // density-map samples used per cached AO probe, not per particle.
    ambientOcclusionCacheSampleCount = Math.max(1, Math.min(64, Math.floor(x)));
    comp_build_ambient_occlusion_cache.param("AMBIENT_OCCLUSION_CACHE_SAMPLE_COUNT", ambientOcclusionCacheSampleCount);
}

function setambient_occlusion_cache_samples(x){
    setambient_occlusion_samples(x);
}

function setambient_occlusion_cache_resolution(x){
    ambientOcclusionCacheResolution = Math.max(1, Math.floor(x));
    resize_ambient_occlusion_cache();
}

function setambient_occlusion_cache_history_weight(x){
    ambientOcclusionCacheHistoryWeight = Math.max(0.0, Math.min(1.0, x));
    comp_filter_ambient_occlusion_cache.param("AMBIENT_OCCLUSION_CACHE_HISTORY_WEIGHT", ambientOcclusionCacheHistoryWeight);
}

function setambient_occlusion_cache_feedback_weight(x){
    setambient_occlusion_cache_history_weight(x);
}

function resetambient_occlusion_cache_history(){
    reset_ambient_occlusion_cache_history();
}

function setambient(){
    let bkg = [arguments[0], arguments[1], arguments[2]];

    ambientLight = [bkg[0] + lightColor[0]*0.03, bkg[1] + lightColor[1]*0.03, bkg[2] + lightColor[2]*0.03];

    comp_prepare_eye_particles.param("AMBIENT_LIGHT", ambientLight);
    comp_composite_background.param("background", [bkg[0], bkg[1], bkg[2], 0.0]);
}

function setmotion_blur_strength(x){
    motionBlurStrength = Math.max(0, Math.min(1, x));
    comp_prepare_eye_particles.param("MOTION_BLUR_STRENGTH", motionBlurStrength);
}

function setmotion_blur_max_stretch(x){
    motionBlurMaxStretch = Math.max(0, x);
    comp_prepare_eye_particles.param("MOTION_BLUR_MAX_STRETCH", motionBlurMaxStretch);
}


function setlight_color() {
    if (arguments.length >= 3) {
        lightColor = [Math.max(0, arguments[0]), Math.max(0, arguments[1]), Math.max(0, arguments[2])];
    } else if (arguments.length == 1) {
        let x = Math.max(0, arguments[0]);
        lightColor = [x, x, x];
    }
    comp_prepare_eye_particles.param("LIGHT_COLOR", [lightColor[0], lightColor[1], lightColor[2], 0.0]);
    comp_composite_background.param("flareColorStrength", [lightColor[0], lightColor[1], lightColor[2], lightFlareStrength]);
}

function setlight_flare_strength(x) {
    lightFlareStrength = Math.max(0, x);
    comp_composite_background.param("flareColorStrength", [lightColor[0], lightColor[1], lightColor[2], lightFlareStrength]);
}

function setlight_flare_core_radius(x) {
    lightFlareCoreRadius = Math.max(0.0001, x);
    comp_composite_background.param("flareShape", [lightFlareCoreRadius, lightFlareHaloRadius, RATIO, lightFlareHaloStrength]);
}

function setlight_flare_halo_radius(x) {
    lightFlareHaloRadius = Math.max(0.0001, x);
    comp_composite_background.param("flareShape", [lightFlareCoreRadius, lightFlareHaloRadius, RATIO, lightFlareHaloStrength]);
}

function setlight_flare_halo_strength(x) {
    lightFlareHaloStrength = Math.max(0, x);
    comp_composite_background.param("flareShape", [lightFlareCoreRadius, lightFlareHaloRadius, RATIO, lightFlareHaloStrength]);
}

function mulMat4Vec4(mat, v) {
    return [
        mat[0] * v[0] + mat[4] * v[1] + mat[8]  * v[2] + mat[12] * v[3],
        mat[1] * v[0] + mat[5] * v[1] + mat[9]  * v[2] + mat[13] * v[3],
        mat[2] * v[0] + mat[6] * v[1] + mat[10] * v[2] + mat[14] * v[3],
        mat[3] * v[0] + mat[7] * v[1] + mat[11] * v[2] + mat[15] * v[3]
    ];
}

function update_background_environment_camera() {
    let worldUp = [0, 1, 0];
    let right = normalizeVec3(cross(viewDir, worldUp));

    // Handle the rare case where the camera points almost exactly along worldUp.
    if (Math.hypot(right[0], right[1], right[2]) < 1e-6) {
        right = normalizeVec3(cross(viewDir, [0, 0, 1]));
    }

    let cameraUp = normalizeVec3(cross(right, viewDir));
    let tanHalfFov = Math.tan((lensAngle * Math.PI / 180.0) * 0.5);

    comp_composite_background.param("cameraForwardTanHalfFov", [viewDir[0], viewDir[1], viewDir[2], tanHalfFov]);
    comp_composite_background.param("cameraRightAspect", [right[0], right[1], right[2], RATIO]);
    comp_composite_background.param("cameraUp", [cameraUp[0], cameraUp[1], cameraUp[2], 0.0]);
}

function update_light_flare(matrices) {
    // lightDir points along light travel, so -lightDir points toward the visible light source.
    let toLight = [-lightDir[0], -lightDir[1], -lightDir[2]];
    let flarePoint = [
        pos[0] + toLight[0] * 1000.0,
        pos[1] + toLight[1] * 1000.0,
        pos[2] + toLight[2] * 1000.0
    ];
    let clip = mulMat4Vec4(matrices.VP, [flarePoint[0], flarePoint[1], flarePoint[2], 1.0]);

    let visibility = 0.0;
    let uvx = 0.5;
    let uvy = 0.5;

    if (clip[3] > 1e-6) {
        let ndcX = clip[0] / clip[3];
        let ndcY = clip[1] / clip[3];
        uvx = ndcX * 0.5 + 0.5;
        uvy = 1.0 - (ndcY * 0.5 + 0.5);
        visibility = 1.0;
    }

    comp_composite_background.param("flareCenterVisibility", [uvx, uvy, visibility, 0.0]);
    comp_composite_background.param("flareShape", [lightFlareCoreRadius, lightFlareHaloRadius, RATIO, lightFlareHaloStrength]);
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

function setlight_fit_padding(x) {
    lightFitPadding = Math.max(0.0, x);
    comp_compute_light_slice_range.param("LIGHT_FIT_PADDING", lightFitPadding);
}

function count(N) {

    particleCount = Math.max(0, Math.floor(N));

    const SIZE_OF_UINT = 4;
    const PARTICLE_STRIDE_BYTES = 64;

    const particleBytes = Math.max(1, particleCount) * PARTICLE_STRIDE_BYTES;

    comp_generate_position.workgroups = [Math.max(1, Math.ceil(particleCount / 256)), 1, 1];

    buff_particles.bytecount = particleBytes;
    buff_sorted_indices.bytecount = Math.max(1, particleCount) * SIZE_OF_UINT;
    buff_particle_draw.bytecount = Math.max(1, particleCount) * 64;

    buff_radix_histogram.bytecount = RADIX_BINS * SIZE_OF_UINT;
    buff_radix_cursor.bytecount = RADIX_BINS * SIZE_OF_UINT;
    buff_radix_bin_offsets.bytecount = RADIX_BINS * SIZE_OF_UINT;
    buff_radix_block_sums.bytecount = RADIX_BLOCK_COUNT * SIZE_OF_UINT;
    buff_radix_block_offsets.bytecount = RADIX_BLOCK_COUNT * SIZE_OF_UINT;
    buff_sort_range.bytecount = 2 * SIZE_OF_UINT;
    buff_light_slice_range.bytecount = 4 * 4; // vec4: lightMin, lightMax, invRange, unused
    buff_light_matrices.bytecount = 16 * 4 * 2; // mat4 lightP + mat4 lightVP
    buff_particle_aabb.bytecount = 2 * 4 * 4; // two vec4s: decoded AABB min/max
    buff_previous_particle_aabb.bytecount = 2 * 4 * 4; // previous-frame AABB used for temporal cache reprojection

    const AABB_RECORD_STRIDE_BYTES = 2 * 4 * 4;
    aabbReductionInitialRecordCount = Math.max(1, Math.ceil(particleCount / 256));
    buff_aabb_reduce_a.bytecount = aabbReductionInitialRecordCount * AABB_RECORD_STRIDE_BYTES;
    buff_aabb_reduce_b.bytecount = aabbReductionInitialRecordCount * AABB_RECORD_STRIDE_BYTES;
    resize_ambient_occlusion_cache();

    comp_generate_position.param("COUNT", particleCount);
    comp_generate_position.param("SQRT_COUNT", Math.ceil(Math.sqrt(Math.max(1, particleCount))));

    comp_radix_find_key_range.workgroups = [Math.max(1, Math.ceil(particleCount / 256)), 1, 1];
    comp_radix_find_key_range.param("COUNT", particleCount);

    comp_reduce_particle_aabb_from_particles.workgroups = [aabbReductionInitialRecordCount, 1, 1];
    comp_reduce_particle_aabb_from_particles.param("COUNT", particleCount);

    comp_radix_build_histogram.workgroups = [Math.max(1, Math.ceil(particleCount / 256)), 1, 1];
    comp_radix_build_histogram.param("COUNT", particleCount);

    comp_radix_scatter_particles.workgroups = [Math.max(1, Math.ceil(particleCount / 256)), 1, 1];
    comp_radix_scatter_particles.param("COUNT", particleCount);

    sliceSize = Math.ceil(particleCount / sliceCount);

    comp_render_shadow.param("COUNT", particleCount);
    comp_render_shadow.param("SLICE_SIZE", sliceSize);
    comp_render_shadow.workgroups = [Math.ceil(particleCount / 256), 1, 1];

    comp_prepare_eye_particles.param("pc.SLICE_SIZE", sliceSize);

    comp_prepare_eye_particles.workgroups = [Math.max(1, Math.ceil(particleCount / 256)), 1, 1];

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
    // Use the look-at vector as the canonical camera-forward direction for
    // front-to-back blending; this avoids depending on proxy-specific sign conventions.
    viewDir = normalizeVec3(subVec3(at, pos));

    let up = [0, 1, 0];

    let matrices = {
        V: m.lookAt(pos, at, up),
        P: m.perspective(lensAngle, RATIO, nearClip, farClip)
    };

    matrices.VP = m.mulMat4(matrices.P, matrices.V);
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

function reduce_particle_aabb(){
    // Pass 0: particle positions -> one AabbRecord per workgroup in buffer A.
    comp_reduce_particle_aabb_from_particles.bang();

    let recordCount = aabbReductionInitialRecordCount;
    let finalInA = 1;

    // Passes 1..N: reduce AabbRecords until a single record remains.
    while(recordCount > 1){
        const outputRecordCount = Math.max(1, Math.ceil(recordCount / 256));
        comp_reduce_particle_aabb_records.workgroups = [outputRecordCount, 1, 1];
        comp_reduce_particle_aabb_records.param("COUNT", recordCount);
        comp_reduce_particle_aabb_records.param("READ_FROM_A", finalInA);
        comp_reduce_particle_aabb_records.bang();

        recordCount = outputRecordCount;
        finalInA = 1 - finalInA;
    }

    aabbReductionFinalInA = finalInA;
}

function compute_light_slice_range(){
    comp_compute_light_slice_range.param("lightDir", lightDir);
    comp_compute_light_slice_range.param("AABB_REDUCTION_FINAL_IN_A", aabbReductionFinalInA);
    comp_compute_light_slice_range.param("LIGHT_FIT_PADDING", lightFitPadding);
    comp_compute_light_slice_range.bang();
}

function build_ambient_occlusion_cache(matrices, frame) {
    update_ambient_occlusion_cache_offsets_for_frame();
    comp_build_ambient_occlusion_cache.param("lightDir", lightDir);
    comp_build_ambient_occlusion_cache.param("AMBIENT_OCCLUSION_STRENGTH", ambientOcclusionStrength);
    comp_build_ambient_occlusion_cache.param("AMBIENT_OCCLUSION_RADIUS", ambientOcclusionRadius);
    comp_build_ambient_occlusion_cache.param("AMBIENT_OCCLUSION_CACHE_RESOLUTION", ambientOcclusionCacheResolution);
    comp_build_ambient_occlusion_cache.param("AMBIENT_OCCLUSION_CACHE_SAMPLE_COUNT", ambientOcclusionCacheSampleCount);
    comp_build_ambient_occlusion_cache.param("frame", frame);
    comp_build_ambient_occlusion_cache.bang();

    comp_filter_ambient_occlusion_cache.param("AMBIENT_OCCLUSION_CACHE_RESOLUTION", ambientOcclusionCacheResolution);
    comp_filter_ambient_occlusion_cache.param("AMBIENT_OCCLUSION_CACHE_HISTORY_WEIGHT", ambientOcclusionCacheHistoryWeight);
    comp_filter_ambient_occlusion_cache.param("AMBIENT_OCCLUSION_CACHE_HISTORY_VALID", ambientOcclusionCacheHistoryValid);
    comp_filter_ambient_occlusion_cache.bang();

    // This frame's filtered cache is now complete. Persist the AABB that defines
    // its probe lattice so the next frame can reproject history consistently.
    comp_store_previous_particle_aabb.bang();
}

function advance_ambient_occlusion_cache_history() {
    ambientOcclusionCacheHistoryValid = 1;
    ambientOcclusionCacheWriteIndex = 1 - ambientOcclusionCacheWriteIndex;
}

function clear_color_attachments(){

    comp_clear_color_target.bang();
    comp_clear_shadow_map.bang();
}

function prepare_eye_particles(matrices) {
    comp_prepare_eye_particles.param("VP", matrices.VP);
    comp_prepare_eye_particles.param("P", matrices.P);
    comp_prepare_eye_particles.param("lightDir", lightDir);
    comp_prepare_eye_particles.param("ALPHA", ALPHA);
    comp_prepare_eye_particles.param("MOTION_BLUR_STRENGTH", motionBlurStrength);
    comp_prepare_eye_particles.param("MOTION_BLUR_MAX_STRETCH", motionBlurMaxStretch);
    comp_prepare_eye_particles.param("ASPECT", RATIO);
    comp_prepare_eye_particles.param("SHADOW_BLUR_SCALE", shadowBlurScale);
    comp_prepare_eye_particles.param("AMBIENT_OCCLUSION_CACHE_RESOLUTION", ambientOcclusionCacheResolution);
    comp_prepare_eye_particles.param("AMBIENT_LIGHT", ambientLight);
    comp_prepare_eye_particles.param("pc.COUNT", particleCount);
    comp_prepare_eye_particles.param("pc.SLICE_SIZE", sliceSize);
    comp_prepare_eye_particles.param("opacityMultiplier", 4*ALPHA / DENSITY_WORD_COUNT);
    comp_prepare_eye_particles.bang();
}

let frame = 0;

function bang() {

    let transfrom = calc_matrices();
    update_background_environment_camera();
    update_light_flare(transfrom);

    comp_generate_position.param("TIME", TIME * 0.1);
    comp_generate_position.param("SCALE", scale);
    comp_generate_position.param("SORT_VECTOR", viewDir);
    comp_generate_position.param("CAMERA_POS", pos);
    comp_generate_position.bang();

    // Compute the current particle AABB with a multi-pass reduction instead of
    // six contended global atomics per particle.
    reduce_particle_aabb();
    compute_light_slice_range();

    clear_color_attachments();

    comp_render_shadow.param("lightDir", lightDir);
    comp_render_shadow.bang();

    // Build the shadow metadata in one traversal of the packed density words:
    //  - cumulative per-word bit-count prefixes for fast eye-pass shadow reads;
    //  - closest-to-light blocker slice for blur-radius estimation.
    comp_build_density_metadata.bang();

    // Populate the cached cubic AO probe volume once per frame. The eye pass then
    // performs only an 8-probe trilinear interpolation per particle.
    build_ambient_occlusion_cache(transfrom, frame);

    // Sort order is still built by the radix path. The particle AABB has already
    // been reduced above through the dedicated atomics-free reduction pipeline.
	sort_particles();

	if(display_probes){
	    draw_probes.param("V", transfrom.V);
	    draw_probes.param("P", transfrom.P);
	    draw_probes.param("res", ambientOcclusionCacheResolution);
	    render_particles.jit_gpu_draw(draw_probes.name);		
	}

    prepare_eye_particles(transfrom);
    advance_ambient_occlusion_cache_history();
    render_particles.jit_gpu_draw(draw_particles.name);

    render_particles.bang();

    //composite background
    comp_composite_background.bang();

    outlet(0, "source", img_color_target.name);
    outlet(0, "bang");
    frame++;
}
