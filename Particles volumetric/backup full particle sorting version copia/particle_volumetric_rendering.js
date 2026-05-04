autowhatch = 1; inlets = 1; outlets = 1;

const m = require("Patcher://vector_math.js");

let TIME = 0;
let VIEWPORT = [1920, 1080];
let RATIO = VIEWPORT[0] / VIEWPORT[1];
let RADIUS = 0.01;
let ALPHA = 0.2;
let pos, at, farClip, nearClip, lensAngle, viewDir;
let numStages;
let sliceSize;
let particleCount;
const sliceCount = 64;


let lightDir = normalizeVec3([1,1,1]);
let halfVector, flipped;

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

let buff_slice = new JitterObject("jit.gpu.buffer");
buff_slice.bytecount = 4 * 4;

let comp_set_slice = new JitterObject("jit.gpu.compute");
comp_set_slice.shader = "comp_set_slice.comp";
comp_set_slice.workgroups = [1,1,1];
comp_set_slice.bind("buff_slice", buff_slice.name);

let buff_inside_counter = new JitterObject("jit.gpu.buffer"); // indirect draw args / visible count
buff_inside_counter.bytecount = 4 * 4;

let buff_inside = new JitterObject("jit.gpu.buffer"); // per-particle visibility flag
let buff_visible_indices = new JitterObject("jit.gpu.buffer");

let comp_reset_inside_counter = new JitterObject("jit.gpu.compute");
comp_reset_inside_counter.file = "comp_reset_inside_counter.comp";
comp_reset_inside_counter.workgroups = [1, 1, 1];
comp_reset_inside_counter.bind("buff_inside_counter", buff_inside_counter.name);

let comp_generate_position = new JitterObject("jit.gpu.compute");
comp_generate_position.file = "comp_generate_position.comp";
comp_generate_position.bind("buff_particles", buff_particles.name);
comp_generate_position.param("RADIUS", RADIUS);

var comp_sort = new JitterObject("jit.gpu.compute");
comp_sort.shader = "comp_sort.comp";
comp_sort.bind("buff_particles", buff_particles.name);

let img_color_target = new JitterObject("jit.gpu.image");
img_color_target.dim = [VIEWPORT[0], VIEWPORT[1]];
img_color_target.format = "rgba32_float";

let shadowMapSize = 512;
let img_shadow_map = new JitterObject("jit.gpu.image");
img_shadow_map.dim = [shadowMapSize, shadowMapSize];
img_shadow_map.format = "r32_float";

let comp_clear_color_target = new JitterObject("jit.gpu.compute");
comp_clear_color_target.shader = "comp_clear_color_target.comp";
comp_clear_color_target.bind("img_color_target", img_color_target.name);
comp_clear_color_target.workgroups = [Math.ceil(VIEWPORT[0] / 16), Math.ceil(VIEWPORT[1] / 16), 1];

let comp_clear_shadow_map = new JitterObject("jit.gpu.compute");
comp_clear_shadow_map.shader = "comp_clear_shadow_map.comp";
comp_clear_shadow_map.bind("img_shadow_map", img_shadow_map.name);
comp_clear_shadow_map.workgroups = [Math.ceil(shadowMapSize / 16), Math.ceil(shadowMapSize / 16), 1];

// shadow pass

let draw_shadow = new JitterObject("jit.gpu.draw");
draw_shadow.shader = "draw_shadow.rend";
draw_shadow.vb0 = buff_quad.name;
draw_shadow.buff_particles = buff_particles.name;
draw_shadow.indirect = buff_slice.name;
draw_shadow.buff_inside_counter = buff_inside_counter.name;
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
draw_particles.buff_particles = buff_particles.name;
draw_particles.indirect = buff_slice.name;
draw_particles.buff_inside_counter = buff_inside_counter.name;
draw_particles.img_shadow_map = img_shadow_map.name;
draw_particles.img_prev_slice = img_color_target.name;
draw_particles.topology = "trianglestrip";
draw_particles.elemcount = 4;
draw_particles.blendenable = true;//true;
draw_particles.depth_write = false;
draw_particles.param("ALPHA", ALPHA);
//draw_particles.blendcolordst = "one";
//draw_particles.blendcolorsrc = "inv_src_color";

let render_particles = new JitterObject("jit.gpu.render");
render_particles.colorattachments = 1;
render_particles.depth = false;
render_particles.colorimg0 = img_color_target.name;
//render_particles.clearcolor0 = [0.2, 0.2, 0.2, 0.0];
render_particles.msaa = 1;
render_particles.colorloadop0 = "load";


count(100000);

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
function dot(a, b){ return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }

function light_direction(){
    lightDir = normalizeVec3([arguments[0], arguments[1], arguments[2]]);
}

function compute_half_vector(){

    // The paper's prose describes the branch as positive vs. negative dot product.
    // Its pseudocode says > 1.0, but normalized vectors cannot exceed 1.0.
    if(dot(viewDir, lightDir) > 0.0) {
        halfVector = normalizeVec3(sumVec3(viewDir, lightDir));
        flipped = true;
    } else {
        halfVector = normalizeVec3(sumVec3(mulVec3Float(viewDir, -1.0), mulVec3Float(lightDir, -1.0)));
        flipped = false;
    }
}

function nextPowerOfTwo(N){ //bit-twiddling
    if (N <= 1){ return 1; }
    N--;
    N |= N >> 1; N |= N >> 2; N |= N >> 4; N |= N >> 8; N |= N >> 16;
    return N + 1;
}

function count(N) {

    particleCount = N;

    let np2 = nextPowerOfTwo(N);
    numStages = Math.log2(np2);
    
    comp_sort.workgroups = [Math.ceil((np2 / 2) / 128), 1, 1];
    comp_sort.param("numValues", N);

    const SIZE_OF_FLOAT = 4;
    const SIZE_OF_UINT = 4;

    comp_generate_position.workgroups   = [Math.ceil(N / 256), 1, 1];

    // Particle and visibility buffers track the requested count exactly.
    buff_particles.bytecount = N * 48;//N * (SIZE_OF_FLOAT * 9 + SIZE_OF_UINT);
    buff_inside.bytecount = N * SIZE_OF_UINT;
    buff_visible_indices.bytecount = N * SIZE_OF_UINT;

    // Set params for existing passes.
    comp_generate_position.param("COUNT", N);
    comp_generate_position.param("SQRT_COUNT", Math.ceil(Math.sqrt(Math.max(1, N))));

    // Indirect draw uses buff_inside_counter.instanceCount. This is just the
    // non-indirect upper bound/debug fallback.
    draw_particles.instancecount = Math.ceil(N / sliceCount);
    draw_shadow.instancecount = Math.ceil(N / sliceCount);

    sliceSize = Math.ceil(N / sliceCount);
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

function sort_particles(){

    for(let stageIndex = 0; stageIndex < numStages; stageIndex++){
        for(let stepIndex = 0; stepIndex < stageIndex + 1; stepIndex++){
            
            let groupWidth = 1 << (stageIndex - stepIndex);
            let groupHeight = 2 * groupWidth - 1;
            comp_sort.param("groupWidth", groupWidth);
            comp_sort.param("groupHeight", groupHeight);
            comp_sort.param("stepIndex", stepIndex);
            comp_sort.bang();
        }
    }
}

function clear_color_attachments(){

    comp_clear_color_target.bang();
    comp_clear_shadow_map.bang();
}

function bang() {

    let transfrom = calc_matrices();
    compute_half_vector();

    comp_generate_position.param("TIME", TIME*0.1);
    comp_generate_position.param("SCALE", 0.0015);
    comp_generate_position.param("HALF_VECTOR", halfVector);
    comp_generate_position.param("FLIPPED", flipped);
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

    //sliced rendering
    for (let si = 0; si < sliceCount; si++) {
        let i = flipped ? (sliceCount - 1 - si) : si;
        let offset = sliceSize * i;

        let count = Math.min(particleCount, offset + sliceSize) - offset;
        if (count <= 0) continue;

        comp_set_slice.param("pc.instanceCount", count);
        comp_set_slice.param("pc.firstInstance", offset);
        comp_set_slice.bang();

        // Eye pass: uses shadow/light buffer accumulated from previous slices.
        render_particles.jit_gpu_draw(draw_particles.name);
        render_particles.bang();

        // Shadow pass: updates light buffer with this slice for later slices.
        render_shadow.jit_gpu_draw(draw_shadow.name);
        render_shadow.bang();
    }

    outlet(0, "source", img_color_target.name);
    outlet(0, "bang");
}
