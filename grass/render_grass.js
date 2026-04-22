autowhatch = 1; inlets = 2; outlets = 2;

const m = require("Patcher://vector_math.js");
const SIZE_OF_UINT = 4;
const WG_SIZE = 256;
const GROUND_SIZE = 100;
const GROUND_VERTICES = 64;
let _debug = 0;
let time = 0;

let viewport = [1920, 1080];
let ratio = viewport[0] / viewport[1];

let proxyCam = new JitterObject("jit.proxy");

let numOfInstances = 3000000;
let instanceMatrix = null;
let gpuData = null;

let img_env = new JitterObject("jit.gpu.image");

let mat_grass = new JitterMatrix();
mat_grass.read("grass_blade.jxf");
post("dim", mat_grass.dim);

// find grass model min/max
let mm = [1e10, 1e10, 1e10];
let MM = [-1e10, -1e10, -1e10];
let val;

for (let i = 0; i < mat_grass.dim; i++) {
    val = mat_grass.getcell(i).slice(0,3);
    mm = [Math.min(mm[0], val[0]), Math.min(mm[1], val[1]), Math.min(mm[2], val[2])];
    MM = [Math.max(MM[0], val[0]), Math.max(MM[1], val[1]), Math.max(MM[2], val[2])];
}

let bladeHeightLocal = MM[1];
let bendPad = bladeHeightLocal * bladeHeightLocal * 0.55;

let mat_reset = new JitterMatrix(1, "long", 4);
mat_reset.setall(0);

let num_ground_triangles = 2 * Math.pow(GROUND_VERTICES - 1, 2);
let buff_vert_ground = new JitterObject("jit.gpu.buffer");
buff_vert_ground.bytecount = num_ground_triangles * 16 * 3; //num of triangles * sizeOFVec3 * 3

let buff_grass_vertex = new JitterObject("jit.gpu.buffer");
buff_grass_vertex.jit_matrix(mat_grass.name);

let img_terrain = new JitterObject("jit.gpu.image");
img_terrain.format = "r32_float";
img_terrain.dim = [GROUND_VERTICES, GROUND_VERTICES];

let img_perlin = new JitterObject("jit.gpu.image");
img_perlin.format = "rg32_float";
img_perlin.dim = [512, 512];

let img_cloud = new JitterObject("jit.gpu.image");
img_cloud.format = "r32_float";
img_cloud.dim = [128, 128];

let img_fake_shadow = new JitterObject("jit.gpu.image");
img_fake_shadow.format = "r32_float";
img_fake_shadow.dim = [64, 64];

let buff_grass_instanceinfo = new JitterObject("jit.gpu.buffer");

let buff_inside = new JitterObject("jit.gpu.buffer");
buff_inside.bytecount = numOfInstances * SIZE_OF_UINT;

let buff_visible_indices = new JitterObject("jit.gpu.buffer");
buff_visible_indices.bytecount = numOfInstances * SIZE_OF_UINT;

let buff_indirect = new JitterObject("jit.gpu.buffer");
buff_indirect.bytecount = SIZE_OF_UINT * 4;

// generate the terrain
let comp_terrain = new JitterObject("jit.gpu.compute");
comp_terrain.file = "comp_terrain.comp";
comp_terrain.bind("img_terrain", img_terrain.name);
comp_terrain.workgroups = [Math.ceil(img_terrain.dim[0] / 16), Math.ceil(img_terrain.dim[1] / 16), 1];
comp_terrain.param("scale", 5.0);
comp_terrain.param("amt", 10.0);
comp_terrain.bang();

// generate fake shadow map
let comp_fake_shadow = new JitterObject("jit.gpu.compute");
comp_fake_shadow.file = "comp_fake_shadow.comp";
comp_fake_shadow.bind("img_fake_shadow", img_fake_shadow.name);
comp_fake_shadow.workgroups = [Math.ceil(img_fake_shadow.dim[0] / 16), Math.ceil(img_fake_shadow.dim[1] / 16), 1];

// change grass elevation according to the terrain
let comp_elevate_grass = new JitterObject("jit.gpu.compute");
comp_elevate_grass.file = "comp_elevate_grass.comp";
comp_elevate_grass.bind("buff_grass_instanceinfo", buff_grass_instanceinfo.name);
comp_elevate_grass.bind("img_terrain", img_terrain.name);
comp_elevate_grass.param("ground_size", GROUND_SIZE);
comp_elevate_grass.param("numOfInstances", numOfInstances);
comp_elevate_grass.workgroups = [Math.ceil(numOfInstances / 256), 1, 1];

// triangulate the height map
let comp_ground = new JitterObject("jit.gpu.compute");
comp_ground.file = "comp_ground.comp";
comp_ground.bind("buff_vert_ground", buff_vert_ground.name);
comp_ground.bind("img_terrain", img_terrain.name);
comp_ground.param("ground_size", GROUND_SIZE);
let num_cell_x = img_terrain.dim[0] - 1;
let num_cell_y = img_terrain.dim[1] - 1;
comp_ground.workgroups = [Math.ceil(num_cell_x / 16), Math.ceil(num_cell_y / 16), 1]; //one invocation for each square


// compute vector wind field
let comp_wind = new JitterObject("jit.gpu.compute");
comp_wind.file = "comp_wind.comp";
comp_wind.workgroups = [Math.ceil(img_perlin.dim[0] / 16), Math.ceil(img_perlin.dim[1] / 16), 1];
comp_wind.bind("img_perlin", img_perlin.name);

// compute clouds
let comp_cloud = new JitterObject("jit.gpu.compute");
comp_cloud.file = "comp_cloud.comp";
comp_cloud.workgroups = [Math.ceil(img_cloud.dim[0] / 16), Math.ceil(img_cloud.dim[1] / 16), 1];
comp_cloud.bind("img_cloud", img_cloud.name);

// compute frustum culling
let comp_inside_frustum = new JitterObject("jit.gpu.compute");
comp_inside_frustum.file = "comp_inside_frustum.comp";
comp_inside_frustum.bind("buff_grass_instanceinfo", buff_grass_instanceinfo.name);
comp_inside_frustum.bind("buff_inside", buff_inside.name);
comp_inside_frustum.param("numOfInstances", numOfInstances);
comp_inside_frustum.param("modelMin", mm);
comp_inside_frustum.param("modelMax", MM);
comp_inside_frustum.param("bendPaddingLocal", [bendPad, bendPad * 0.6, bendPad]);

// compact the visible grass blades
let comp_compact_inside = new JitterObject("jit.gpu.compute");
comp_compact_inside.file = "comp_compact_inside.comp";
comp_compact_inside.bind("buff_inside", buff_inside.name);
comp_compact_inside.bind("buff_visible_indices", buff_visible_indices.name);
comp_compact_inside.bind("buff_indirect", buff_indirect.name);
comp_compact_inside.param("numOfInstances", numOfInstances);
comp_compact_inside.param("num_vertices", mat_grass.dim);

// Main pass
let img_color_target = new JitterObject("jit.gpu.image");
img_color_target.dim = [viewport[0], viewport[1]];
img_color_target.format = "rgba32_float";

let img_depth_target = new JitterObject("jit.gpu.image");
img_depth_target.dim = [viewport[0], viewport[1]];
img_depth_target.format = "d32_float";

let draw_env = new JitterObject("jit.gpu.draw");
draw_env.shader = "draw_env.rend";
draw_env.elemcount = 6;
draw_env.img_env = img_env.name;

let draw_ground = new JitterObject("jit.gpu.draw");
draw_ground.shader = "draw_ground.rend";
draw_ground.vb0 = buff_vert_ground.name;
draw_ground.elemcount = num_ground_triangles * 3;

let draw_grass = new JitterObject("jit.gpu.draw");
draw_grass.shader = "draw_grass.rend";
draw_grass.vb0 = buff_grass_vertex.name;
draw_grass.buff_grass_instanceinfo = buff_grass_instanceinfo.name;
draw_grass.buff_visible_indices = buff_visible_indices.name;
draw_grass.img_perlin = img_perlin.name;
draw_grass.img_cloud = img_cloud.name;
draw_grass.img_fake_shadow = img_fake_shadow.name;
draw_grass.indirect = buff_indirect.name;
draw_grass.elemcount = mat_grass.dim;
draw_grass.numOfInstances = numOfInstances;
draw_grass.param("ligDir", normalizeVec3([1,1,-0.5]));

let render_env = new JitterObject("jit.gpu.render");
render_env.colorattachments = 1;
render_env.depth = false;
render_env.colorimg0 = img_color_target.name;

let render = new JitterObject("jit.gpu.render");
render.colorattachments = 1;
render.depth = true;
render.colorimg0 = img_color_target.name;
render.clearcolor0 = [0.3, 0.5, 0.9, 1.0];
render.depthimg = img_depth_target.name;
render.msaa = 1;

initInstances(numOfInstances);

// callback for when the model is loaded
function jit_matrix(name) {
    if(inlet != 1) return;

    img_env.jit_matrix(name);
}

function debug(x) {
    _debug = x;
}

function set_time(x){
    time = x;
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

function calc_frustum(fov, ratio, near, far, pos, at) {
    const fovRad = fov * Math.PI / 180.0;

    const halfVSide = far * Math.tan(fovRad * 0.5);
    const halfHSide = halfVSide * ratio;

    const front = normalizeVec3(subVec3(at, pos));
    const right = normalizeVec3(cross(front, [0, 1, 0]));
    const up = normalizeVec3(cross(right, front));

    const frontMultNear = mulVec3Float(front, near);
    const frontMultFar = mulVec3Float(front, far);

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

function initInstances() {
    const INSTANCE_FLOATS = 36; // mat4 + mat4 + vec4 bend params

    gpuData = new Float32Array(numOfInstances * INSTANCE_FLOATS);
    instanceMatrix = new JitterMatrix(1, "float32", gpuData.length);

    for (let i = 0; i < numOfInstances; i++) {
        let offset = i * INSTANCE_FLOATS;

        let px = (Math.random() * 2 - 1) * GROUND_SIZE;
        let py = 0.0;
        let pz = (Math.random() * 2 - 1) * GROUND_SIZE;
        let angle = Math.random() * Math.PI * 2;
        let scale = [1.0, Math.random() * 1.5 + 0.3, 1.0];
        //let scale = [1,1,1];

        let T = m.mat4FromTranslation(new Float32Array([px, py, pz]));
        let R = m.mat4FromAxisAngle([0, 1, 0], angle);
        let S = m.mat4FromUniformScale(scale);
        let mdl = m.mulMat4(T, m.mulMat4(R, S));
        let normalMat = m.normalMatrix4FromMat4(mdl);

        let k1 = 0.045 + Math.random() * 0.015;
        let k2 = 0.080 + Math.random() * 0.030;
        let lambda = 0.22 + Math.random() * 0.12;
        let drag = 0.85 + Math.random() * 0.30;

        gpuData.set(mdl, offset);
        gpuData.set(normalMat, offset + 16);
        gpuData.set([k1, k2, lambda, drag], offset + 32);
    }

    instanceMatrix.copyarraytomatrix(gpuData);
    buff_grass_instanceinfo.jit_matrix(instanceMatrix.name);

    comp_elevate_grass.bang();
}

function name(n) {
    proxyCam.name = n;
}

function farPlaneCornersView(fovyDeg, aspect, far) {
    const t = Math.tan(fovyDeg * Math.PI / 180 / 2);
    const halfH = far * t;
    const halfW = halfH * aspect;
    return [ halfW,  halfH, -far];
}

function inverse(view) {
    const rx = view[0], tx = view[1], fx = view[2];
    const ry = view[4], ty = view[5], fy = view[6];
    const rz = view[8], tz = view[9], fz = view[10];

    const dx = view[12], dy = view[13], dz = view[14];

    const ex = -(rx * dx + tx * dy + fx * dz);
    const ey = -(ry * dx + ty * dy + fy * dz);
    const ez = -(rz * dx + tz * dy + fz * dz);

    return new Float32Array([
        rx, ry, rz, 0,
        tx, ty, tz, 0,
        fx, fy, fz, 0,
        ex, ey, ez, 1
    ]);
}

function bang() {
    //time += 0.65;

    comp_wind.param("time", time*0.5);
    comp_wind.param("scale", 7.0);
    comp_wind.param("amt", 3.);
    comp_wind.bang();

    comp_cloud.param("time", time*0.001);
    comp_cloud.param("scale", 0.2);
    comp_cloud.bang();

    comp_ground.bang(); //*** WHY DO I HAVE TO RUN THIS EVERY FRAME??

    comp_fake_shadow.param("scale", 10.0);
    comp_fake_shadow.param("amt", 1.0);
    comp_fake_shadow.param("time", time*0.01);
    comp_fake_shadow.bang();

    outlet(1, "source", img_cloud.name);
    outlet(1, "bang");

    let pos = proxyCam.send("getposition");
    let at = proxyCam.send("getlookat");

    let view = _debug == 1 ? m.lookAt([10, 10, 10], [0, 0, 0], [0, 1, 0]) : m.lookAt(pos, at, [0, 1, 0]);
    let projection = m.perspective(60, ratio, 0.1, 100);
    let mvp = m.mulMat4(projection, view);

    // reset GPU counter
    buff_indirect.jit_matrix(mat_reset.name);

    // pass 1: write 0/1 visibility flags
    calc_frustum(60, ratio, 0.1, 100, pos, at);
    comp_inside_frustum.workgroups = [Math.ceil(numOfInstances / WG_SIZE), 1, 1];
    comp_inside_frustum.bang();

    // pass 2: compact visible instance ids into buff_visible_indices
    comp_compact_inside.workgroups = [Math.ceil(numOfInstances / WG_SIZE), 1, 1];
    comp_compact_inside.bang();

    let farCorner = farPlaneCornersView(60, ratio, 100);
    let invV = inverse(view);
    draw_env.param("farCorner", ...Array.from(farCorner));
    draw_env.param("invV", ...Array.from(invV));
    render.jit_gpu_draw(draw_env.name);
    //render_env.bang();

    draw_ground.param("mvp", ...Array.from(mvp));
    render.jit_gpu_draw(draw_ground.name);
    draw_grass.param("mvp", ...Array.from(mvp));
    draw_grass.param("eye", ...Array.from(pos));
    render.jit_gpu_draw(draw_grass.name);

    render.bang();

    outlet(0, "source", img_color_target.name);
    outlet(0, "bang");
}
