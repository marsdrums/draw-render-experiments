const m = require("Patcher://math_utils.js");

let time = 0;
function setTime(t) {
    time = t;
}

function setupCamera(name) {
    drawName = name;

    const rad = 3;
    const x = Math.sin(time) * rad;
    const z = Math.cos(time) * rad;

    // post("camPos:", [x, 0, z], "\n");

    let view = m.lookAt([x, 0, z], [0, 0, 0], [0, 1, 0]);
    let normal = m.normalMatrix4FromMat4(view);
    let projection = m.perspective(60, 1920.0/1080.0, 0.1, 100);
    let mvp = m.mulMat4(projection, view);

    // post("normalMatrix:", normal, "\n");
    // post("mvp:", mvp, "\n");

    let proxy = new JitterObject("jit.proxy");
    proxy.name = name;
    proxy.send("lightDir", m.normalizeVec3([-1, -1, 0]));
    proxy.send("param", "mvp", mvp);
    proxy.send("param", "normalMatrix", normal);
}

