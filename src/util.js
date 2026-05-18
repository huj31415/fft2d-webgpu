

const uni = new Uniforms();
uni.addUniform("resolution", "vec2f");    // canvas resolution: x-width, y-height
uni.addUniform("start_L", "f32");
uni.addUniform("end_L", "f32");
uni.addUniform("steps", "f32");
uni.addUniform("ref_L", "f32");
uni.addUniform("dispMult", "f32");

uni.finalize();

const storage = {
  sourceTex: null,
  inputTex: null,
  rowFreqTex: null,
  colFreqTex: null,
  finalTex: null,
};

const canvas = document.getElementById("canvas");

const gui = new GUI("2D FFT", canvas);
const imgSize = [1024, 1024];

// Performance section
gui.addGroup("perf", "Performance");
gui.addStringOutput("res", "Resolution", "", "perf");
gui.addHalfWidthGroups("perfL", "perfR", "perf");
gui.addNumericOutput("fps", "FPS", "", 1, "perfL");
gui.addNumericOutput("frameTime", "Frame", "ms", 2, "perfL");
gui.addNumericOutput("jsTime", "JS", "ms", 2, "perfL");
gui.addNumericOutput("computeTime", "Compute", "ms", 2, "perfR");
gui.addNumericOutput("renderTime", "Render", "ms", 2, "perfR");

gui.addGroup("dispersion", "Dispersion settings");
gui.addNumericInput("start_L", true, "Start wavelength", { min: 360, max: 820, step: 1, val: 380, float: 0 }, "dispersion", (value) => uni.values.start_L.set([value]));
gui.addNumericInput("end_L", true, "End wavelength", { min: 370, max: 830, step: 1, val: 780, float: 0 }, "dispersion", (value) => uni.values.end_L.set([value]));
gui.addNumericInput("steps", true, "Steps", { min: 8, max: 2048, step: 8, val: 512, float: 0 }, "dispersion", (value) => uni.values.steps.set([value]));
gui.addNumericInput("ref_L", true, "Ref. wavelength", { min: 360, max: 830, step: 5, val: 550, float: 0 }, "dispersion", (value) => uni.values.ref_L.set([value]));
gui.addNumericInput("dispMult", true, "Strength", { min: 0, max: 2, step: 0.05, val: 1, float: 2 }, "dispersion", (value) => uni.values.dispMult.set([value]));


// Extra info
gui.addGroup("guiControls", "GUI controls", `
  <div>
    Click on section titles to expand/collapse
    <br>
    Hover on input labels for more info if applicable
    <br>
    Click to toggle between raw number and slider type input
    <br>
  </div>
`);


// requestAnimationFrame id, fps update id
let rafId, perfIntId;


// timing
let jsTime = 0, lastFrameTime = performance.now(), deltaTime = 10, fps = 0,
  computeTime = 0, renderTime = 0;

// handle resizing
window.onresize = window.onload = () => {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  uni.values.resolution.set([canvas.width, canvas.height]);
  gui.io.res([window.innerWidth, window.innerHeight]);
};