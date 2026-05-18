

const uni = new Uniforms();
uni.addUniform("resolution", "vec2f");    // canvas resolution: x-width, y-height
uni.addUniform("fftsize", "vec2f");    // FFT size
uni.finalize();

const textures = {
  stateTex0: null,
  stateTex1: null,
  energyTex: null,
  speedTex: null,
};

const canvas = document.getElementById("canvas");

const gui = new GUI("2D FFT", canvas);
const imgSize = [1024, 1024];

// Performance section
{
  gui.addGroup("perf", "Performance");
  gui.addStringOutput("res", "Resolution", "", "perf");
  gui.addHalfWidthGroups("perfL", "perfR", "perf");
  gui.addNumericOutput("fps", "FPS", "", 1, "perfL");
  gui.addNumericOutput("frameTime", "Frame", "ms", 2, "perfL");
  gui.addNumericOutput("jsTime", "JS", "ms", 2, "perfL");
  gui.addNumericOutput("computeTime", "Compute", "ms", 2, "perfR");
  gui.addNumericOutput("renderTime", "Render", "ms", 2, "perfR");
}


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