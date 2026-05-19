

const uni = new Uniforms();
uni.addUniform("resRatio", "vec2f");    // canvas resolution: x-width, y-height
uni.addUniform("start_L", "f32");
uni.addUniform("end_L", "f32");
uni.addUniform("steps", "f32");
uni.addUniform("ref_L", "f32");
uni.addUniform("dispMult", "f32");
uni.addUniform("gain", "f32");
uni.addUniform("scale", "f32");

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
gui.addNumericOutput("computeTime", "FFT", "ms", 2, "perfR");
gui.addNumericOutput("dispersionTime", "Dispersion", "ms", 2, "perfR");
gui.addNumericOutput("renderTime", "Render", "ms", 2, "perfR");

gui.addGroup("dispersion", "Dispersion settings");
gui.addNumericInput("start_L", true, "Start wavelength", { min: 360, max: 820, step: 1, val: 380, float: 0 }, "dispersion", (value) => uni.values.start_L.set([value]));
gui.addNumericInput("end_L", true, "End wavelength", { min: 370, max: 830, step: 1, val: 780, float: 0 }, "dispersion", (value) => uni.values.end_L.set([value]));
gui.addNumericInput("steps", true, "Steps", { min: 8, max: 2048, step: 8, val: 512, float: 0 }, "dispersion", (value) => uni.values.steps.set([value]));
gui.addNumericInput("ref_L", true, "Ref. wavelength", { min: 360, max: 830, step: 5, val: 550, float: 0 }, "dispersion", (value) => uni.values.ref_L.set([value]));
gui.addNumericInput("dispMult", true, "Strength", { min: 0, max: 2, step: 0.05, val: 1, float: 2 }, "dispersion", (value) => uni.values.dispMult.set([value]));
gui.addNumericInput("gain", true, "Gain", { min: -5, max: 5, step: 0.1, val: 0, float: 1 }, "dispersion", (value) => uni.values.gain.set([10**value]));
gui.addNumericInput("scale", true, "Scale", { min: 0, max: 3, step: 0.01, val: 0, float: 2 }, "dispersion", (value) => uni.values.scale.set([10**value]));

gui.addDropdown("colorMatching", "Color matching", [
  "CIE 1931 2deg",
  "CIE 1964 10deg",
], "dispersion", null, async (value) => {
  const colorMatchingData = (value === "CIE 1931 2deg") ? cie1931_xyz_2deg_360_830 : cie1964_xyz_10deg_360_830;
  device.queue.writeTexture(
    { texture: storage.colorMatchingTex },
    colorMatchingData.buffer,
    {}, { width: colorMatchingData.length / 4 }
  );
});

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
  computeTime = 0, dispersionTime = 0, renderTime = 0;

// handle resizing
window.onresize = window.onload = () => {
  pixelRatio = window.devicePixelRatio || 1;
  // canvas.style.zoom = 1 / pixelRatio;
  canvas.width = window.innerWidth * pixelRatio;
  canvas.height = window.innerHeight * pixelRatio;
  // uni.values.resolution.set([canvas.width, canvas.height]);
  const invMinRes = 1 / Math.min(canvas.width, canvas.height);
  uni.values.resRatio.set([canvas.width * invMinRes, canvas.height * invMinRes]);
  gui.io.res([canvas.width, canvas.height]);
};


// let rgbmatrix = [
// // sRGB D65
//   // 3.2404542, -1.5371385, -0.4985314,
//   // -0.9692660,  1.8760108,  0.0415560,
//   // 0.0556434, -0.2040259,  1.0572252
// // CIE E
//  2.3706743, -0.9000405, -0.4706338,
// -0.5138850,  1.4253036,  0.0885814,
//  0.0052982, -0.0146949,  1.0093968
// ];

// const xyz2rgb = (x, y, z) => [
//   Math.max(0, rgbmatrix[0] * x + rgbmatrix[1] * y + rgbmatrix[2] * z),
//   Math.max(0, rgbmatrix[3] * x + rgbmatrix[4] * y + rgbmatrix[5] * z),
//   Math.max(0, rgbmatrix[6] * x + rgbmatrix[7] * y + rgbmatrix[8] * z),
// ];

// const lerp = (a, b, t) => a + (b - a) * t;
// const sampleCMF = (l) => {
//   const idx = Math.floor(l - 360) * 4;
//   return [
//     lerp(cie1931_xyz_2deg_360_830[idx], cie1931_xyz_2deg_360_830[idx + 1], l % 1),
//     lerp(cie1931_xyz_2deg_360_830[idx + 1], cie1931_xyz_2deg_360_830[idx + 2], l % 1),
//     lerp(cie1931_xyz_2deg_360_830[idx + 2], cie1931_xyz_2deg_360_830[idx + 3], l % 1)
//   ];
// };

// let integratedCIE = new Float32Array(3);
// for (let i = 0; i < cie1931_xyz_2deg_360_830.length; i += 4) {
//   let x = cie1931_xyz_2deg_360_830[i];
//   let y = cie1931_xyz_2deg_360_830[i + 1];
//   let z = cie1931_xyz_2deg_360_830[i + 2];
//   let rgb = xyz2rgb(x, y, z);

//   integratedCIE[0] += rgb[0] / 100;
//   integratedCIE[1] += rgb[1] / 100;
//   integratedCIE[2] += rgb[2] / 100;
// }
// uni.values.integratedCIE.set(integratedCIE);

// // compute reference white balance
// let refColor = new Float32Array(3);
// let refWeight = new Float32Array(3);
// const startL = 380;//uni.values.start_L[0];
// const endL = 780;//uni.values.end_L[0];
// const numSteps = 512;
// const refL = 550;//uni.values.ref_L[0];
// const step = (endL - startL) / numSteps;

// for (let l = startL; l <= endL; l += step) {
//   const xyz = sampleCMF(l);          // same CMF table as shader
//   const rgb = xyz2rgb(...xyz);
//   const scale = refL / l;

//   for (let c = 0; c < 3; c++) {
//     const clamped = Math.max(0, rgb[c]);
//     refColor[c]  += clamped * scale * scale * step;
//     refWeight[c] += clamped * step;
//   }
// }
// // refColor[c] / refWeight[c] is the average scale^2 bias per channel
// // whiteBalance corrects it back to (1,1,1)
// const whiteBalance = refWeight.map((w, i) => w / refColor[i]);
// console.log("White balance gains:", whiteBalance);
// uni.values.integratedCIE.set(whiteBalance)