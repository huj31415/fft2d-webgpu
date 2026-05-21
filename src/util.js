

const uni = new Uniforms();
uni.addUniform("resolution", "f32");
uni.addUniform("start_L", "f32");
uni.addUniform("end_L", "f32");
uni.addUniform("steps", "f32");
uni.addUniform("ref_L", "f32");
uni.addUniform("dispMult", "f32");
uni.addUniform("gain", "f32");
uni.addUniform("scale", "f32");
uni.addUniform("srcScale", "f32");
uni.addUniform("polygonSides", "f32");
uni.addUniform("rot", "f32");
uni.addUniform("noiseSize", "f32");
uni.addUniform("noiseOctaves", "f32");
uni.addUniform("noiseAmp", "f32");
uni.addUniform("widthFactor", "f32");
uni.addUniform("FFTshift", "f32");

uni.finalize();

const storage = {
  sourceTex: null,
  inputTex: null,
  rowFreqTex: null,
  colFreqTex: null,
  finalTex: null,
};

const canvas = document.getElementById("_canvas");
canvas.style.imageRendering = "pixelated";

const resizeCanvas = (value = window.devicePixelRatio) => {
  // pixelRatio = value / window.innerHeight || 1;
  const minDim = Math.min(window.innerWidth, window.innerHeight);
  canvas.style.width = `${minDim}px`;
  canvas.style.height = `${minDim}px`;
  canvas.width = canvas.height = value;
  uni.set("resolution", [value]);
  gui.io.res([value, value]);
}


async function loadImageBitmap(url) {
  const res = await fetch(url);
  const blob = await res.blob();
  return await createImageBitmap(blob, { colorSpaceConversion: 'none' });
}


const gui = new GUI("2D FFT Diffraction", canvas);
const imgSize = [1024, 1024];

let adapter, device;
let runEveryFrame = false;
let updateAperture = true;
let runIFFT = false;
let normalize = false;

let downloadImg = () => {};

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
gui.addCheckbox("runEveryFrame", "Run every frame", false, "perf", (value) => runEveryFrame = value);
gui.addCheckbox("runFFTIFFT", "Run RGBA FFT/IFFT", false, "perf", (value) => runIFFT = value);
gui.addCheckbox("fftshift", "FFTshift", true, "perf", (value) => uni.set("FFTshift", [value ? 1 : 0]));
gui.addCheckbox("normalize", "Normalize output", false, "perf", (value) => {
  if (!device) return;
  normalize = value;
  if (!value) {
    const zeroData = new Uint8Array([0, 0, 0, 0]);
    device.queue.writeBuffer(storage.normalizationBuffer, 0, zeroData.buffer, 0, zeroData.byteLength);
  }
  uni.valuesChanged = true;
});
gui.addDropdown("canvasResolution", "Canvas resolution", [
  "1024",
  "512",
  "256",
  "128",
], "perf", null, (value) => resizeCanvas(parseInt(value)));
gui.addButton("download", "Download image", true, "perf", () => downloadImg());

gui.addGroup("dispersion", "Dispersion settings");
gui.addNumericInput("start_L", true, "Start wavelength", { min: 360, max: 820, step: 1, val: 380, float: 0 }, "dispersion", (value) => uni.set("start_L", [value]));
gui.addNumericInput("end_L", true, "End wavelength", { min: 370, max: 830, step: 1, val: 780, float: 0 }, "dispersion", (value) => uni.set("end_L", [value]));
gui.addNumericInput("steps", true, "Steps", { min: 8, max: 1024, step: 8, val: 512, float: 0 }, "dispersion", (value) => uni.set("steps", [value]));
gui.addNumericInput("ref_L", true, "Ref. wavelength", { min: 360, max: 830, step: 5, val: 550, float: 0 }, "dispersion", (value) => uni.set("ref_L", [value]));
gui.addNumericInput("dispMult", true, "Strength", { min: 0, max: 2, step: 0.05, val: 1, float: 2 }, "dispersion", (value) => uni.set("dispMult", [value]));
gui.addNumericInput("gain", true, "Gain", { min: -5, max: 5, step: 0.1, val: 0, float: 1 }, "dispersion", (value) => uni.set("gain", [10**value]));
gui.addNumericInput("scale", true, "Scale", { min: 0, max: 3, step: 0.01, val: 0, float: 2 }, "dispersion", (value) => uni.set("scale", [10**value]));

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

gui.addGroup("source", "Source visualization");
gui.addDropdown("aperture", "Aperture shape", [
  "DoubleSlit",
  "Polygon",
  "Square",
  "RoundSquare",
  "Rhombus",
  "Pentagon",
  "HexBand",
  "RectangleV",
  "RoundOctagon",
  "Window",
  "RoundWindow",
  "Bahtinov",
  "Bahtinov2",
  "JWST",
  "Oval"
], "source", { "Polygon": ["polygonSides", "rotation"] }, async (value) => {
  if (value != "Polygon" && device) {
    const source = await loadImageBitmap(`./img/${value}.png`);
    device.queue.copyExternalImageToTexture(
      { source },
      { texture: storage.sourceTex },
      { width: source.width, height: source.height },
    );
    uni.set("polygonSides", [0]);
  } else if (value === "Polygon") {
    uni.set("polygonSides", [parseInt(gui.io.polygonSides.value)]);
  }
  updateAperture = true;
});
gui.addNumericInput("polygonSides", true, "Polygon sides", { min: 3, max: 24, step: 1, val: 6, float: 0 }, "source", (value) => {
  uni.set("polygonSides", [value]);
  updateAperture = true;
});
gui.addNumericInput("rotation", true, "Polygon rotation", { min: 0, max: 360, step: 1, val: 0, float: 0 }, "source", (value) => {
  uni.set("rot", [value * Math.PI / 180]);
  updateAperture = true;
});
gui.addCheckbox("addNoise", "Add noise", true, "source", (value) => {
  uni.set("noiseSize", [value ? 2**gui.io.noiseSize.value : 0]);
  updateAperture = true;
});
gui.addNumericInput("noiseSize", true, "Noise size", { min: 0, max: 10, step: 1, val: 5, float: 0 }, "source", (value) => {
  uni.set("noiseSize", [2**value]);
  updateAperture = true;
});
gui.addNumericInput("noiseOctaves", true, "Noise octaves", { min: 1, max: 10, step: 1, val: 5, float: 0 }, "source", (value) => {
  uni.set("noiseOctaves", [value]);
  updateAperture = true;
});
gui.addNumericInput("noiseAmp", true, "Noise amplitude", { min: 1, max: 10, step: 1, val: 5, float: 0 }, "source", (value) => {
  uni.set("noiseAmp", [1 / (11 - value)]);
  updateAperture = true;
});
gui.addNumericInput("widthFactor", true, "Width factor", { min: 0.1, max: 10, step: 0.1, val: 1, float: 1 }, "source", (value) => {
  uni.set("widthFactor", [value]);
  updateAperture = true;
});
gui.addNumericInput("srcScale", true, "Source scale", { min: -2, max: 2, step: 0.01, val: 0, float: 2 }, "source", (value) => {
  uni.set("srcScale", [10**value]);
  updateAperture = true;
});


const apertureCtx = gui.addCanvas("sourceCanvas", "Aperture", {}, 1, 1, "webgpu", "source");


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
window.onresize = window.onload = () => resizeCanvas(1024);


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
// uni.set("integratedCIE", integratedCIE);

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
// uni.set("integratedCIE", whiteBalance)