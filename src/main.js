let adapter, device;
let gpuInfo = false;

async function loadImageBitmap(url) {
  const res = await fetch(url);
  const blob = await res.blob();
  return await createImageBitmap(blob, { colorSpaceConversion: 'none' });
}

async function main() {

  if (device) device.destroy();

  // let maxComputeInvocationsPerWorkgroup, maxBufferSize, f32filterable;

  // WebGPU Setup
  // if (!device) {
  adapter = await navigator.gpu?.requestAdapter();

  const maxComputeInvocationsPerWorkgroup = adapter.limits.maxComputeInvocationsPerWorkgroup;
  const maxBufferSize = adapter.limits.maxBufferSize;
  const f32filterable = adapter.features.has("float32-filterable");

  // compute workgroup size 16*8*8 | 32*8*4 | 64*4*4 = 1024 threads if maxComputeInvocationsPerWorkgroup >= 1024, otherwise 16*4*4 = 256 threads
  const largeWg = maxComputeInvocationsPerWorkgroup >= 1024;
  const [wg_x, wg_y, wg_z] = largeWg ? [16, 8, 8] : [16, 4, 4];

  if (!gpuInfo) {
    gui.addGroup("deviceInfo", "Device info", `
<pre><span ${!largeWg ? "class='warn'" : ""}>maxComputeInvocationsPerWorkgroup: ${maxComputeInvocationsPerWorkgroup}
workgroup: [${wg_x}, ${wg_y}, ${wg_z}]</span>
maxBufferSize: ${maxBufferSize}
f32filterable: ${f32filterable}
</pre>
    `);
    gpuInfo = true;
  }

  device = await adapter?.requestDevice({
    requiredFeatures: [
      ...(adapter.features.has("timestamp-query") ? ["timestamp-query"] : []),
      ...(f32filterable ? ["float32-filterable"] : []),
      "shader-f16",
    ],
    limits: {
      maxComputeInvocationsPerWorkgroup: 1024,
    }
  });

  // restart if device crashes
  device.lost.then((info) => {
    if (info.reason != "destroyed") {
      hardReset();
      console.warn("WebGPU device lost, reinitializing.");
    }
  });

  // }
  if (!device) {
    alert("Browser does not support WebGPU");
    document.body.textContent = "WebGPU is not supported in this browser.";
    return;
  }
  const context = canvas.getContext("webgpu");
  const swapChainFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device: device,
    format: swapChainFormat,
  });

  const newTexture = (name) => device.createTexture({
    size: imgSize,
    dimension: "2d",
    format: "rgba32float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
    label: `${name} texture`
  });

  storage.colorMatchingTex = device.createTexture({
    size: [cie1931_xyz_2deg_360_830.length / 4],
    dimension: "1d",
    format: "rgba32float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    label: "color matching texture"
  });
  device.queue.writeTexture(
    { texture: storage.colorMatchingTex },
    cie1931_xyz_2deg_360_830.buffer,
    {}, { width: cie1931_xyz_2deg_360_830.length / 4 }
  );

  const source = await loadImageBitmap("./img/Double Slit.png");
  storage.sourceTex = device.createTexture({
    label: "raw image",
    format: 'rgba8unorm',
    size: [source.width, source.height],
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  device.queue.copyExternalImageToTexture(
    { source: source },
    { texture: storage.sourceTex },
    { width: source.width, height: source.height },
  );

  storage.inputTex = newTexture("input");
  storage.rowFreqTex = newTexture("rowFreq");
  storage.colFreqTex = newTexture("colFreq");
  storage.finalTex = newTexture("final");
  const views = Object.fromEntries(
    Object.entries(storage).filter(([key, value]) => value instanceof GPUTexture).map(([key, texture]) => [key, texture.createView()])
  );

  const uniformBuffer = uni.createBuffer(device);

  const newComputePipeline = (shaderCode, name, entryPoint = "main") =>
    device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({
          code: shaderCode,
          label: `${name} compute module`
        }),
        constants: {},
        entryPoint: entryPoint
      },
      label: `${name} compute pipeline`
    });

  const preprocessPipeline = newComputePipeline(imgPreprocessShaderCode, "preprocess", "main");

  const preprocessBindGroup = device.createBindGroup({
    layout: preprocessPipeline.getBindGroupLayout(0),
    entries: [
      // { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: views.sourceTex },
      { binding: 2, resource: views.inputTex },
    ],
    label: "preprocess compute bind group"
  });

  const rowComputePipeline = newComputePipeline(computeShaderCode, "row FFT", "rowFFT_r4");

  const rowComputeBindGroup = device.createBindGroup({
    layout: rowComputePipeline.getBindGroupLayout(0),
    entries: [
      // { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: views.inputTex },
      { binding: 2, resource: views.rowFreqTex },
    ],
    label: "rowfft compute bind group"
  });

  const colComputePipeline = newComputePipeline(computeShaderCode, "column FFT", "colFFT_r4");

  const colComputeBindGroup = device.createBindGroup({
    layout: colComputePipeline.getBindGroupLayout(0),
    entries: [
      // { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: views.rowFreqTex },
      { binding: 2, resource: views.colFreqTex },
    ],
    label: "colfft compute bind group"
  });

  const filter = f32filterable ? "linear" : "nearest";
  const repeatSampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
    addressModeU: "repeat",
    addressModeV: "repeat",
  });

  const dispersionPipeline = newComputePipeline(dispersionShaderCode, "dispersion");

  const dispersionBindGroup = device.createBindGroup({
    layout: dispersionPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: views.colFreqTex },
      { binding: 2, resource: views.finalTex },
      { binding: 3, resource: repeatSampler },
      { binding: 4, resource: views.colorMatchingTex },
    ],
    label: "dispersion compute bind group"
  });


  const renderModule = device.createShaderModule({
    code: renderShaderCode,
    label: "render module"
  });

  const renderPipeline = device.createRenderPipeline({
    label: '3d volume rendering pipeline',
    layout: 'auto',
    vertex: { module: renderModule },
    fragment: {
      module: renderModule,
      targets: [{ format: swapChainFormat }],
    }
  });

  const renderBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: views.finalTex },
      // { binding: 1, resource: views.inputTex },
      // { binding: 1, resource: views.rowFreqTex },
      // { binding: 1, resource: views.sourceTex },
      { binding: 2, resource: repeatSampler },
    ],
  });

  const renderPassDescriptor = {
    label: 'render pass',
    colorAttachments: [
      {
        clearValue: [0, 0, 0, 1],
        loadOp: 'clear',
        storeOp: 'store',
      },
    ]
  };
  const filterStrength = 50;

  const computeTimingHelper = new TimingHelper(device);
  const renderTimingHelper = new TimingHelper(device);

  const encoder = device.createCommandEncoder();
  const preprocessPass = encoder.beginComputePass();
  preprocessPass.setPipeline(preprocessPipeline);
  preprocessPass.setBindGroup(0, preprocessBindGroup);
  preprocessPass.dispatchWorkgroups(Math.ceil(imgSize[0] / 16), Math.ceil(imgSize[1] / 16));
  preprocessPass.end();
  device.queue.submit([encoder.finish()]);


  function render() {
    const startTime = performance.now();
    deltaTime += Math.min(startTime - lastFrameTime - deltaTime, 1e4) / filterStrength;
    const speedMultiplier = Math.min(deltaTime, 50);
    fps += (1e3 / deltaTime - fps) / filterStrength;
    lastFrameTime = startTime;

    const canvasTexture = context.getCurrentTexture();
    renderPassDescriptor.colorAttachments[0].view = canvasTexture.createView();

    uni.update(device.queue);

    const encoder = device.createCommandEncoder();

    const computePass = computeTimingHelper.beginComputePass(encoder);
    // for (let i = 0; i < 50; i++) {
      computePass.setPipeline(rowComputePipeline);
      computePass.setBindGroup(0, rowComputeBindGroup);
      computePass.dispatchWorkgroups(1, 1024);

      computePass.setPipeline(colComputePipeline);
      computePass.setBindGroup(0, colComputeBindGroup);
      computePass.dispatchWorkgroups(1024, 1);
    // }
    computePass.setPipeline(dispersionPipeline);
    computePass.setBindGroup(0, dispersionBindGroup);
    computePass.dispatchWorkgroups(Math.ceil(imgSize[0] / 16), Math.ceil(imgSize[1] / 16));
    computePass.end();

    const renderPass = renderTimingHelper.beginRenderPass(encoder, renderPassDescriptor);
    renderPass.setPipeline(renderPipeline);
    renderPass.setBindGroup(0, renderBindGroup);
    renderPass.draw(3);
    renderPass.end();

    device.queue.submit([encoder.finish()]);
    computeTimingHelper.getResult().then(gpuTime => computeTime += (gpuTime / 1e6 - computeTime) / filterStrength);
    renderTimingHelper.getResult().then(gpuTime => renderTime += (gpuTime / 1e6 - renderTime) / filterStrength);

    jsTime += (performance.now() - startTime - jsTime) / filterStrength;

    rafId = requestAnimationFrame(render);
  }

  perfIntId = setInterval(() => {
    gui.io.fps(fps.toFixed(1));
    gui.io.frameTime(deltaTime.toFixed(2));
    gui.io.jsTime(jsTime.toFixed(2));
    gui.io.computeTime((computeTime).toFixed(2));
    gui.io.renderTime(renderTime.toFixed(2));
  }, 100);
  rafId = requestAnimationFrame(render);
}

uni.values.start_L.set([380]);
uni.values.end_L.set([780]);
uni.values.steps.set([512]);
uni.values.ref_L.set([550]);
uni.values.dispMult.set([1]);

main();