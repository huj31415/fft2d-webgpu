let gpuInfo = false;


async function main() {

  if (device) device.destroy();

  // let maxComputeInvocationsPerWorkgroup, maxBufferSize, f32filterable;

  // WebGPU Setup
  // if (!device) {
  adapter = await navigator.gpu?.requestAdapter();

  const maxComputeInvocationsPerWorkgroup = adapter.limits.maxComputeInvocationsPerWorkgroup;
  const maxComputeWorkgroupSizeX = adapter.limits.maxComputeWorkgroupSizeX;
  const maxComputeWorkgroupStorageSize = adapter.limits.maxComputeWorkgroupStorageSize;
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
    requiredLimits: {
      maxComputeInvocationsPerWorkgroup: maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: maxComputeWorkgroupSizeX,
      maxComputeWorkgroupStorageSize: maxComputeWorkgroupStorageSize,
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
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
    label: `${name} texture`
  });

  // const colorMatchingData = cie1964_xyz_10deg_360_830;
  const colorMatchingData = cie1931_xyz_2deg_360_830;

  storage.colorMatchingTex = device.createTexture({
    size: [colorMatchingData.length / 4],
    dimension: "1d",
    format: "rgba32float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    label: "color matching texture"
  });
  device.queue.writeTexture(
    { texture: storage.colorMatchingTex },
    colorMatchingData.buffer,
    {}, { width: colorMatchingData.length / 4 }
  );

  storage.whitePointTex = device.createTexture({
    size: [cieD65_360_830.length],
    dimension: "1d",
    format: "r32float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    label: "white point texture"
  });
  device.queue.writeTexture(
    { texture: storage.whitePointTex },
    cieD65_360_830.buffer,
    {}, { width: cieD65_360_830.length }
  );

  const source = await loadImageBitmap("./img/DoubleSlit.png");
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
  storage.freqBuffer = device.createBuffer({
    label: "frequency buffer",
    size: imgSize[0] * imgSize[1] * 4 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  storage.normalizationBuffer = device.createBuffer({
    label: "normalization buffer",
    size: 4 * 34, // 1 u32 for each rgb channel
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  const uniformBuffer = uni.createBuffer(device);

  const newComputePipeline = (shaderCode, name, entryPoint = "main", consts = {}) =>
    device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({
          code: shaderCode,
          label: `${name} compute module`
        }),
        constants: consts,
        entryPoint: entryPoint
      },
      label: `${name} compute pipeline`
    });

  const clampSampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
    addressModeU: "clamp-to-edge",
    addressModeV: "clamp-to-edge",
  });
  const preprocessPipeline = newComputePipeline(preprocessShaderCode, "preprocess", "main");

  const preprocessBindGroup = device.createBindGroup({
    layout: preprocessPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: views.sourceTex },
      { binding: 2, resource: views.inputTex },
      { binding: 3, resource: clampSampler },
    ],
    label: "preprocess compute bind group"
  });

  const rowComputePipeline = newComputePipeline(fftShaderCode, "row FFT", "rowFFT_r4");
  const rowComputeBindGroup = device.createBindGroup({
    layout: rowComputePipeline.getBindGroupLayout(0),
    entries: [
      // { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: views.inputTex },
      { binding: 2, resource: views.rowFreqTex },
    ],
    label: "rowfft compute bind group"
  });

  const colComputePipeline = newComputePipeline(fftShaderCode, "column FFT", "colFFT_r4");
  const colComputeBindGroup = device.createBindGroup({
    layout: colComputePipeline.getBindGroupLayout(0),
    entries: [
      // { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: views.rowFreqTex },
      { binding: 2, resource: views.colFreqTex },
    ],
    label: "colfft compute bind group"
  });

  const CTrowComputePipeline = newComputePipeline(CTfftShaderCode, "row FFT", "rowFFT_r4", { INV: 1 });
  const CTrowComputeBindGroup = device.createBindGroup({
    layout: CTrowComputePipeline.getBindGroupLayout(0),
    entries: [
      // { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: views.finalTex },
      { binding: 2, resource: views.rowFreqTex },
    ],
    label: "cooley-tukey rowfft compute bind group"
  });

  const CTcolComputePipeline = newComputePipeline(CTfftShaderCode, "column FFT", "rowFFT_r4", { INV: 1 });
  const CTcolComputeBindGroup = device.createBindGroup({
    layout: CTcolComputePipeline.getBindGroupLayout(0),
    entries: [
      // { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: views.rowFreqTex },
      { binding: 2, resource: views.finalTex },
    ],
    label: "cooley-tukey colfft compute bind group"
  });

  const invCTrowComputePipeline = newComputePipeline(CTfftShaderCode, "row FFT", "rowFFT_r4", { INV: -1 });
  const invCTrowComputeBindGroup = device.createBindGroup({
    layout: invCTrowComputePipeline.getBindGroupLayout(0),
    entries: [
      // { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: views.finalTex },
      { binding: 2, resource: views.rowFreqTex },
    ],
    label: "cooley-tukey rowfft compute bind group"
  });

  const invCTcolComputePipeline = newComputePipeline(CTfftShaderCode, "column FFT", "rowFFT_r4", { INV: -1 });
  const invCTcolComputeBindGroup = device.createBindGroup({
    layout: invCTcolComputePipeline.getBindGroupLayout(0),
    entries: [
      // { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: views.rowFreqTex },
      { binding: 2, resource: views.finalTex },
    ],
    label: "cooley-tukey colfft compute bind group"
  });

  const normalizeComputePipeline = newComputePipeline(normalizationShaderCode, "normalization");
  const normalizeBindGroup = device.createBindGroup({
    layout: normalizeComputePipeline.getBindGroupLayout(0),
    entries: [
      // { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 0, resource: { buffer: storage.freqBuffer } },
      { binding: 1, resource: { buffer: storage.normalizationBuffer } },
    ],
    label: "normalization compute bind group"
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
      { binding: 5, resource: views.whitePointTex },
      { binding: 6, resource: { buffer: storage.normalizationBuffer } },
    ],
    label: "dispersion compute bind group"
  });


  const renderModule = device.createShaderModule({
    code: AgXRenderShaderCode,
    label: "render module"
  });

  const renderPipeline = device.createRenderPipeline({
    label: 'main rendering pipeline',
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

  const copyRenderPipeline = device.createRenderPipeline({
    label: 'copy rendering pipeline',
    layout: 'auto',
    vertex: { module: renderModule },
    fragment: {
      module: renderModule,
      targets: [{ format: "rgba32float" }],
    }
  });

  const copyRenderBindGroup = device.createBindGroup({
    layout: copyRenderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: views.finalTex },
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
  
  const apertureRenderModule = device.createShaderModule({
    code: directRenderShaderCode,
    label: "aperture render module"
  });
  const apertureRenderPipeline = device.createRenderPipeline({
    label: 'aperture render pipeline',
    layout: 'auto',
    vertex: { module: apertureRenderModule },
    fragment: {
      module: apertureRenderModule,
      targets: [{ format: swapChainFormat }],
    }
  });
  const apertureRenderBindGroup = device.createBindGroup({
    layout: apertureRenderPipeline.getBindGroupLayout(0),
    entries: [
      // { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: views.inputTex },
      { binding: 2, resource: repeatSampler },
    ],
  });
  const apertureRenderPassDescriptor = {
    label: 'aperture render pass',
    colorAttachments: [
      {
        clearValue: [0, 0, 0, 1],
        loadOp: 'clear',
        storeOp: 'store',
      },
    ]
  };
  apertureCtx.configure({
    device: device,
    format: swapChainFormat,
  });

  const filterStrength = 50;

  const fftComputeTimingHelper = new TimingHelper(device);
  const dispersionComputeTimingHelper = new TimingHelper(device);
  const renderTimingHelper = new TimingHelper(device);

  downloadImg = () => {
    const imgDim = gui.io.canvasResolution.value;
    const texture = device.createTexture({
      label: "output texture",
      format: "rgba32float",
      size: [imgDim, imgDim],
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const copyBuffer = device.createBuffer({
      label: "copy buffer",
      size: imgDim * imgDim * 4 * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    renderPassDescriptor.colorAttachments[0].view = texture.createView();
    const encoder = device.createCommandEncoder();
    const renderPass = encoder.beginRenderPass(renderPassDescriptor);
    renderPass.setPipeline(copyRenderPipeline);
    renderPass.setBindGroup(0, copyRenderBindGroup);
    renderPass.draw(3);
    renderPass.end();
    encoder.copyTextureToBuffer(
      { texture: texture },
      { buffer: copyBuffer, bytesPerRow: imgDim * 4 * 4 },
      [imgDim, imgDim]
    );
    device.queue.submit([encoder.finish()]);
    
    let arrayBuffer;
    copyBuffer.mapAsync(GPUMapMode.READ).then(() => {
      arrayBuffer = copyBuffer.getMappedRange();
      const data = new Float32Array(arrayBuffer);

      // Create a blob representing raw binary data
      const blob = new Blob([arrayBuffer], { type: "application/octet-stream" });
      
      // Generate a secure DOM URL referencing our memory block
      const url = URL.createObjectURL(blob);
      
      // Construct a temporary link element off-screen
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = "kernelTex.bin";
      
      // Programmatically trigger the native browser download dialog
      anchor.click();
      
      // Clean up memory to avoid leaks
      URL.revokeObjectURL(url);
      copyBuffer.unmap();
      texture.destroy();
      copyBuffer.destroy();
    });
  }

  function render() {
    const startTime = performance.now();
    deltaTime += Math.min(startTime - lastFrameTime - deltaTime, 1e4) / filterStrength;
    const speedMultiplier = Math.min(deltaTime, 50);
    fps += (1e3 / deltaTime - fps) / filterStrength;
    lastFrameTime = startTime;

    if (uni.update(device.queue) || runEveryFrame) {
      const encoder = device.createCommandEncoder();
      if (updateAperture) {
        const apertureTexture = apertureCtx.getCurrentTexture();
        apertureRenderPassDescriptor.colorAttachments[0].view = apertureTexture.createView();

        const preprocessPass = encoder.beginComputePass();
        preprocessPass.setPipeline(preprocessPipeline);
        preprocessPass.setBindGroup(0, preprocessBindGroup);
        preprocessPass.dispatchWorkgroups(Math.ceil(imgSize[0] / 16), Math.ceil(imgSize[1] / 16));
        preprocessPass.end();
        updateAperture = false;

        const apertureRenderPass = encoder.beginRenderPass(apertureRenderPassDescriptor);
        apertureRenderPass.setPipeline(apertureRenderPipeline);
        apertureRenderPass.setBindGroup(0, apertureRenderBindGroup);
        apertureRenderPass.draw(3);
        apertureRenderPass.end();
      }
      const canvasTexture = context.getCurrentTexture();
      renderPassDescriptor.colorAttachments[0].view = canvasTexture.createView();


      const fftComputePass = runIFFT ? encoder.beginComputePass() : fftComputeTimingHelper.beginComputePass(encoder);
      
      fftComputePass.setPipeline(rowComputePipeline);
      fftComputePass.setBindGroup(0, rowComputeBindGroup);
      fftComputePass.dispatchWorkgroups(1, 1024, 1);
      
      fftComputePass.setPipeline(colComputePipeline);
      fftComputePass.setBindGroup(0, colComputeBindGroup);
      fftComputePass.dispatchWorkgroups(1024, 1, 1);
      // fftComputePass.dispatchWorkgroups(1, 1024, 1);
      
      fftComputePass.end();
      
      if (normalize) {
        encoder.copyTextureToBuffer(
          { texture: storage.colFreqTex },
          { buffer: storage.freqBuffer, bytesPerRow: imgSize[0] * 4 * 4 },
          [imgSize[0], imgSize[1]]
        );
        const normalizePass = encoder.beginComputePass();
        normalizePass.setPipeline(normalizeComputePipeline);
        normalizePass.setBindGroup(0, normalizeBindGroup);
        normalizePass.dispatchWorkgroups(1024, 1, 1);
        normalizePass.end();
      }
      const dispersionComputePass = dispersionComputeTimingHelper.beginComputePass(encoder);
      dispersionComputePass.setPipeline(dispersionPipeline);
      dispersionComputePass.setBindGroup(0, dispersionBindGroup);
      dispersionComputePass.dispatchWorkgroups(Math.ceil(imgSize[0] / 16), Math.ceil(imgSize[1] / 16), 1);
      dispersionComputePass.end();
      
      if (runIFFT) {
        const fftComputePass2 = fftComputeTimingHelper.beginComputePass(encoder);
        
        fftComputePass2.setPipeline(CTrowComputePipeline);
        fftComputePass2.setBindGroup(0, CTrowComputeBindGroup);
        fftComputePass2.dispatchWorkgroups(1, 1024, 1);
        
        fftComputePass2.setPipeline(CTcolComputePipeline);
        fftComputePass2.setBindGroup(0, CTcolComputeBindGroup);
        fftComputePass2.dispatchWorkgroups(1, 1024, 1);

        fftComputePass2.setPipeline(invCTrowComputePipeline);
        fftComputePass2.setBindGroup(0, invCTrowComputeBindGroup);
        fftComputePass2.dispatchWorkgroups(1, 1024, 1);
        
        fftComputePass2.setPipeline(invCTcolComputePipeline);
        fftComputePass2.setBindGroup(0, invCTcolComputeBindGroup);
        fftComputePass2.dispatchWorkgroups(1, 1024, 1);
        
        fftComputePass2.end();
      }

      const renderPass = renderTimingHelper.beginRenderPass(encoder, renderPassDescriptor);
      renderPass.setPipeline(renderPipeline);
      renderPass.setBindGroup(0, renderBindGroup);
      renderPass.draw(3);
      renderPass.end();

      device.queue.submit([encoder.finish()]);
      fftComputeTimingHelper.getResult().then(gpuTime => computeTime += (gpuTime / 1e6 - computeTime) / filterStrength);
      dispersionComputeTimingHelper.getResult().then(gpuTime => dispersionTime += (gpuTime / 1e6 - dispersionTime) / filterStrength);
      renderTimingHelper.getResult().then(gpuTime => renderTime += (gpuTime / 1e6 - renderTime) / filterStrength);
    }

    jsTime += (performance.now() - startTime - jsTime) / filterStrength;

    rafId = requestAnimationFrame(render);
  }

  perfIntId = setInterval(() => {
    gui.io.fps(fps.toFixed(1));
    gui.io.frameTime(deltaTime.toFixed(2));
    gui.io.jsTime(jsTime.toFixed(2));
    gui.io.computeTime(computeTime.toFixed(2));
    gui.io.dispersionTime(dispersionTime.toFixed(2));
    gui.io.renderTime(renderTime.toFixed(2));
  }, 100);
  rafId = requestAnimationFrame(render);
}

gui.updateAllVisibility();

uni.set("start_L", [380]);
uni.set("end_L", [780]);
uni.set("steps", [512]);
uni.set("ref_L", [550]);
uni.set("dispMult", [1]);
uni.set("gain", [1]);
uni.set("scale", [1]);
uni.set("srcScale", [1]);
uni.set("noiseSize", [64]);
uni.set("noiseOctaves", [5]);
uni.set("noiseAmp", [1 / 5]);
uni.set("widthFactor", [1]);
uni.set("FFTshift", [1]);

main();