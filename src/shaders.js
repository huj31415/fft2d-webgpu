const preprocessShaderCode = /* wgsl */`
${uni.uniformStruct}

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var input: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var texSampler: sampler;

const PI = 3.14159265359;
const TAU = 6.28318530718;
const N = 1024u;

fn pcgHash(input: u32) -> f32 {
  let state = input * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return f32((word >> 22) ^ word) / 4294967295.0;
}

fn smoothNoise2D(uv: vec2f) -> f32 {
  let i = vec2u(floor(uv));
  let f = fract(uv);

  let u = f * f * (3.0 - 2.0 * f); // Equivalent to smoothstep

  let val00 = pcgHash((i.x + 0u) * N + (i.y + 0u));
  let val10 = pcgHash((i.x + 1u) * N + (i.y + 0u));
  let val01 = pcgHash((i.x + 0u) * N + (i.y + 1u));
  let val11 = pcgHash((i.x + 1u) * N + (i.y + 1u));

  let mix_bottom = mix(val00, val10, u.x);
  let mix_top    = mix(val01, val11, u.x);
  
  return mix(mix_bottom, mix_top, u.y);
}

fn getGradient(p: vec2u) -> vec2f {
  let h = pcgHash(dot(p, vec2u(127, 311)));
  let angle = h * TAU; // Map to 0 to 2*PI
  return vec2f(cos(angle), sin(angle));
}

// 2D Perlin Noise function (outputs a range of roughly [-0.7, 0.7])
fn perlinNoise2D(uv: vec2f) -> f32 {
  let i = vec2u(floor(uv));
  let f = fract(uv);

  // Quintic interpolation curve: 6t^5 - 15t^4 + 10t^3
  // This gives smoother second derivatives than standard smoothstep (3t^2 - 2t^3)
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

  // 1. Get gradients for all 4 corners of the cell
  let grad00 = getGradient(i + vec2u(0, 0));
  let grad10 = getGradient(i + vec2u(1, 0));
  let grad01 = getGradient(i + vec2u(0, 1));
  let grad11 = getGradient(i + vec2u(1, 1));

  // 2. Calculate displacement vectors from corners to current point
  let dist00 = f - vec2f(0.0, 0.0);
  let dist10 = f - vec2f(1.0, 0.0);
  let dist01 = f - vec2f(0.0, 1.0);
  let dist11 = f - vec2f(1.0, 1.0);

  // 3. Dot product between gradients and displacements
  let dot00 = dot(grad00, dist00);
  let dot10 = dot(grad10, dist10);
  let dot01 = dot(grad01, dist01);
  let dot11 = dot(grad11, dist11);

  // 4. Bilinear interpolation using the quintic blend weights
  let mix_bottom = mix(dot00, dot10, u.x);
  let mix_top    = mix(dot01, dot11, u.x);
  let noise      = mix(mix_bottom, mix_top, u.y);

  return noise;
}

fn modulo(a: f32, b: f32) -> f32 {
  return a - b * floor(a / b);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let gid_f = vec2f(gid.xy);
  let uv = gid_f / vec2f(1024.0);

  // let dist = (distance(vec2f(pos), vec2f(vec2u(r))) / f32(r));
  // let window = (0.4243801 - 0.4973406 * cos(PI * (dist - 1.0)) + 0.0782793 * cos(TAU * (dist - 1.0)));
  // let window = 0.5 - 0.5 * cos(PI * (dist - 1.0));
  // let window = smoothstep(1, 0.8, dist);

  let vecToCenter = (uv - 0.5) * vec2f(uni.widthFactor, 1);
  var sample = vec4f(0);
  if (uni.polygonSides > 2) {
    let sectorAngle = PI / uni.polygonSides;
    let num = cos(sectorAngle) * 0.4 * uni.srcScale;
    sample = vec4f(f32(length(vecToCenter) < num / cos(modulo(atan2(vecToCenter.y, -vecToCenter.x) - uni.rot, 2 * sectorAngle) - sectorAngle)));
  } else {
    let sample_uv = vecToCenter / uni.srcScale + 0.5;
    sample = textureSampleLevel(input, texSampler, sample_uv + 0.5 * (1 / 1024.0), 0.0);
  }
  if (uni.noiseSize > 0) {
    var noise = 0.0;
    var scale = 0.0;
    let minNoiseSize = u32(uni.noiseSize) >> (u32(uni.noiseOctaves) - 1u);
    for (var size = u32(uni.noiseSize); size >= max(minNoiseSize, 1u); size >>= 1u) {
      noise += (perlinNoise2D(vec2f(gid.xy) / f32(size)) + 1.0) * 0.5 * f32(size);
      scale += f32(size);
    }
    sample *= pow(noise / scale, uni.noiseAmp);
  }
  textureStore(output, gid.xy, sample);
}
`;

const fftShaderCode = /* wgsl */`
// enable f16;
// enable subgroups;
// can run first log_SIZE(subgroup_size) passes with subgroups

${uni.uniformStruct}

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var input: texture_2d<f32>;
@group(0) @binding(2) var freqTex: texture_storage_2d<rgba32float, write>;

const PI = 3.14159265359;
const TAU = 6.28318530718;
const N = 1024u;    // (image + kernel - 1), power of 4
const N4_1 = N >> 2u; // N / 4
const N4_2 = N4_1 * 2u;
const N4_3 = N4_1 * 3u;

var<workgroup> row: array<array<vec2f, 1024>, 2>;

fn cmul(a: vec2f, b: vec2f) -> vec2f {
  return vec2f(a.x * b.x - a.y * b.y, dot(a, b.yx));
}

fn swizzle(linear_index: u32) -> u32 {
  // Take the block identifier (bits above the bank size) 
  // and XOR it into the lower bank bits
  // Reconstruct the index with the scrambled bank bits
  return (linear_index & ~31u) | ((linear_index & 31u) ^ ((linear_index >> 5u) & 31u));
}

fn fftR2(x: u32, Ns: u32, i: u32) -> vec2f {
  let halfNs = Ns / 2;
  let base = x / Ns * halfNs;
  let offset = x % halfNs;
  let x0 = base + offset;
  let x1 = x0 + (N >> 1u);
  let c0 = row[i][x0];
  let c1 = row[1 - i][x1];
  let angle = -TAU * (f32(x) / f32(Ns));
  let cT = vec2f(cos(angle), sin(angle));
  return c0 + cmul(cT, c1);
}

fn fftR2_prepass(x: u32, i: u32) {
  let t0 = row[i][x];
  let t1 = row[i][x + (N >> 1u)];
  row[1 - i][x * 2u]      = t0 + t1;
  row[1 - i][x * 2u + 1u] = t0 - t1;
}

fn fftR4_stockham(x: u32, Ns: u32, i: u32) {
  let Ns4 = Ns >> 2u; // Ns / 4
  let offset = x & (Ns4 - 1); // x % Ns4;

  // replace with shared memory LUT
  let angle = -TAU * (f32(offset) / f32(Ns));
  let w1 = vec2f(cos(angle), sin(angle));
  let w2 = cmul(w1, w1);
  let w3 = cmul(w2, w1);

  let t0 = row[i][swizzle(x) & ((1u << 10u) - 1u)]; // causes DXC crash without the binary mask
  let t1 = cmul(w1, row[i][swizzle(x + N4_1)]);
  let t2 = cmul(w2, row[i][swizzle(x + N4_2)]);
  let t3 = cmul(w3, row[i][swizzle(x + N4_3)]);

  let base = x / Ns4 * Ns + offset; // floor(x / Ns4) * Ns + offset;
  let dst = 1 - i;
  row[dst][swizzle(base)]            = t0 + t1 + t2 + t3;
  row[dst][swizzle(base + Ns4)]      = t0 + vec2f(t1.y, -t1.x) - t2 + vec2f(-t3.y, t3.x);
  row[dst][swizzle(base + Ns4 * 2u)] = t0 - t1 + t2 - t3;
  row[dst][swizzle(base + Ns4 * 3u)] = t0 + vec2f(-t1.y, t1.x) - t2 + vec2f(t3.y, -t3.x);
}

@compute @workgroup_size(N4_1)
fn rowFFT_r4(@builtin(global_invocation_id) gid: vec3u) {
  let y = gid.y;  // 0 to N
  let x0 = gid.x; // 0 to N/4
  let x1 = x0 + N4_1;
  let x2 = x1 + N4_1;
  let x3 = x2 + N4_1;

  // load rg as real and imag
  row[0][swizzle(x0)] = vec2f(textureLoad(input, (vec2u(x0, y)), 0).rg);
  row[0][swizzle(x1)] = vec2f(textureLoad(input, (vec2u(x1, y)), 0).rg);
  row[0][swizzle(x2)] = vec2f(textureLoad(input, (vec2u(x2, y)), 0).rg);
  row[0][swizzle(x3)] = vec2f(textureLoad(input, (vec2u(x3, y)), 0).rg);
  workgroupBarrier();

  // first src is idx 0
  var src = 0u;
  for (var Ns = 4u; Ns <= N; Ns <<= 2u) {
    fftR4_stockham(x0, Ns, src);
    src = 1u - src;
    workgroupBarrier();
  }

  // store combination of r and g FFT output
  let normalizer = 1.0 / f32(N);
  textureStore(freqTex, gid.xy,       vec4f(row[src][swizzle(x0)] * normalizer, 0, 1));
  textureStore(freqTex, vec2u(x1, y), vec4f(row[src][swizzle(x1)] * normalizer, 0, 1));
  textureStore(freqTex, vec2u(x2, y), vec4f(row[src][swizzle(x2)] * normalizer, 0, 1));
  textureStore(freqTex, vec2u(x3, y), vec4f(row[src][swizzle(x3)] * normalizer, 0, 1));
}

@compute @workgroup_size(1, N4_1)
fn colFFT_r4(@builtin(global_invocation_id) gid: vec3u) {
  let x = gid.x;  // 0 to N
  let y0 = gid.y; // 0 to N/4
  let y1 = y0 + N4_1;
  let y2 = y1 + N4_1;
  let y3 = y2 + N4_1;

  // load rg as real and imag
  row[0][swizzle(y0)] = textureLoad(input, (vec2u(x, y0)), 0).rg;
  row[0][swizzle(y1)] = textureLoad(input, (vec2u(x, y1)), 0).rg;
  row[0][swizzle(y2)] = textureLoad(input, (vec2u(x, y2)), 0).rg;
  row[0][swizzle(y3)] = textureLoad(input, (vec2u(x, y3)), 0).rg;
  workgroupBarrier();

  // first src is idx 0
  var src = 0u;
  for (var Ns = 4u; Ns <= N; Ns <<= 2u) {
    fftR4_stockham(y0, Ns, src);
    src = 1u - src;
    workgroupBarrier();
  }
  
  // store combination of r and g FFT output
  let normalizer = 1.0 / f32(N);
  textureStore(freqTex, gid.xy,       vec4f(dot(row[src][swizzle(y0)], row[src][swizzle(y0)]) * normalizer));
  textureStore(freqTex, vec2u(x, y1), vec4f(dot(row[src][swizzle(y1)], row[src][swizzle(y1)]) * normalizer));
  textureStore(freqTex, vec2u(x, y2), vec4f(dot(row[src][swizzle(y2)], row[src][swizzle(y2)]) * normalizer));
  textureStore(freqTex, vec2u(x, y3), vec4f(dot(row[src][swizzle(y3)], row[src][swizzle(y3)]) * normalizer));
}
`;

const dispersionShaderCode = /* wgsl */`
${uni.uniformStruct}

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var freqTex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var texSampler: sampler;
@group(0) @binding(4) var colorMatchingTex: texture_1d<f32>;
@group(0) @binding(5) var d65tex: texture_1d<f32>;

const xyz2rgb = transpose(mat3x3f(
// sRGB D65
 3.2404542, -1.5371385, -0.4985314,
-0.9692660,  1.8760108,  0.0415560,
 0.0556434, -0.2040259,  1.0572252

// CIE E
//  2.3706743, -0.9000405, -0.4706338,
// -0.5138850,  1.4253036,  0.0885814,
//  0.0052982, -0.0146949,  1.0093968
));

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let texel = gid.xy;
  let uv = vec2f(texel) / vec2f(1024.0);
  var value = vec3f(0.0);
  let step = max(1 / uni.steps, (uni.end_L - uni.start_L) / uni.steps);
  
  for (var l = uni.start_L; l <= uni.end_L; l += step) {
    let lookupUV = (l - 360) / (830 - 360);
    let d65 = textureSampleLevel(d65tex, texSampler, lookupUV, 0.0).r / 100.0;
    let xyz = textureSampleLevel(colorMatchingTex, texSampler, lookupUV, 0.0).xyz * step * d65;

    let uvScale = mix(1.0, uni.ref_L / l, uni.dispMult);
    // relative to center, add 0.5 for fftshift
    let sample_uv = (uv - 0.5) * uvScale / uni.scale + 0.5;
    let sample = textureSampleLevel(freqTex, texSampler, sample_uv + 0.5 * (1 + 1 / 1024.0), 0.0).rgb;
    value += xyz * sample * (uvScale * uvScale * uvScale); // account for energy conservation
  }
  textureStore(output, texel, vec4f(max(vec3f(0), xyz2rgb * value), 1));
}
`;

const renderShaderCode = /* wgsl */`
${uni.uniformStruct}

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var freqTex: texture_2d<f32>;
@group(0) @binding(2) var texSampler: sampler;

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) fragCoord: vec2f,
};

const pos = array<vec2f, 3>(
  vec2f(-1.0, -1.0),
  vec2f( 3.0, -1.0),
  vec2f(-1.0,  3.0)
);

fn linear2srgb(color: vec4f) -> vec4f {
  let cutoff = color.rgb < vec3f(0.0031308);
  let higher = 1.055 * pow(color.rgb, vec3f(1.0 / 2.4)) - 0.055;
  let lower = 12.92 * color.rgb;
  return vec4f(select(higher, lower, cutoff), color.a);
}

fn AgxDefaultContrastApprox(x: vec3f) -> vec3f {
	return (((((15.5 * x - 40.14) * x + 31.96) * x - 6.868) * x + 0.4298) * x + 0.1191) * x - 0.00232;		
}

fn AgxCurve(color: vec3f) -> vec3f {
	let hev = 14 * 0.5;
	let midGrey = 0.18;
	let c = (clamp(log2(color / midGrey), vec3f(-hev), vec3f(hev)) + hev) / 14;
	return AgxDefaultContrastApprox(c);
}

fn AgX(c: vec3f) -> vec3f {
  var color = c;
	color *= 2.3;
  // abney effect
  color *= mat3x3f(
    0.99999976, -1.26657e-7, -1.29064e-9,
    1.67316e-8, 0.99999976, -5.32026e-9,
    -0.00725587, 6.47740e-9, 1.00725580
  );

	color *= mat3x3f(
    0.842479062253094, 0.0784335999999992, 0.0792237451477643,
    0.0423282422610123, 0.878468636469772, 0.0791661274605434,
    0.0423756549057051, 0.0784336, 0.879142973793104
  );
  color = AgxCurve(color);
	color *= mat3x3f(
    1.19687900512017, -0.0980208811401368, -0.0990297440797205,
    -0.0528968517574562, 1.15190312990417, -0.0989611768448433,
    -0.0529716355144438, -0.0980434501171241, 1.15107367264116
  );
	return color;
}

@vertex
fn vs(@builtin(vertex_index) vIdx: u32) -> VertexOut {
  let currentPos = pos[vIdx];
  return VertexOut(vec4f(currentPos, 0.0, 1.0), 0.5 * (currentPos + 1.0));
}

@fragment
fn fs(input: VertexOut) -> @location(0) vec4f {
  // return log(textureSample(freqTex, texSampler, input.fragCoord * (uni.resolution / uni.resolution.y))) / 1e1;
  return (vec4f(AgX(textureSample(freqTex, texSampler, input.fragCoord).rgb * uni.gain), 1.0));
}
`;

const apertureRenderShaderCode = /* wgsl */`
${uni.uniformStruct}

// @group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var srcTex: texture_2d<f32>;
@group(0) @binding(2) var texSampler: sampler;

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) fragCoord: vec2f,
};

const pos = array<vec2f, 3>(
  vec2f(-1.0, -1.0),
  vec2f( 3.0, -1.0),
  vec2f(-1.0,  3.0)
);

@vertex
fn vs(@builtin(vertex_index) vIdx: u32) -> VertexOut {
  let currentPos = pos[vIdx];
  return VertexOut(vec4f(currentPos, 0.0, 1.0), 0.5 * (currentPos + 1.0));
}

@fragment
fn fs(input: VertexOut) -> @location(0) vec4f {
  // return log(textureSample(freqTex, texSampler, input.fragCoord * (uni.resolution / uni.resolution.y))) / 1e1;
  return textureSample(srcTex, texSampler, input.fragCoord);
}
`;