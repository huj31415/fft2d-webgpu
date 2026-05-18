const imgPreprocessShaderCode = /* wgsl */`
${uni.uniformStruct}

// @group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var input: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba32float, write>;

const PI = 3.14159265359;
const TAU = 6.28318530718;
const N = 1024u;    // (image + kernel - 1), power of 4
const r = N >> 1u; // N / 2

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let pos = gid.xy;

  let pixel = textureLoad(input, pos, 0);
  let dist = (distance(vec2f(pos), vec2f(vec2u(r))) / f32(r));
  // let window = (0.4243801 - 0.4973406 * cos(PI * (dist - 1.0)) + 0.0782793 * cos(TAU * (dist - 1.0)));
  // let window = 0.5 - 0.5 * cos(PI * (dist - 1.0));
  // let window = smoothstep(1, 0.8, dist);
  let window = 1.0;
  textureStore(output, pos, pixel * window);
}
`;

const computeShaderCode = /* wgsl */`
// enable f16;
// enable subgroups;
// can run first log2(subgroup_size) passes with subgroups

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
  let bank_bits = linear_index & 31u;
  let block_bits = linear_index >> 5u;
  
  let swizzled_bank = bank_bits ^ (block_bits & 31u);
  
  // Reconstruct the index with the scrambled bank bits
  return (linear_index & ~31u) | swizzled_bank;
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

fn fftR4(x: u32, Ns: u32, i: u32) {
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
  // let shift = N >> 1u;
  // let modulo = vec2u(N - 1u);
  row[0][swizzle(x0)] = vec2f(textureLoad(input, (vec2u(x0, y)), 0).rg); // + shift) & modulo
  row[0][swizzle(x1)] = vec2f(textureLoad(input, (vec2u(x1, y)), 0).rg); // + shift) & modulo
  row[0][swizzle(x2)] = vec2f(textureLoad(input, (vec2u(x2, y)), 0).rg); // + shift) & modulo
  row[0][swizzle(x3)] = vec2f(textureLoad(input, (vec2u(x3, y)), 0).rg); // + shift) & modulo
  workgroupBarrier();

  // first src is idx 0
  var src = 0u;
  for (var Ns = 4u; Ns <= N; Ns <<= 2u) {
    fftR4(x0, Ns, src);
    src = 1u - src;
    workgroupBarrier();
  }

  // store combination of r and g FFT output
  textureStore(freqTex, gid.xy,       vec4f(row[src][swizzle(x0)], 0, 1));
  textureStore(freqTex, vec2u(x1, y), vec4f(row[src][swizzle(x1)], 0, 1));
  textureStore(freqTex, vec2u(x2, y), vec4f(row[src][swizzle(x2)], 0, 1));
  textureStore(freqTex, vec2u(x3, y), vec4f(row[src][swizzle(x3)], 0, 1));
}

@compute @workgroup_size(1, N4_1)
fn colFFT_r4(@builtin(global_invocation_id) gid: vec3u) {
  let x = gid.x;  // 0 to N
  let y0 = gid.y; // 0 to N/4
  let y1 = y0 + N4_1;
  let y2 = y1 + N4_1;
  let y3 = y2 + N4_1;

  // load rg as real and imag
  // let shift = N >> 1u;
  // let modulo = vec2u(N - 1u);
  row[0][swizzle(y0)] = textureLoad(input, (vec2u(x, y0)), 0).rg; //  + shift) & modulo
  row[0][swizzle(y1)] = textureLoad(input, (vec2u(x, y1)), 0).rg; //  + shift) & modulo
  row[0][swizzle(y2)] = textureLoad(input, (vec2u(x, y2)), 0).rg; //  + shift) & modulo
  row[0][swizzle(y3)] = textureLoad(input, (vec2u(x, y3)), 0).rg; //  + shift) & modulo
  workgroupBarrier();

  // first src is idx 0
  var src = 0u;
  for (var Ns = 4u; Ns <= N; Ns <<= 2u) {
    fftR4(y0, Ns, src);
    src = 1u - src;
    workgroupBarrier();
  }
  
  // store combination of r and g FFT output
  textureStore(freqTex, gid.xy,       vec4f(dot(row[src][swizzle(y0)], row[src][swizzle(y0)])));
  textureStore(freqTex, vec2u(x, y1), vec4f(dot(row[src][swizzle(y1)], row[src][swizzle(y1)])));
  textureStore(freqTex, vec2u(x, y2), vec4f(dot(row[src][swizzle(y2)], row[src][swizzle(y2)])));
  textureStore(freqTex, vec2u(x, y3), vec4f(dot(row[src][swizzle(y3)], row[src][swizzle(y3)])));
}
`;

const dispersionShaderCode = /* wgsl */`
${uni.uniformStruct}

@group(0) @binding(1) var freqTex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba32float, write>;

fn linear2srgb(color: vec4f) -> vec4f {
  let cutoff = color.rgb < vec3f(0.0031308);
  let higher = 1.055 * pow(color.rgb, vec3f(1.0 / 2.4)) - 0.055;
  let lower = 12.92 * color.rgb;
  return vec4f(select(higher, lower, cutoff), color.a);
}

// https://www.baeldung.com/cs/rgb-color-light-frequency
fn wavelengthToColor(l: f32) -> vec4f {
  let Xt = vec3f(
    (l - 442.0) * select(0.0374, 0.0624, l < 442.0),
    (l - 599.8) * select(0.0323, 0.0264, l < 599.8),
    (l - 501.1) * select(0.0382, 0.0490, l < 501.1)
  );
  let x = dot(exp(-0.5 * Xt * Xt), vec3f(0.362, 1.056, -0.065));

  let Yt = vec2f(
    (l - 568.8) * select(0.0247, 0.0213, l < 568.8),
    (l - 530.9) * select(0.0322, 0.0613, l < 530.9)
  );
  let y = dot(exp(-0.5 * Yt * Yt), vec2f(0.821, 0.286));
  
  let Zt = vec2f(
    (l - 437.0) * select(0.0278, 0.0845, l < 437.0),
    (l - 459.0) * select(0.0725, 0.0385, l < 459.0)
  );
  let z = dot(exp(-0.5 * Zt * Zt), vec2f(1.217, 0.681));

  return vec4f(
     3.2406255 * x - 1.5372080 * y - 0.4986286 * z,
    -0.9689307 * x + 1.8757561 * y + 0.0415175 * z,
     0.0557101 * x - 0.2040211 * y + 1.0569959 * z,
     1.0
  );
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let pos = gid.xy;
  let value = textureLoad(freqTex, pos, 0).r;
  textureStore(output, pos, vec4f(value, value, value, 1));
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

@vertex
fn vs(@builtin(vertex_index) vIdx: u32) -> VertexOut {
  let currentPos = pos[vIdx];
  return VertexOut(vec4f(currentPos, 0.0, 1.0), 0.5 * (currentPos + 1.0));
}

@fragment
fn fs(input: VertexOut) -> @location(0) vec4f {
  return log(textureSample(freqTex, texSampler, input.fragCoord * (uni.resolution / uni.resolution.y) + 0.5)) / 1e1;
  // return (textureSample(freqTex, texSampler, input.fragCoord * (uni.resolution / uni.resolution.y)));
}
`;