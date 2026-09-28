// 字を画面に描く: くっきりさせる(境界を溶かすノイズつき) → 色づけ → 発光を足す → 背景のノイズを重ねる。
// 輪郭のやわらげと発光のぼかしは、拡大する前の小さな画像の上で先にかけてある(blur.frag)
in vec2 uv;
out vec4 color;

// 字の画像の置き方: 画面の中央の正方形(短い辺を一辺とする)に対する、画面全体の大きさの比と、正方形に対する字の画像の大きさ
uniform vec2 square_scale;
uniform float image_fraction;
// 輪郭: やわらげたインクの濃さ(拡大する前の画像の大きさ)と、くっきりさせる切り替えの中心・幅
uniform sampler2D soft_ink;
uniform float edge_center;
uniform float edge_width;
// ノイズ: 同じ1つの模様で、字の境界を背景に溶け込ませ、背景に粒を重ねる。
// 模様は、格子を2通りにずらした模様の間を noise_blend でなめらかに移り変わらせて、ゆっくり動かす
uniform float noise_grain;
uniform vec2 noise_offsets[2];
uniform float noise_blend;
uniform float edge_noise_amount;
uniform float background_noise_amount;
// 色
uniform vec3 background_color;
uniform vec3 ink_color;
// 発光: 周りに余白を持たせた発光の画像と、その上での字の画像の位置
uniform sampler2D glow;
uniform vec2 glow_scale;
uniform vec2 glow_offset;
uniform vec3 glow_color;
uniform float glow_strength;

uint lowbias32(uint value) {
    // 整数のハッシュ(入力の1ビットの違いが、出力の全ビットに偏りなく広がる)。
    // Chris Wellons の hash-prospector(https://github.com/skeeto/hash-prospector、Unlicense)の lowbias32
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

float random(vec2 cell) {
    // 格子の点ごとに、0〜1の一様な乱数。整数のハッシュで作る(sin を使う簡易な乱数は、値が大きいと精度が落ちて斜めの縞が出たため)。
    // 2つの座標は、縦のハッシュに横を混ぜてから、もう1度ハッシュする
    uvec2 bits = uvec2(ivec2(cell));
    return float(lowbias32(bits.x ^ lowbias32(bits.y))) / 4294967295.0;  // uint の最大値で割って0〜1にする
}

float smooth_noise(vec2 position, vec2 offset) {
    // 格子の点の乱数を、なめらかにつないだ揺らぎ(0〜1)。画素ごとにばらばらな乱数より、ざらつきが少ない。
    // offset は格子をずらす整数(番号を1ずつ足すと、同じ模様が斜めに流れて見えるので、大きくばらばらにずらす)
    vec2 cell = floor(position) + offset;
    vec2 blend = smoothstep(0.0, 1.0, fract(position));
    float bottom = mix(random(cell), random(cell + vec2(1.0, 0.0)), blend.x);
    float top = mix(random(cell + vec2(0.0, 1.0)), random(cell + vec2(1.0, 1.0)), blend.x);
    return mix(bottom, top, blend.y);
}

float noise_pattern() {
    // 今のフレームのノイズの模様(0〜1)
    vec2 position = gl_FragCoord.xy / noise_grain;
    return mix(smooth_noise(position, noise_offsets[0]), smooth_noise(position, noise_offsets[1]), smoothstep(0.0, 1.0, noise_blend));
}

vec2 image_position_of(vec2 screen_uv) {
    // 画面全体の位置を、中央の正方形の上の位置に直し、字の画像を中央に縮めて置いたときの画像の上での位置にする
    // (画像の1行目が上になるよう、縦を反転する)
    vec2 square_position = (screen_uv - 0.5) * square_scale + 0.5;
    vec2 image_position = (square_position - 0.5) / image_fraction + 0.5;
    return vec2(image_position.x, 1.0 - image_position.y);
}

bool in_unit_square(vec2 position) {
    return all(greaterThanEqual(position, vec2(0.0))) && all(lessThanEqual(position, vec2(1.0)));
}

void main() {
    vec2 image_position = image_position_of(uv);
    float pattern = noise_pattern();
    float soft = in_unit_square(image_position) ? texture(soft_ink, image_position).r : 0.0;
    float edge_noise = (pattern - 0.5) * 2.0 * edge_noise_amount;  // 模様(0〜1)を、-1〜1の揺らぎに直して振れ幅を掛ける
    float ink_amount = smoothstep(edge_center - edge_width, edge_center + edge_width, soft + edge_noise);
    vec2 glow_position = image_position * glow_scale + glow_offset;
    float halo = in_unit_square(glow_position) ? texture(glow, glow_position).r : 0.0;
    vec3 rgb = mix(background_color, ink_color, ink_amount) + glow_strength * halo * glow_color;
    rgb += background_noise_amount * pattern;
    color = vec4(min(rgb, vec3(1.0)), 1.0);
}
