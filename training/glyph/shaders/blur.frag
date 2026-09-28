// 縦か横の1方向のガウスのぼかし(輪郭のやわらげと発光に使う)。元の画像の外は0として読む。
// 発光の画像は、光が字の画像の外へ広がっても切れないよう、周りに余白を持たせて作る(source_scale・source_offset で元の画像の外まで読む)
in vec2 uv;
out vec4 color;
uniform sampler2D source;
uniform vec2 source_scale;
uniform vec2 source_offset;
uniform vec2 blur_step;
uniform float sigma;
uniform int radius;

bool in_unit_square(vec2 position) {
    return all(greaterThanEqual(position, vec2(0.0))) && all(lessThanEqual(position, vec2(1.0)));
}

float source_at(vec2 position) {
    return in_unit_square(position) ? texture(source, position).r : 0.0;
}

void main() {
    vec2 position = uv * source_scale + source_offset;
    float total = 0.0;
    float weights = 0.0;
    for (int i = -radius; i <= radius; i++) {
        float weight = exp(-float(i * i) / (2.0 * sigma * sigma));
        total += weight * source_at(position + float(i) * blur_step);
        weights += weight;
    }
    color = vec4(total / weights, 0.0, 0.0, 1.0);
}
