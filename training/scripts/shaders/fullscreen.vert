out vec2 uv;  // 描く先の上での位置(描く先の中では0〜1。左下が(0, 0))
void main() {
    // 描く先全体を覆う三角形1つ。頂点のデータは使わず、頂点の番号0・1・2から、位置 (0, 0)・(2, 0)・(0, 2) を作る
    // (描く先の四角(0〜1)をはみ出して覆い、はみ出した部分はGPUが切り捨てる)
    vec2 position = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    uv = position;
    gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}
