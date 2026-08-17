// Taki OS — 滝のシミュレータ / 水の流れを線で描く   (C++ / WASM)
//
// 何をしているか
// --------------
// 水を**粒子**として解いて、粒子が動いた軌跡をそのまま**線**として焼き付ける。
// 絵として滝を描いているのではなく、水が岩を落ちた結果として線が引かれる。
//
//   1) 岩      岩は高さ場 H(x,z)。落ち口(リップ)の直下で急に落ち、下で緩んで滝壺になる。
//              そこへ巨石や畝を足すと、水が自分で筋に分かれる ── 筋は描いていない。
//   2) 落下    粒子は重力で自由落下する。落差 h から着水速度は v=√(2gh) になるはずで、
//              これは**積分の結果として出てくる**（表示して確かめられるようにしてある）。
//   3) 水らしさ 粒子だけだと砂になる。格子に密度と運動量を撒いて、
//              圧力 p = K·max(0, ρ−ρ0) の勾配で押し返し、速度を近傍平均へ寄せる(粘性)。
//              これで水は面に広がり、束になって落ち、滝壺で跳ねる。O(N) で済む。
//   4) 衝突    高さ場の勾配から法線を作り、法線成分をはね返して接線成分に摩擦をかける。
//              強く当たった粒子には泡(foam)を立て、白く明るくする ── 白泡は速度の記録。
//   5) 線      毎フレーム、粒子の「前の位置 → 今の位置」を HDR 加算バッファに引く。
//              バッファは毎フレーム少しずつ減衰するので、線は軌跡として尾を引く。
//              **単位長さあたり一定量**を置くので、速く動く粒子ほど長く明るい線になる。
//
// 計算も描画もすべて C++。テキストのみ olive.c。
#define OLIVEC_IMPLEMENTATION
#include "olive.c"
#include <vector>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <chrono>
#include <algorithm>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#define KEEP EMSCRIPTEN_KEEPALIVE
#else
#define KEEP
#endif

// ---------------------------------------------------------------- 画面と時間
static const int FW = 960, FH = 600;
static const int SUBSTEPS = 3;
static const float DT = 1.0f / (60.0f * SUBSTEPS);

// ---------------------------------------------------------------- 世界の寸法 [m]
static const float XR = 13.0f;          // 左右は ±XR
static const float ZR = 34.0f;          // 手前 0（滝壺）〜 奥 ZR（上流）
static const float LIP_Z = 26.0f;       // 落ち口
static const float BASE_Z = 20.0f;      // 滝壺の縁
static const float TOP_H = 15.5f;       // 落ち口の高さ
static const float POOL_H = 1.2f;       // 滝壺の底

static const size_t MAXP = 220000;

// ---------------------------------------------------------------- 乱数
static uint32_t rng = 88675123u;
static inline float rnd() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return (rng & 0xFFFFFF) / (float)0x1000000; }
static inline float rnds() { return rnd() * 2.0f - 1.0f; }
static inline float clampf(float v, float a, float b) { return v < a ? a : (v > b ? b : v); }
static inline float smoothstep01(float t) { t = clampf(t, 0, 1); return t * t * (3.0f - 2.0f * t); }
static inline uint32_t rgb(int r, int g, int b) {
    return 0xFF000000u | ((uint32_t)b << 16) | ((uint32_t)g << 8) | (uint32_t)r;
}

// ================================================================ 岩（高さ場）
// H(x,z) を格子に焼いておいて双一次で読む。勾配も焼く（法線に使う）。
static const int TGX = 321, TGZ = 385;
static std::vector<float> tH, tGx, tGz;
static float tdx, tdz;

// 岩の形のプリセット
static int p_rock = 2;

static float terrain_raw(float x, float z) {
    float h;
    // --- 縦の断面 ---
    // 落ち口の直下 3.2m ぶんを一気に落として「壁」にする。水はそこに触れずに自由落下する。
    // 高さ場なので真の垂直は作れないが、勾配 6 前後（80度）あれば水は面を離れて落ちる。
    const float ZF = LIP_Z - 3.4f;      // 壁の下端
    const float LEDGE = 2.0f;           // 壁の下端の高さ
    if (z >= LIP_Z) {
        h = TOP_H + (z - LIP_Z) * 0.055f;                  // 上流のゆるい川床
    } else if (z >= ZF) {
        float u = (LIP_Z - z) / (LIP_Z - ZF);              // 0=落ち口 1=壁の下端
        h = TOP_H - (TOP_H - LEDGE) * powf(u, 0.55f);
    } else if (z >= BASE_Z) {
        float v = (ZF - z) / (ZF - BASE_Z);
        h = LEDGE - (LEDGE - POOL_H) * smoothstep01(v);    // 壁の下は緩い瀬
    } else {
        // 滝壺：椀。手前(z→0)に土手を立てて水を溜める
        float bowl = 1.4f * expf(-(((z - 16.0f) * (z - 16.0f)) * 0.35f + x * x * 0.42f) / 20.0f);
        h = POOL_H - bowl;
        h += 4.0f * smoothstep01((6.0f - z) / 5.0f);       // 手前の土手
    }
    // --- 左右の岸（水を谷に集める） ---
    float bank = fabsf(x) - 8.5f;
    if (bank > 0) h += bank * bank * 0.85f;

    // --- プリセットごとの造形 ---
    auto bump = [&](float bx, float bz, float amp, float sx, float sz) {
        float ax = (x - bx) / sx, az = (z - bz) / sz;
        h += amp * expf(-(ax * ax + az * az));
    };
    // 溝（ガリー）: 落ち口から下へ走る窪み。水はここに集まって「筋」になる。
    // ★筋を描いているのではなく、地形が水を集めている。
    auto gully = [&](float gxc, float depth, float wid, float z0, float z1) {
        float a = (x - gxc) / wid;
        float zz = smoothstep01((z - z0) / 2.5f) * (1.0f - smoothstep01((z - z1) / 3.0f));
        h -= depth * expf(-a * a) * zz;
    };
    switch (p_rock) {
    case 0:   // 一枚の滝 — 落ち口をまっすぐにして幅広の水膜にする
        gully(0.0f, 0.9f, 5.0f, LIP_Z - 3.0f, LIP_Z + 5.0f);
        break;
    case 1:   // 段瀑 — 途中に段を作って二段・三段に落とす
        h += 2.4f * smoothstep01((z - 19.0f) / 1.2f) * (1.0f - smoothstep01((z - 21.6f) / 1.2f));
        h += 1.6f * smoothstep01((z - 14.0f) / 1.1f) * (1.0f - smoothstep01((z - 16.4f) / 1.1f));
        gully(-1.0f, 0.8f, 3.0f, LIP_Z - 3.0f, LIP_Z + 5.0f);
        break;
    case 2:   // 分岐 — 巨石で水を筋に割る（既定）
        gully(-4.2f, 1.3f, 1.7f, LIP_Z - 3.0f, LIP_Z + 5.0f);   // 三本の溝 → 三筋の滝
        gully( 0.2f, 1.5f, 1.9f, LIP_Z - 3.0f, LIP_Z + 5.0f);
        gully( 4.6f, 1.2f, 1.6f, LIP_Z - 3.0f, LIP_Z + 5.0f);
        break;
    case 3:   // 樋（とい）— 中央を彫って一本の太い流れにする
        gully(0.0f, 2.4f, 1.9f, LIP_Z - 3.0f, LIP_Z + 6.0f);    // 一本の太い樋
        break;
    default:  // 岩場 — 小さい岩を散らして水を暴れさせる
        for (int i = 0; i < 14; ++i) {
            float a = i * 2.399963f;
            float bx = cosf(a) * (1.5f + 5.0f * ((i * 37 % 11) / 11.0f));
            float bz = LIP_Z + 1.0f + 6.0f * ((i * 53 % 17) / 17.0f);
            bump(bx, bz, 0.9f + 1.0f * ((i * 29 % 7) / 7.0f), 1.3f, 1.3f);
        }
        break;
    }
    return h;
}

static void build_terrain() {
    tH.assign((size_t)TGX * TGZ, 0.0f);
    tGx.assign((size_t)TGX * TGZ, 0.0f);
    tGz.assign((size_t)TGX * TGZ, 0.0f);
    tdx = 2.0f * XR / (TGX - 1);
    tdz = ZR / (TGZ - 1);
    for (int j = 0; j < TGZ; ++j) {
        float z = j * tdz;
        for (int i = 0; i < TGX; ++i) {
            float x = -XR + i * tdx;
            tH[(size_t)j * TGX + i] = terrain_raw(x, z);
        }
    }
    for (int j = 0; j < TGZ; ++j) for (int i = 0; i < TGX; ++i) {
        int i0 = i > 0 ? i - 1 : i, i1 = i < TGX - 1 ? i + 1 : i;
        int j0 = j > 0 ? j - 1 : j, j1 = j < TGZ - 1 ? j + 1 : j;
        tGx[(size_t)j * TGX + i] = (tH[(size_t)j * TGX + i1] - tH[(size_t)j * TGX + i0]) / ((i1 - i0) * tdx);
        tGz[(size_t)j * TGX + i] = (tH[(size_t)j1 * TGX + i] - tH[(size_t)j0 * TGX + i]) / ((j1 - j0) * tdz);
    }
}

static inline void terrain(float x, float z, float& h, float& gx, float& gz) {
    float fi = (x + XR) / tdx, fj = z / tdz;
    int i = (int)fi, j = (int)fj;
    if (i < 0) i = 0; if (i > TGX - 2) i = TGX - 2;
    if (j < 0) j = 0; if (j > TGZ - 2) j = TGZ - 2;
    float ax = clampf(fi - i, 0, 1), az = clampf(fj - j, 0, 1);
    size_t a = (size_t)j * TGX + i, b = a + 1, c = a + TGX, d = c + 1;
    float w0 = (1 - ax) * (1 - az), w1 = ax * (1 - az), w2 = (1 - ax) * az, w3 = ax * az;
    h  = tH[a] * w0 + tH[b] * w1 + tH[c] * w2 + tH[d] * w3;
    gx = tGx[a] * w0 + tGx[b] * w1 + tGx[c] * w2 + tGx[d] * w3;
    gz = tGz[a] * w0 + tGz[b] * w1 + tGz[c] * w2 + tGz[d] * w3;
}

// ================================================================ 水の粒子
struct Drop {
    float x, y, z;
    float vx, vy, vz;
    float sx, sy;        // 前フレームの画面座標
    float foam;          // 白泡（強く当たると立ち、指数的に消える）
    float age;
    float pool;          // 滝壺に居た時間
    float ytop;          // 今回の落下の開始高さ（落差の実測用）
    uint8_t on;          // 画面座標が有効か
};
static std::vector<Drop> drops;

// ================================================================ 格子（圧力と粘性）
// これが「水らしさ」の全部。粒子を格子に撒いて、混みすぎたセルから押し返す。
static const float CELL = 0.78f;
static int GX, GY, GZ;
static const float GY_TOP = 26.0f;
static std::vector<float> gDen, gVx, gVy, gVz, gP;

static inline int gidx(int i, int j, int k) { return (k * GY + j) * GX + i; }

static void build_grid() {
    GX = (int)(2 * XR / CELL) + 2;
    GY = (int)(GY_TOP / CELL) + 2;
    GZ = (int)(ZR / CELL) + 2;
    size_t n = (size_t)GX * GY * GZ;
    gDen.assign(n, 0.0f); gVx.assign(n, 0.0f); gVy.assign(n, 0.0f); gVz.assign(n, 0.0f); gP.assign(n, 0.0f);
}

// ================================================================ カメラ
static float camAz = 0.05f, camEl = 0.32f, camR = 40.0f, camF = 1000.0f;
static float tgtX = 0.0f, tgtY = 7.0f, tgtZ = 18.0f;
static float camX, camY, camZ, cyaw, syaw, cpit, spit;

static void setup_camera() {
    camX = tgtX + camR * cosf(camEl) * sinf(camAz);
    camY = tgtY + camR * sinf(camEl);
    camZ = tgtZ - camR * cosf(camEl) * cosf(camAz);
    float yaw = -camAz;
    cyaw = cosf(yaw); syaw = sinf(yaw);
    cpit = cosf(camEl); spit = sinf(camEl);
}

static inline bool project(float x, float y, float z, float& sx, float& sy, float& depth) {
    float dx = x - camX, dy = y - camY, dz = z - camZ;
    float cx = cyaw * dx - syaw * dz;
    float cz = syaw * dx + cyaw * dz;
    float cy2 = cpit * dy + spit * cz;
    float cz2 = -spit * dy + cpit * cz;
    if (cz2 < 1.2f) return false;
    float inv = camF / cz2;
    sx = FW * 0.5f + cx * inv;
    sy = FH * 0.5f - cy2 * inv;
    depth = cz2;
    return true;
}

// ================================================================ バッファ
static std::vector<float> acc;      // HDR 加算（減衰して尾を引く）
static std::vector<float> rockBuf;  // 岩のワイヤ（カメラが動いたときだけ焼き直す）
static std::vector<uint32_t> px;
// x/(1+x) → ガンマ の合成を引くだけの表。powf を毎フレーム170万回呼ぶと重い。
static const int TMN = 4096;
static uint8_t tmLut[TMN + 1];
static void build_lut() {
    for (int i = 0; i <= TMN; ++i) {
        float t = (float)i / TMN;                 // t = x/(1+x) ∈ [0,1)
        int v = (int)(powf(t, 0.86f) * 255.0f + 0.5f);
        tmLut[i] = (uint8_t)(v > 255 ? 255 : v);
    }
}
static inline int tone(float x) {
    float t = x / (1.0f + x);
    int i = (int)(t * TMN);
    return tmLut[i < 0 ? 0 : (i > TMN ? TMN : i)];
}
static bool rockDirty = true;

// ---------------------------------------------------------------- つまみ
static float p_flow = 1.0f;      // 流量
static float p_grav = 9.80665f;  // 重力
static float p_visc = 0.35f;     // 粘り
static float p_fric = 0.32f;     // 岩の摩擦
static float p_decay = 0.83f;    // 線の残り
static float p_bright = 0.85f;    // 線の明るさ
static float p_spray = 1.0f;     // しぶき
static float p_width = 1.0f;     // 水の広がり
static float p_amount = 1.0f;    // 水の量（粒子予算の倍率）
static int   p_hud = 1;
static int   p_showRock = 1;
static int   p_color = 0;
static int   p_orbit = 0;

// ---------------------------------------------------------------- 計測
static float simTime = 0;
static float measDrop = 0;       // 落差 [m]
static float measImpact = 0;     // 着水速度 [m/s]（移動平均）
static float measVmax = 0;
static long  segCount = 0;
static float dbgMaxDen=0, dbgMeanDen=0, dbgMaxX=0, dbgMaxY=0; static int dbgCells=0, dbgWide=0, dbgHigh=0;

// ================================================================ 線を引く
// 単位長さあたり一定量を置く。太さは奥行きで細くする。
static inline void put(float fx, float fy, float r, float g, float b) {
    int ix = (int)fx, iy = (int)fy;
    if (ix < 0 || iy < 0 || ix >= FW - 1 || iy >= FH - 1) return;
    float ax = fx - ix, ay = fy - iy;
    float w[4] = { (1 - ax) * (1 - ay), ax * (1 - ay), (1 - ax) * ay, ax * ay };
    int o[4] = { iy * FW + ix, iy * FW + ix + 1, (iy + 1) * FW + ix, (iy + 1) * FW + ix + 1 };
    for (int k = 0; k < 4; ++k) {
        float* q = &acc[(size_t)o[k] * 3];
        q[0] += r * w[k]; q[1] += g * w[k]; q[2] += b * w[k];
    }
}

static inline void line(float x0, float y0, float x1, float y1, float r, float g, float b) {
    float dx = x1 - x0, dy = y1 - y0;
    float len = sqrtf(dx * dx + dy * dy);
    if (len < 0.001f) { put(x0, y0, r, g, b); return; }
    if (len > 90.0f) return;                         // ワープ（再投入など）は描かない
    int n = (int)len + 1;
    float ux = dx / n, uy = dy / n;
    for (int i = 0; i <= n; ++i) put(x0 + ux * i, y0 + uy * i, r, g, b);
    segCount += n;
}

// ================================================================ 岩のワイヤ
// 岩そのものは面で塗らない。等高線と縦筋の**線**で描く ── 水と同じ言葉で描きたいので。
static void bake_rock() {
    rockBuf.assign((size_t)FW * FH * 3, 0.0f);
    if (!p_showRock) { rockDirty = false; return; }
    std::vector<float> save; save.swap(acc);
    acc.assign((size_t)FW * FH * 3, 0.0f);
    const float cr = 0.022f, cg = 0.034f, cb = 0.055f;
    // z 方向の稜線
    for (float x = -9.0f; x <= 9.0f; x += 1.00f) {
        bool have = false; float px_ = 0, py_ = 0;
        for (float z = 2.0f; z <= LIP_Z + 5.0f; z += 0.32f) {
            float h, gx, gz; terrain(x, z, h, gx, gz);
            float sx, sy, dep;
            if (project(x, h, z, sx, sy, dep)) {
                if (have) line(px_, py_, sx, sy, cr, cg, cb);
                px_ = sx; py_ = sy; have = true;
            } else have = false;
        }
    }
    // x 方向の横筋
    for (float z = 3.0f; z <= LIP_Z + 5.0f; z += 1.25f) {
        bool have = false; float px_ = 0, py_ = 0;
        for (float x = -9.0f; x <= 9.0f; x += 0.30f) {
            float h, gx, gz; terrain(x, z, h, gx, gz);
            float sx, sy, dep;
            if (project(x, h, z, sx, sy, dep)) {
                if (have) line(px_, py_, sx, sy, cr, cg, cb);
                px_ = sx; py_ = sy; have = true;
            } else have = false;
        }
    }
    rockBuf.swap(acc);
    acc.swap(save);
    if (acc.empty()) acc.assign((size_t)FW * FH * 3, 0.0f);
    rockDirty = false;
}

// ================================================================ 粒子の投入
// 上流の川へ投入する（使い終わった粒子はここへ戻る）
static void spawn(Drop& d) {
    float w = 6.6f * p_width;
    d.x = rnds() * w;
    // ★上流の川「全体」に配る。落ち口の手前だけに詰めると、そこだけ密度が桁違いになって
    //   圧力で横に吹き飛ぶ（実際それで水が蝶の羽になった）。
    //   流量 Q = 幅×深さ×流速 が成り立つ深さで入れてやる必要がある。
    d.z = LIP_Z + 0.6f + rnd() * (ZR - LIP_Z - 1.2f);
    float h, gx, gz; terrain(d.x, d.z, h, gx, gz);
    d.y = h + 0.05f + rnd() * 1.15f;
    float v = 2.4f * p_flow;
    d.vx = rnds() * 0.3f;
    d.vy = 0.0f;
    d.vz = -v * (0.85f + 0.3f * rnd());
    d.foam = 0.0f;
    d.age = 0.0f;
    d.pool = 0.0f;
    d.ytop = d.y;
    d.on = 0;
}

// 最初から流路いっぱいに散らしておく。
// ★全部を発生点に置くと1セルに数千粒たまって圧力で爆発する（最初これで壊した）
static void scatter(Drop& d) {
    spawn(d);
    d.z = 3.5f + rnd() * (ZR - 5.0f);
    d.x = rnds() * 6.6f * p_width;
    float h, gx, gz; terrain(d.x, d.z, h, gx, gz);
    d.y = h + 0.08f + rnd() * (d.z < BASE_Z ? 1.4f : 0.3f);
    d.vz = -2.0f * p_flow;
    d.ytop = d.y; d.age = rnd() * 4.0f; d.pool = (d.z < BASE_Z) ? rnd() * 0.9f : 0.0f;
}

static size_t target_count() {
    size_t n = (size_t)(100000.0f * p_amount);
    return n > MAXP ? MAXP : (n < 4000 ? 4000 : n);
}

// ================================================================ 1 サブステップ
static void substep() {
    const size_t N = drops.size();
    if (!N) return;
    // --- 格子へ撒く ---
    std::fill(gDen.begin(), gDen.end(), 0.0f);
    std::fill(gVx.begin(), gVx.end(), 0.0f);
    std::fill(gVy.begin(), gVy.end(), 0.0f);
    std::fill(gVz.begin(), gVz.end(), 0.0f);
    for (size_t p = 0; p < N; ++p) {
        const Drop& d = drops[p];
        int i = (int)((d.x + XR) / CELL), j = (int)(d.y / CELL), k = (int)(d.z / CELL);
        if (i < 0 || j < 0 || k < 0 || i >= GX || j >= GY || k >= GZ) continue;
        int q = gidx(i, j, k);
        gDen[q] += 1.0f; gVx[q] += d.vx; gVy[q] += d.vy; gVz[q] += d.vz;
    }
    // --- 密度をならす（セル単位のムラがそのまま塊に見えるので）---
    {
        static std::vector<float> tmp;
        tmp.assign(gDen.size(), 0.0f);
        for (int k = 1; k < GZ - 1; ++k) for (int j = 1; j < GY - 1; ++j) for (int i = 1; i < GX - 1; ++i) {
            int q = gidx(i, j, k);
            tmp[q] = (gDen[q] * 2.0f
                    + gDen[q - 1] + gDen[q + 1]
                    + gDen[q - GX] + gDen[q + GX]
                    + gDen[q - GX * GY] + gDen[q + GX * GY]) * (1.0f / 8.0f);
        }
        gDen.swap(tmp);
    }
    // 診断：密度の実測
    { float mx=0; double sum=0; int nz=0;
      for(size_t q=0;q<gDen.size();++q){ if(gDen[q]>mx) mx=gDen[q]; if(gDen[q]>0.5f){sum+=gDen[q];nz++;} }
      dbgMaxDen=mx; dbgMeanDen=nz?(float)(sum/nz):0; dbgCells=nz; }
    // --- 圧力と平均速度 ---
    // ★ 水が居られる体積はだいたい 500 セルぶん。REST（押し返しが始まる密度）は
    //   粒子数から決めないといけない。固定値にすると粒子を増やしただけで爆発する。
    const float REST = clampf((float)N / 780.0f, 3.0f, 4000.0f);
    const float KP = 110.0f / REST;          // KP·REST を一定に＝水の固さを一定に保つ
    const float AMAX = 55.0f;               // 圧力加速度の頭打ち（保険）
    for (size_t q = 0; q < gDen.size(); ++q) {
        float rho = gDen[q];
        if (rho > 0) { float inv = 1.0f / rho; gVx[q] *= inv; gVy[q] *= inv; gVz[q] *= inv; }
        gP[q] = rho > REST ? KP * (rho - REST) : 0.0f;
    }
    // --- 粒子を進める ---
    const float g = p_grav;
    const float visc = clampf(p_visc, 0, 1) * 0.55f;
    const float fric = clampf(p_fric, 0, 1) * 7.0f;
    float vmax = 0, impSum = 0, dropSum = 0; int impN = 0;
    for (size_t p = 0; p < N; ++p) {
        Drop& d = drops[p];
        int i = (int)((d.x + XR) / CELL), j = (int)(d.y / CELL), k = (int)(d.z / CELL);
        if (i > 0 && j > 0 && k > 0 && i < GX - 1 && j < GY - 1 && k < GZ - 1) {
            float px_ = (gP[gidx(i + 1, j, k)] - gP[gidx(i - 1, j, k)]) / (2 * CELL);
            float py_ = (gP[gidx(i, j + 1, k)] - gP[gidx(i, j - 1, k)]) / (2 * CELL);
            float pz_ = (gP[gidx(i, j, k + 1)] - gP[gidx(i, j, k - 1)]) / (2 * CELL);
            float am = sqrtf(px_ * px_ + py_ * py_ + pz_ * pz_);
            if (am > AMAX) { float s2 = AMAX / am; px_ *= s2; py_ *= s2; pz_ *= s2; }
            d.vx -= px_ * DT; d.vy -= py_ * DT; d.vz -= pz_ * DT;
            int q = gidx(i, j, k);
            if (gDen[q] > 1.5f) {                       // 近所と速度を揃える＝粘性
                d.vx += (gVx[q] - d.vx) * visc;
                d.vy += (gVy[q] - d.vy) * visc;
                d.vz += (gVz[q] - d.vz) * visc;
            }
        }
        d.vy -= g * DT;
        d.x += d.vx * DT; d.y += d.vy * DT; d.z += d.vz * DT;

        // --- 岩との衝突 ---
        float h, gx, gz; terrain(d.x, d.z, h, gx, gz);
        if (d.y < h) {
            d.y = h;
            // 勾配は頭打ちにする。壁のように急な面でも法線が寝すぎないように。
            const float GMAX = 5.0f;
            float gm = sqrtf(gx * gx + gz * gz);
            if (gm > GMAX) { float s2 = GMAX / gm; gx *= s2; gz *= s2; }
            float nx = -gx, ny = 1.0f, nz = -gz;
            float inv = 1.0f / sqrtf(nx * nx + 1.0f + nz * nz);
            nx *= inv; ny *= inv; nz *= inv;
            float vn = d.vx * nx + d.vy * ny + d.vz * nz;
            if (vn < 0) {
                d.vx -= nx * vn; d.vy -= ny * vn; d.vz -= nz * vn;
                float imp = -vn;
                if (imp > 3.0f) {
                    d.foam = clampf(d.foam + imp * 0.09f, 0.0f, 1.6f);
                    float s = p_spray * clampf(imp * 0.020f, 0.0f, 0.55f);
                    d.vx += rnds() * s * 2.4f; d.vy += rnd() * s * 2.6f; d.vz += rnds() * s * 2.4f;
                    // ★ 落差は粒子ごとに実測する（地形から引いた見込みではなく）。
                    //   「今回いちばん高かった所」から「ぶつかった所」までが、その粒子の落差 h。
                    //   速さ |v| と h を対にして貯めれば、v と √(2gh) を直接比べられる。
                    float hh = d.ytop - d.y;
                    if (hh > 4.0f) {
                        impSum += sqrtf(d.vx * d.vx + d.vy * d.vy + d.vz * d.vz);
                        dropSum += hh; impN++;
                    }
                }
            }
            float k2 = 1.0f - fric * DT; if (k2 < 0) k2 = 0;
            d.vx *= k2; d.vz *= k2;
        }
        if (d.y > d.ytop) d.ytop = d.y;
        d.foam *= (1.0f - 1.7f * DT);
        d.age += DT;
        // ★滞留時間の配分がそのまま絵の密度になる。滝壺に居座らせると滝が痩せる。
        if (d.z < BASE_Z) d.pool += DT;

        // --- 端の始末 ---
        const float XW = 8.2f;                 // 谷の幅。ここから外へは出さない
        if (d.x < -XW) { d.x = -XW; if (d.vx < 0) d.vx *= -0.25f; }
        if (d.x >  XW) { d.x =  XW; if (d.vx > 0) d.vx *= -0.25f; }
        if (d.z < 3.0f || d.y < -1.0f || d.y > GY_TOP - 1.0f || d.pool > 1.15f || d.age > 24.0f) { spawn(d); }

        float sp = sqrtf(d.vx * d.vx + d.vy * d.vy + d.vz * d.vz);
        const float VMAX = 24.0f;            // 物理的にありえない速さは切る
        if (sp > VMAX) { float s2 = VMAX / sp; d.vx *= s2; d.vy *= s2; d.vz *= s2; sp = VMAX; }
        if (sp > vmax) vmax = sp;
    }
    if (vmax > measVmax) measVmax = vmax;
    { float mx=0,my=0; int wide=0,high=0;
      for(size_t q=0;q<N;++q){ const Drop&d=drops[q];
        if(fabsf(d.x)>mx) mx=fabsf(d.x); if(d.y>my) my=d.y;
        if(fabsf(d.x)>13.0f) wide++; if(d.y>16.5f) high++; }
      dbgMaxX=mx; dbgMaxY=my; dbgWide=wide; dbgHigh=high; }
    if (impN) {
        float v = impSum / impN, h = dropSum / impN;
        measImpact = measImpact > 0 ? measImpact * 0.985f + v * 0.015f : v;
        measDrop   = measDrop   > 0 ? measDrop   * 0.985f + h * 0.015f : h;
    }
    simTime += DT;
}

// ================================================================ 描画
static void draw() {
    // 尾を引かせる：前フレームを少し残す
    float dec = clampf(p_decay, 0.55f, 0.975f);
    for (size_t i = 0; i < acc.size(); ++i) acc[i] *= dec;

    const float bri = 0.0072f * p_bright;
    segCount = 0;
    for (size_t p = 0; p < drops.size(); ++p) {
        Drop& d = drops[p];
        float sx, sy, dep;
        bool ok = project(d.x, d.y, d.z, sx, sy, dep);
        if (ok) {
            float sp = sqrtf(d.vx * d.vx + d.vy * d.vy + d.vz * d.vz);
            float t = clampf(sp / 17.0f, 0, 1);
            float f = clampf(d.foam, 0, 1);
            float r, g, b;
            if (p_color == 1) {                       // 藍
                r = 0.10f + 0.55f * t; g = 0.20f + 0.62f * t; b = 0.52f + 0.46f * t;
            } else if (p_color == 2) {                // 玉虫
                float hgt = clampf(d.y / 16.0f, 0, 1);
                r = 0.20f + 0.75f * t * (0.4f + 0.6f * hgt);
                g = 0.34f + 0.60f * t;
                b = 0.62f + 0.38f * (1.0f - hgt) * t + 0.2f * t;
            } else {                                   // 白青（既定）
                r = 0.16f + 0.80f * t * t; g = 0.34f + 0.66f * t; b = 0.58f + 0.42f * t;
            }
            // 白泡は白く明るく
            r += (1.0f - r) * f; g += (1.0f - g) * f; b += (1.0f - b) * f;
            float fade = clampf((1.15f - d.pool) * 3.4f, 0.0f, 1.0f);   // 回収直前をなじませる
            float att = bri * (1.0f + 1.3f * f) * fade * (34.0f / (dep + 14.0f));
            r *= att; g *= att; b *= att;
            if (d.on) line(d.sx, d.sy, sx, sy, r, g, b);
            else      put(sx, sy, r, g, b);
        }
        d.sx = sx; d.sy = sy; d.on = ok ? 1 : 0;
    }

    // --- 合成（トーンマップ） ---
    for (int i = 0; i < FW * FH; ++i) {
        // 夜の谷の空気。上ほどわずかに明るい
        int yy = i / FW;
        float bg = 0.014f + 0.026f * (1.0f - yy / (float)FH);
        float r = acc[(size_t)i * 3 + 0] + rockBuf[(size_t)i * 3 + 0] + bg * 0.55f;
        float g = acc[(size_t)i * 3 + 1] + rockBuf[(size_t)i * 3 + 1] + bg * 0.80f;
        float b = acc[(size_t)i * 3 + 2] + rockBuf[(size_t)i * 3 + 2] + bg * 1.25f;
        px[i] = rgb(tone(r), tone(g), tone(b));
    }
}

// ================================================================ ABI
extern "C" {

KEEP int sim_w() { return FW; }
KEEP int sim_h() { return FH; }

KEEP void sim_reset() {
    build_terrain();
    build_grid();
    setup_camera();
    acc.assign((size_t)FW * FH * 3, 0.0f);
    rockDirty = true;
    drops.assign(target_count(), Drop());
    for (auto& d : drops) scatter(d);
    // 落差＝落ち口の高さ − 滝壺の水面
    measDrop = 0; measImpact = 0; measVmax = 0; simTime = 0;
}

KEEP int sim_init(int seed, int) {
    if (seed == 0) {
        uint64_t t = (uint64_t)std::chrono::system_clock::now().time_since_epoch().count();
        seed = (int)(t ^ (t >> 32));
    }
    rng = (uint32_t)seed | 1u;
    for (int i = 0; i < 8; ++i) rnd();
    build_lut();
    px.assign((size_t)FW * FH, 0);
    rockBuf.assign((size_t)FW * FH * 3, 0.0f);
    sim_reset();
    return 1;
}

KEEP void sim_set(int id, double v) {
    switch (id) {
    case 0: {                                     // 水の量
        float old = p_amount; p_amount = (float)v;
        if (fabsf(old - p_amount) > 1e-6f) {
            size_t n = target_count(), cur = drops.size();
            if (n > cur) { drops.resize(n); for (size_t i = cur; i < n; ++i) scatter(drops[i]); }
            else drops.resize(n);
        }
        break;
    }
    case 1: p_flow = (float)v; break;
    case 2: p_grav = (float)v; break;
    case 3: p_visc = (float)v; break;
    case 4: p_fric = (float)v; break;
    case 5: p_decay = (float)v; break;
    case 6: p_bright = (float)v; break;
    case 7: camAz = (float)v * 0.0174532925f; setup_camera(); rockDirty = true; break;
    case 8: camEl = (float)v * 0.0174532925f; setup_camera(); rockDirty = true; break;
    case 9: {                                     // 岩の形
        int r = (int)floor(v + 0.5);
        if (r != p_rock) { p_rock = r; build_terrain(); rockDirty = true;
                           for (auto& d : drops) scatter(d);
                           acc.assign((size_t)FW * FH * 3, 0.0f);
                           measDrop = 0; measImpact = 0; measVmax = 0; }
        break;
    }
    case 10: p_spray = (float)v; break;
    case 11: p_width = (float)v; break;
    case 12: camR = (float)v; setup_camera(); rockDirty = true; break;
    case 13: p_hud = v > 0.5 ? 1 : 0; break;
    case 14: p_showRock = v > 0.5 ? 1 : 0; rockDirty = true; break;
    case 15: p_color = (int)floor(v + 0.5); break;
    case 16: p_orbit = v > 0.5 ? 1 : 0; break;
    default: break;
    }
}

KEEP void sim_action(int id) {
    switch (id) {
    case 0: sim_reset(); break;
    case 1: p_showRock = !p_showRock; rockDirty = true; break;
    case 2: p_color = (p_color + 1) % 3; break;
    case 3: p_orbit = !p_orbit; break;
    default: break;
    }
}

KEEP double sim_get(int id) {
    switch (id) {
    case 0: return (double)drops.size();
    case 1: return (double)segCount;
    case 2: return measVmax;
    case 3: return measDrop;
    case 4: return measImpact;
    case 5: return sqrt(2.0 * p_grav * (measDrop > 0 ? measDrop : 0));   // √(2gh)
    case 6: return p_showRock;
    case 7: return p_color;
    case 8: return p_orbit;
    case 9: return camAz * 57.29577951;
    case 10: return simTime;
    default: return 0;
    }
}

KEEP void sim_click(double nx, double ny) {
    // クリックした先へ石を投げ込む気分で、その付近の水を跳ね上げる
    float tx = (float)(nx * 2.0 - 1.0) * 14.0f;
    float tz = (float)(1.0 - ny) * ZR;
    for (auto& d : drops) {
        float ddx = d.x - tx, ddz = d.z - tz;
        float r2 = ddx * ddx + ddz * ddz;
        if (r2 < 25.0f) {
            float w = 1.0f - r2 / 25.0f;
            d.vy += 9.0f * w; d.vx += ddx * 0.9f * w; d.vz += ddz * 0.9f * w;
            d.foam = clampf(d.foam + w, 0.0f, 1.6f);
        }
    }
}

KEEP void sim_step(int n) {
    for (int f = 0; f < n; ++f) {
        if (p_orbit) { camAz += 0.0022f; if (camAz > 6.2831853f) camAz -= 6.2831853f; setup_camera(); rockDirty = true; }
        for (int s = 0; s < SUBSTEPS; ++s) substep();
    }
    if (rockDirty) bake_rock();
    draw();
}

KEEP uint32_t* sim_render() {
    if (p_hud) {
        char buf[192];
        Olivec_Canvas oc = olivec_canvas(px.data(), FW, FH, FW);
        snprintf(buf, sizeof buf, "TAKI  drops %zu  drop %.1fm  impact %.1f m/s  sqrt(2gh) %.1f",
                 drops.size(), measDrop, measImpact, sqrt(2.0 * p_grav * measDrop));
        olivec_text(oc, buf, 14, 12, olivec_default_font, 2, rgb(150, 200, 235));
        snprintf(buf, sizeof buf, "vmax %.1f m/s   segments %ld", measVmax, segCount);
        olivec_text(oc, buf, 14, FH - 26, olivec_default_font, 2, rgb(90, 130, 165));
    }
    return px.data();
}

}  // extern "C"

// ================================================================ ネイティブ自己テスト
#ifndef __EMSCRIPTEN__
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cstdio>
#include <cstdlib>
int main(int argc, char** argv) {
    int frames = argc > 1 ? atoi(argv[1]) : 420;
    int rock = argc > 2 ? atoi(argv[2]) : 2;
    const char* out = argc > 3 ? argv[3] : "preview.png";
    sim_init(12345, 0);
    sim_set(9, rock);
    if (const char* e = getenv("CAM")) {            // CAM="az el R f" で視点を変える（診断用）
        float az, el, R, f;
        if (sscanf(e, "%f %f %f %f", &az, &el, &R, &f) == 4) {
            camAz = az * 0.0174532925f; camEl = el * 0.0174532925f; camR = R; camF = f;
            setup_camera(); rockDirty = true;
        }
    }
    if (const char* e = getenv("DECAY")) { p_decay = (float)atof(e); }
    for (int i = 0; i < frames; ++i) sim_step(1);
    sim_render();
    std::vector<uint8_t> rgbbuf((size_t)FW * FH * 3);
    for (int i = 0; i < FW * FH; ++i) {
        uint32_t c = px[i];
        rgbbuf[(size_t)i * 3 + 0] = c & 0xFF;
        rgbbuf[(size_t)i * 3 + 1] = (c >> 8) & 0xFF;
        rgbbuf[(size_t)i * 3 + 2] = (c >> 16) & 0xFF;
    }
    stbi_write_png(out, FW, FH, 3, rgbbuf.data(), FW * 3);
    printf("%s  drops=%zu drop=%.2fm impact=%.2f sqrt(2gh)=%.2f vmax=%.2f seg=%ld\n"
           "     REST=%.1f maxDen=%.0f meanDen=%.1f wetCells=%d\n",
           out, drops.size(), measDrop, measImpact, sqrt(2.0 * p_grav * measDrop), measVmax, segCount,
           drops.size() / 780.0, dbgMaxDen, dbgMeanDen, dbgCells);
    printf("     maxX=%.1f (channel 13)  maxY=%.1f (lip 15.5)  |x|>13: %d  y>16.5: %d\n",
           dbgMaxX, dbgMaxY, dbgWide, dbgHigh);
    return 0;
}
#endif
