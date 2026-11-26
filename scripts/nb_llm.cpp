// Greedy decode loop for the single-token SmolLM2 .nb (see llm_build.py). CPU does embedding lookup, RoPE tables, argmax.
//   nb_llm -m model.nb -e embed.f32 -p prompt_ids.txt -c 128 -n 40 -fl x,cos,sin,mask,pk,pv -eos 2[,..] [-float]
// input dtype int16 (dynamic fixed point, fractional lengths from -fl) unless -float.
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <stdint.h>
#include "awnn_inference/session.h"
using clk = std::chrono::steady_clock;
static double ms(clk::time_point a) { return std::chrono::duration<double, std::milli>(clk::now() - a).count(); }
int main(int argc, char** argv) {
    std::string model, emb, pf, fls = "0,0,0,0,0,0", eos = "2"; int C = 128, nnew = 40; bool flt = false, half = false; std::string prefix, dump; int dumpat = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-m") model = argv[++i]; else if (a == "-e") emb = argv[++i]; else if (a == "-p") pf = argv[++i];
        else if (a == "-c") C = atoi(argv[++i]); else if (a == "-n") nnew = atoi(argv[++i]); else if (a == "-fl") fls = argv[++i];
        else if (a == "-eos") eos = argv[++i]; else if (a == "-pre") prefix = argv[++i]; else if (a == "-dump") dump = argv[++i]; else if (a == "-dumpat") dumpat = atoi(argv[++i]); else if (a == "-float") flt = true; else if (a == "-half") half = true;
    }
    const int DM = 576, HD = 64, L3 = 90, V = 49152;
    int fl[6]; { std::stringstream ss(fls); std::string t; int k = 0; while (std::getline(ss, t, ',') && k < 6) fl[k++] = atoi(t.c_str()); }
    std::set<int> eosset; { std::stringstream ss(eos); std::string t; while (std::getline(ss, t, ',')) eosset.insert(atoi(t.c_str())); }
    std::vector<int> ids; { std::ifstream f(pf); int v; while (f >> v) ids.push_back(v); }
    std::vector<float> E((size_t)V * DM); { std::ifstream f(emb, std::ios::binary); f.read((char*)E.data(), E.size() * 4); if (!f) { printf("embed read failed\n"); return 1; } }
    awnn::Session s; auto t0 = clk::now(); if (!s.load(model)) { printf("load failed\n"); return 2; }
    printf("loaded %.0f ms\n", ms(t0));
    const int KVN = L3 * (C - 1) * HD; // elements per kv input
    // quantised caches kept in the input domain
    std::vector<int16_t> pk(KVN, 0), pv(KVN, 0);
    std::vector<float> pkf(flt ? KVN : 0, 0), pvf(flt ? KVN : 0, 0);
    auto q16 = [&](float x, int f) { float r = roundf(x * (float)(1 << f)); if (r > 32767) r = 32767; if (r < -32768) r = -32768; return (int16_t)r; };
    auto f2h = [](float f) -> uint16_t { union { float f; uint32_t u; } x{f}; uint32_t sg = (x.u >> 16) & 0x8000; uint32_t a = x.u & 0x7fffffff;
        if (a >= 0x47800000) return sg | 0x7bff;                       // overflow -> max finite
        if (a < 0x33000000) return sg;                                  // underflow -> 0
        if (a < 0x38800000) { uint32_t m = (a & 0x7fffff) | 0x800000; int sh = 126 - (a >> 23) + 13 - 13 + 0; sh = 126 - (int)(a >> 23); uint32_t r = m >> (sh + 1 - 0 + 13 - 13 + 0); (void)r;
            int shift = 125 - (int)(a >> 23) + 14; uint32_t v = m >> shift; uint32_t rem = m & ((1u << shift) - 1), half_ = 1u << (shift - 1); if (rem > half_ || (rem == half_ && (v & 1))) ++v; return sg | v; }
        uint32_t v = ((a - 0x38000000) >> 13); uint32_t rem = a & 0x1fff; if (rem > 0x1000 || (rem == 0x1000 && (v & 1))) ++v; return sg | v; };
    auto put = [&](int idx, const std::vector<float>& v) -> bool {
        if (flt) return s.copy_input(idx, v.data(), v.size() * 4);
        if (half) { std::vector<uint16_t> h(v.size()); for (size_t i = 0; i < v.size(); ++i) h[i] = f2h(v[i]); return s.copy_input(idx, h.data(), h.size() * 2); }
        if (idx == 3) { std::vector<uint16_t> h(v.size()); for (size_t i = 0; i < v.size(); ++i) h[i] = f2h(v[i]); return s.copy_input(idx, h.data(), h.size() * 2); }
        std::vector<int16_t> q(v.size()); for (size_t i = 0; i < v.size(); ++i) q[i] = q16(v[i], fl[idx]);
        return s.copy_input(idx, q.data(), q.size() * 2);
    };
    int pos = 0;
    if (!prefix.empty()) {   // preloaded KV of the constant chat prefix (computed offline in float): int32 P, pk[90][P][64], pv[90][P][64]
        std::ifstream f(prefix, std::ios::binary); int32_t P; f.read((char*)&P, 4); std::vector<float> a((size_t)L3 * P * HD), b(a.size());
        f.read((char*)a.data(), a.size() * 4); f.read((char*)b.data(), b.size() * 4); if (!f) { printf("prefix read failed\n"); return 1; }
        for (int l = 0; l < L3; ++l) for (int t = 0; t < P; ++t) for (int d = 0; d < HD; ++d) {
            size_t o = ((size_t)l * (C - 1) + t) * HD + d, i = ((size_t)l * P + t) * HD + d;
            if (flt) { pkf[o] = a[i]; pvf[o] = b[i]; } else if (half) { pk[o] = (int16_t)f2h(a[i]); pv[o] = (int16_t)f2h(b[i]); } else { pk[o] = q16(a[i], fl[4]); pv[o] = q16(b[i], fl[5]); }
        }
        pos = P; printf("prefix %d tokens preloaded\n", P);
    }
    std::vector<float> x(DM), cs(HD), sn(HD), mask(C);
    std::vector<int> out; int cur = ids[pos];
    double t_dec = 0, t_pre = 0; int n_dec = 0, n_pre = 0;
    while (pos < C - 1 && (int)out.size() < nnew) {
        auto ts = clk::now();
        memcpy(x.data(), &E[(size_t)cur * DM], DM * 4);
        for (int j = 0; j < HD / 2; ++j) { double f = pos * pow(100000.0, -2.0 * j / HD); cs[j] = cs[j + 32] = cos(f); sn[j] = sn[j + 32] = sin(f); }
        for (int j = 0; j < C; ++j) mask[j] = (j < pos || j == C - 1) ? 0.f : -30.f;
        bool ok = put(0, x) && put(1, cs) && put(2, sn) && put(3, mask);
        if (flt) ok = ok && s.copy_input(4, pkf.data(), KVN * 4) && s.copy_input(5, pvf.data(), KVN * 4);
        else ok = ok && s.copy_input(4, pk.data(), KVN * 2) && s.copy_input(5, pv.data(), KVN * 2);
        if (!ok) { printf("copy_input failed\n"); return 3; }
        awnn::RunProfile p; if (!s.run(&p)) { printf("run failed at pos %d\n", pos); return 4; }
        awnn::TensorView lg, nk, nv;
        if (!s.output_tensor(0, &lg) || !s.output_tensor(1, &nk) || !s.output_tensor(2, &nv)) { printf("outputs failed\n"); return 5; }
        if (!dump.empty() && pos == dumpat) { std::ofstream f(dump, std::ios::binary); f.write((char*)lg.data, V * 4); f.write((char*)nk.data, L3 * HD * 4); f.write((char*)nv.data, L3 * HD * 4); }
        for (int l = 0; l < L3; ++l) for (int d = 0; d < HD; ++d) {
            size_t o = ((size_t)l * (C - 1) + pos) * HD + d;
            if (flt) { pkf[o] = nk.data[l * HD + d]; pvf[o] = nv.data[l * HD + d]; }
            else if (half) { pk[o] = (int16_t)f2h(nk.data[l * HD + d]); pv[o] = (int16_t)f2h(nv.data[l * HD + d]); }
            else { pk[o] = q16(nk.data[l * HD + d], fl[4]); pv[o] = q16(nv.data[l * HD + d], fl[5]); }
        }
        ++pos;
        if (pos < (int)ids.size()) { cur = ids[pos]; t_pre += ms(ts); ++n_pre; }
        else {
            int best = std::max_element(lg.data, lg.data + V) - lg.data; t_dec += ms(ts); ++n_dec;
            if (eosset.count(best)) break;
            out.push_back(best); cur = best;
        }
    }
    printf("IDS"); for (int t : out) printf(" %d", t); printf("\n");
    printf("prefill %d tok %.1f ms/tok | decode %d tok %.1f ms/tok = %.2f tok/s (incl. CPU copies)\n", n_pre, n_pre ? t_pre / n_pre : 0.0, n_dec, n_dec ? t_dec / n_dec : 0.0, n_dec ? 1000.0 * n_dec / t_dec : 0.0);
    return 0;
}
