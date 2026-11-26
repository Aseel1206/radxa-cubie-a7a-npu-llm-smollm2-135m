// Persistent chat engine for the SmolLM2 int16 .nb. Protocol on stdin/stdout (line based):
//   RESET                       -> back to the preloaded chat prefix
//   TEMP <t> <topk> <rep>       -> sampling (t=0 greedy)
//   GEN <max_new> <id> <id>...  -> feed these tokens (continuing the conversation), then generate
//   replies: "TOK <id>" per token, "DONE <n_new> <ms_per_tok> <pos> <flag>" (flag 1 = hit eos, 0 = limit, 2 = context full)
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <iostream>
#include "awnn_inference/session.h"
using clk = std::chrono::steady_clock;
static double ms(clk::time_point a) { return std::chrono::duration<double, std::milli>(clk::now() - a).count(); }
int main(int argc, char** argv) {
    std::string model, emb, prefix, fls = "14,15,15,0,10,12"; int C = 128;
    for (int i = 1; i < argc; ++i) { std::string a = argv[i];
        if (a == "-m") model = argv[++i]; else if (a == "-e") emb = argv[++i]; else if (a == "-pre") prefix = argv[++i]; else if (a == "-fl") fls = argv[++i]; else if (a == "-c") C = atoi(argv[++i]); }
    const int DM = 576, HD = 64, L3 = 90, V = 49152, EOS = 2;
    int fl[6]; { std::stringstream ss(fls); std::string t; int k = 0; while (std::getline(ss, t, ',') && k < 6) fl[k++] = atoi(t.c_str()); }
    std::vector<float> E((size_t)V * DM); { std::ifstream f(emb, std::ios::binary); f.read((char*)E.data(), E.size() * 4); if (!f) { printf("ERR embed\n"); return 1; } }
    awnn::Session s; if (!s.load(model)) { printf("ERR load\n"); return 2; }
    const int KVN = L3 * (C - 1) * HD;
    std::vector<int16_t> pk(KVN, 0), pv(KVN, 0);
    auto q16 = [&](float x, int f) { float r = roundf(x * (float)(1 << f)); if (r > 32767) r = 32767; if (r < -32768) r = -32768; return (int16_t)r; };
    auto f2h = [](float f) -> uint16_t { union { float f; uint32_t u; } x{f}; uint32_t sg = (x.u >> 16) & 0x8000; uint32_t a = x.u & 0x7fffffff;
        if (a >= 0x47800000) return sg | 0x7bff; if (a < 0x33000000) return sg;
        if (a < 0x38800000) { uint32_t m = (a & 0x7fffff) | 0x800000; int shift = 125 - (int)(a >> 23) + 14; uint32_t v = m >> shift; uint32_t rem = m & ((1u << shift) - 1), h = 1u << (shift - 1); if (rem > h || (rem == h && (v & 1))) ++v; return sg | v; }
        uint32_t v = ((a - 0x38000000) >> 13); uint32_t rem = a & 0x1fff; if (rem > 0x1000 || (rem == 0x1000 && (v & 1))) ++v; return sg | v; };
    auto put = [&](int idx, const std::vector<float>& v) {
        if (idx == 3) { std::vector<uint16_t> h(v.size()); for (size_t i = 0; i < v.size(); ++i) h[i] = f2h(v[i]); return s.copy_input(idx, h.data(), h.size() * 2); }
        std::vector<int16_t> q(v.size()); for (size_t i = 0; i < v.size(); ++i) q[i] = q16(v[i], fl[idx]); return s.copy_input(idx, q.data(), q.size() * 2); };
    // preload prefix KV
    int P = 0; std::vector<int16_t> pk0, pv0;
    { std::ifstream f(prefix, std::ios::binary); int32_t n; f.read((char*)&n, 4); P = n; std::vector<float> a((size_t)L3 * P * HD), b(a.size());
      f.read((char*)a.data(), a.size() * 4); f.read((char*)b.data(), b.size() * 4); if (!f) { printf("ERR prefix\n"); return 1; }
      for (int l = 0; l < L3; ++l) for (int t = 0; t < P; ++t) for (int d = 0; d < HD; ++d) { size_t o = ((size_t)l * (C - 1) + t) * HD + d, i = ((size_t)l * P + t) * HD + d; pk[o] = q16(a[i], fl[4]); pv[o] = q16(b[i], fl[5]); } }
    int pos = P; float temp = 0, rep = 1.1f; int topk = 40; std::mt19937 rng(1234);
    std::vector<float> x(DM), cs(HD), sn(HD), mask(C);
    auto step = [&](int tok, awnn::TensorView* lg) -> bool {
        memcpy(x.data(), &E[(size_t)tok * DM], DM * 4);
        for (int j = 0; j < HD / 2; ++j) { double f = pos * pow(100000.0, -2.0 * j / HD); cs[j] = cs[j + 32] = cos(f); sn[j] = sn[j + 32] = sin(f); }
        for (int j = 0; j < C; ++j) mask[j] = (j < pos || j == C - 1) ? 0.f : -30.f;
        if (!(put(0, x) && put(1, cs) && put(2, sn) && put(3, mask) && s.copy_input(4, pk.data(), KVN * 2) && s.copy_input(5, pv.data(), KVN * 2))) return false;
        if (!s.run(nullptr)) return false;
        awnn::TensorView nk, nv; if (!s.output_tensor(0, lg) || !s.output_tensor(1, &nk) || !s.output_tensor(2, &nv)) return false;
        for (int l = 0; l < L3; ++l) for (int d = 0; d < HD; ++d) { size_t o = ((size_t)l * (C - 1) + pos) * HD + d; pk[o] = q16(nk.data[l * HD + d], fl[4]); pv[o] = q16(nv.data[l * HD + d], fl[5]); }
        ++pos; return true; };
    printf("READY %d %d\n", P, C - 1); fflush(stdout);
    std::string line;
    while (std::getline(std::cin, line)) {
        std::stringstream ss(line); std::string cmd; ss >> cmd;
        if (cmd == "RESET") { pos = P; for (int l = 0; l < L3; ++l) { size_t b = ((size_t)l * (C - 1) + P) * HD, e = ((size_t)l * (C - 1) + (C - 1)) * HD; std::fill(pk.begin() + b, pk.begin() + e, 0); std::fill(pv.begin() + b, pv.begin() + e, 0); } printf("OK %d\n", pos); fflush(stdout); }
        else if (cmd == "TEMP") { ss >> temp >> topk >> rep; printf("OK\n"); fflush(stdout); }
        else if (cmd == "QUIT") break;
        else if (cmd == "GEN") {
            int maxnew; ss >> maxnew; std::vector<int> in; int v; while (ss >> v) in.push_back(v);
            awnn::TensorView lg; bool ok = true; int flag = 0;
            if (pos + (int)in.size() >= C - 2) { printf("DONE 0 0 %d 2\n", pos); fflush(stdout); continue; }
            auto t0 = clk::now(); for (size_t i = 0; i < in.size() && ok; ++i) ok = step(in[i], &lg);
            double tpre = ms(t0); std::vector<int> gen; auto t1 = clk::now();
            while (ok && (int)gen.size() < maxnew) {
                if (pos >= C - 2) { flag = 2; break; }
                std::vector<float> l(lg.data, lg.data + V);
                for (int g : gen) l[g] = l[g] > 0 ? l[g] / rep : l[g] * rep;
                int best;
                if (temp <= 0) best = std::max_element(l.begin(), l.end()) - l.begin();
                else { std::vector<int> idx(V); for (int i = 0; i < V; ++i) idx[i] = i; int k = std::min(topk, V);
                    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return l[a] > l[b]; });
                    double m = l[idx[0]], sum = 0; std::vector<double> p(k); for (int i = 0; i < k; ++i) { p[i] = exp((l[idx[i]] - m) / temp); sum += p[i]; }
                    double r = std::uniform_real_distribution<double>(0, sum)(rng), acc = 0; best = idx[k - 1]; for (int i = 0; i < k; ++i) { acc += p[i]; if (r <= acc) { best = idx[i]; break; } } }
                if (best == EOS) { flag = 1; break; }
                gen.push_back(best); printf("TOK %d\n", best); fflush(stdout);
                ok = step(best, &lg);
            }
            double tg = ms(t1); if (!ok) { printf("ERR run\n"); fflush(stdout); return 4; }
            printf("DONE %zu %.1f %d %d %.1f\n", gen.size(), gen.empty() ? 0.0 : tg / gen.size(), pos, flag, tpre); fflush(stdout);
        }
    }
    return 0;
}
