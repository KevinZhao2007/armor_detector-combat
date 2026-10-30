#include "state_plot.hpp"
#include <cstdio>
#include <algorithm>
#include <cmath>
using namespace std;
using namespace cv;
namespace {

const char* kName[8] = {"xc",  "vx",   "yc",   "vy",  "za", "vz",   "yaw",  "wyaw"};
const char* kUnit[8] = {"m",   "m/s",  "m",    "m/s", "m",  "m/s",  "rad",  "rad/s"};

inline Point P2I(double px, double py) {
    return Point(static_cast<int>(lround(px)),
                     static_cast<int>(lround(py)));
}

// 画一个子图：把 (t, 估计值, σ) 映射到 cell 里
void drawCell(Mat& img, const Rect& cell,
              const vector<double>& ts,
              const vector<double>& vs,
              const vector<double>& ss,
              double t0, double t1, int idx)
{
    const int padL = 52, padR = 12, padT = 28, padB = 22;
    Rect plot(cell.x + padL, cell.y + padT,
                  cell.width  - padL - padR,
                  cell.height - padT - padB);
    if (plot.width < 20 || plot.height < 20) return;

    // ---- 纵轴范围：把 ±2σ 也包进去（否则阴影会被裁掉）----
    double lo =  1e18, hi = -1e18;
    for (size_t i = 0; i < ts.size(); ++i) {
        if (ts[i] < t0 || ts[i] > t1) continue;
        lo = min(lo, vs[i] - 2.0 * ss[i]);
        hi = max(hi, vs[i] + 2.0 * ss[i]);
    }
    if (!(hi > lo)) { lo = -1.0; hi = 1.0; }      // 没有数据
    if (hi - lo < 1e-9) { hi += 0.5; lo -= 0.5; }
    const double margin = 0.10 * (hi - lo);
    lo -= margin; hi += margin;

    auto mapX = [&](double tt) {
        return plot.x + (tt - t0) / (t1 - t0) * plot.width; };
    auto mapY = [&](double vv) {
        return plot.y + plot.height - (vv - lo) / (hi - lo) * plot.height; };

    // ---- 背景 / 网格 / 刻度 ----
    rectangle(img, plot, Scalar(248, 248, 250), FILLED);
    for (int k = 0; k <= 2; ++k) {
        const double vv = lo + (hi - lo) * k / 2.0;
        const int yy = static_cast<int>(lround(mapY(vv)));
        line(img, Point(plot.x, yy),
                      Point(plot.x + plot.width, yy),
                 Scalar(214, 214, 220), 1);
        char b[32];
        snprintf(b, sizeof(b), "%.3g", vv);
        putText(img, b, Point(cell.x + 3, yy + 5),
                    FONT_HERSHEY_SIMPLEX, 0.34, Scalar(110, 110, 125), 1, LINE_AA);
    }
    rectangle(img, plot, Scalar(178, 178, 190), 1);

    // ---- ±2σ 阴影：先画，压在线下面 ----
    vector<Point> upper, lower;
    for (size_t i = 0; i < ts.size(); ++i) {
        if (ts[i] < t0 || ts[i] > t1) continue;
        upper.push_back(P2I(mapX(ts[i]), mapY(vs[i] + 2.0 * ss[i])));
        lower.push_back(P2I(mapX(ts[i]), mapY(vs[i] - 2.0 * ss[i])));
    }
    if (upper.size() >= 2) {
        vector<Point> poly = upper;
        poly.insert(poly.end(), lower.rbegin(), lower.rend());
        vector<vector<Point>> polys;
        polys.push_back(poly);
        fillPoly(img, polys, Scalar(255, 208, 150), LINE_AA);
    }

    // 估计曲线 
    vector<Point> line;
    for (size_t i = 0; i < ts.size(); ++i) {
        if (ts[i] < t0 || ts[i] > t1) continue;
        line.push_back(P2I(mapX(ts[i]), mapY(vs[i])));
    }
    if (line.size() >= 2) {
        vector<vector<Point>> lines;
        lines.push_back(line);
        polylines(img, lines, false, Scalar(30, 90, 220), 2, LINE_AA);
    }

    // ---- 标题：名字 + 单位 + 当前值 ----
    char title[64];
    snprintf(title, sizeof(title), "%s [%s]", kName[idx], kUnit[idx]);
    putText(img, title, Point(cell.x + 4, cell.y + 18),
                FONT_HERSHEY_SIMPLEX, 0.45, Scalar(40, 40, 50), 1, LINE_AA);

    if (!line.empty()) {
        char val[32];
        snprintf(val, sizeof(val), "%.3f", vs.back());
        putText(img, val, Point(cell.x + cell.width - 78, cell.y + 18),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(30, 90, 220), 1, LINE_AA);
    }
}

}  // namespace

Mat drawStatePanel(const StateHistory& h, int W, int H, double window_s) {
    Mat img(H, W, CV_8UC3, Scalar(252, 252, 254));
    if (h.t.empty()) {
        putText(img, "no data", Point(24, H / 2),
                    FONT_HERSHEY_SIMPLEX, 0.8, Scalar(150, 150, 160), 2);
        return img;
    }

    const double t1 = h.t.back();
    double t0 = (window_s > 0.0) ? max(h.t.front(), t1 - window_s) : h.t.front();
    if (t1 - t0 < 1e-6) t0 = t1 - 1.0;

    const int cw = W / 2, ch = H / 4;

    vector<double> ts(h.t);
    vector<double> vs(h.t.size()), ss(h.t.size());
    for (int idx = 0; idx < 8; ++idx) {
        for (size_t i = 0; i < h.t.size(); ++i) {
            vs[i] = h.x[i][idx];
            ss[i] = h.s[i][idx];
        }
        const int r = idx / 2, c = idx % 2;
        drawCell(img, Rect(c * cw, r * ch, cw, ch), ts, vs, ss, t0, t1, idx);
    }

    char tb[80];
    snprintf(tb, sizeof(tb), "t = %.1f s  (window %.1f s)", t1, t1 - t0);
    putText(img, tb, Point(W - 230, 15),
                FONT_HERSHEY_SIMPLEX, 0.4, Scalar(115, 115, 130), 1, LINE_AA);
    return img;
}
