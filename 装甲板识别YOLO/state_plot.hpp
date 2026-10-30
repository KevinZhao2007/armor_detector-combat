#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <array>
#include <vector>
using namespace std;
using namespace Eigen;
using namespace cv;
struct StateHistory {
    vector<double>               t;
    vector<array<double,8>> x;     // 估计值
    vector<array<double,8>> s;     // 标准差 sqrt(P_ii)
    vector<int>                  state; // 0=LOST 1=DETECTING 2=TRACKING 3=TEMP_LOST

    // 每帧调用一次
    void push(double tt, const VectorXd& xv, const MatrixXd& P, int st) {
        array<double,8> a{}, b{};
        for (int i = 0; i < 8; ++i) {
            a[i] = xv(i);
            b[i] = sqrt(max(0.0, P(i,i)));
        }
        t.push_back(tt); x.push_back(a); s.push_back(b); state.push_back(st);

        // 只保留最近 3000 帧，防止长时间录制内存一直涨
        if (t.size() > 3000) {
            t.erase(t.begin()); x.erase(x.begin());
            s.erase(s.begin()); state.erase(state.begin());
        }
    }
    size_t size() const { return t.size(); }
    void clear() { t.clear(); x.clear(); s.clear(); state.clear(); }
};

// 画面板。window_s > 0 = 只显示最近这么多秒；<= 0 = 显示全部
Mat drawStatePanel(const StateHistory& h, int W = 760, int H = 600,
                       double window_s = 10.0);
