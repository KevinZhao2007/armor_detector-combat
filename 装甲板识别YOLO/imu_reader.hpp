#pragma once
#include <array>
#include <vector>
#include <string>
using namespace std;
struct ImuLog
{
    struct S
    {
        double t;
        array<double, 4> q; 
    };
    vector<S> s;
    bool load(const string& path);
    array<double, 4> at(double tt) const;
    static array<array<double,3>,3> toRot(const array<double,4>& q);
};