#include "imu_reader.hpp"
#include <fstream>
#include <cmath>
#include <algorithm>
using namespace std;
bool ImuLog::load(const string& path)
{
    ifstream in(path);
    if(!in)return false;
    s.clear();
    double t,w,x,y,z;
    while(in>>t>>w>>x>>y>>z)
    s.push_back({t,{w,x,y,z}});
    return s.size()>=2;
}
array<double,4> ImuLog::at(double tt) const
{
    if(tt<=s.front().t) return s.front().q;
    if(tt>=s.back().t) return s.back().q;
    auto it=upper_bound(s.begin(),s.end(),tt,[](double v,const S& a){return v<a.t;});
    const S& b= *it;
    const S& a= *(it-1);
    const double u=(tt-a.t)/(b.t-a.t);
    array<double,4> qb=b.q;
    if(a.q[0]*qb[0]+a.q[1]*qb[1]+a.q[2]*qb[2]+a.q[3]*qb[3]<0)
    for (auto& v : qb) v = -v;
    array<double,4> q;
    for(int i=0;i<4;i++)
    q[i]=a.q[i]+u*(qb[i]-a.q[i]);
    const double n=sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    for (auto& v : q) v /= n;
    return q;
}
array<array<double,3>,3> ImuLog::toRot(const array<double,4>& q)
{
    const double w=q[0],x=q[1],y=q[2],z=q[3];
    return {{
        {1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)},
        {2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)},
        {2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)}
    }};
}