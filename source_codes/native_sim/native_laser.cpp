// 冻结的OSD固定步离散等价规则，C++加速版。没有改用正确时间RK4。
// 逐路参数、100步初态二分、插值区间外推、步末输出均保留。
#include <cmath>
#include <cstdint>
#include <algorithm>

struct State { double n,s,phase; };
static State add(State x, State d, double scale) {
    return {x.n+scale*d.n,x.s+scale*d.s,x.phase+scale*d.phase};
}

extern "C" int native_laser(const double* input, const double* parameters,
    int routes, int64_t count, double fs, double* fields, double* current_bounds) {
    if(routes<1 || count<4 || !(fs>0)) return -1;
    constexpr double q=1.602176634e-19, h=6.62607015e-34;
    const double integration_ratio=double(count-1)/double(count);
    const double selector_ratio=double(count-2)/double(count+1);
    const double step_ns=1e9/fs*integration_ratio;
    for(int route=0;route<routes;++route) {
        const double* p=parameters+route*14;
        const double v=p[0],eta=p[1],beta=p[2],eps=p[3],nt=p[4];
        const double g=p[5]*p[6],gamma=p[7],tn=p[8],tp=p[9],alpha=p[10];
        const double bias=p[11],peak=p[12],frequency=p[13];
        const double* x=input+route*count;
        const double initial_current=bias+peak*x[0];
        if(!(initial_current>0)) return 1000+route;
        const double factor=v*eta*h*frequency/(2*gamma*tp);
        double lo=0,hi=1e18,photons=0,carriers=0;
        for(int i=0;i<100;++i) {
            photons=(lo+hi)/2;
            double a=g*photons/(1+eps*photons);
            carriers=(a*nt+photons/(gamma*tp))/(a+beta/tn);
            double current=q*v*(carriers/tn+g*(carriers-nt)*photons/(1+eps*photons));
            if(current>initial_current) hi=photons; else lo=photons;
        }
        State y={carriers/1e18,photons/1e15,0};
        auto rhs=[&](State s,double current)->State {
            const double carrier=s.n*1e18, photon=s.s*1e15;
            const double stimulated=g*(carrier-nt)*photon/(1+eps*photon);
            return {(current/(q*v)-carrier/tn-stimulated)*1e-27,
                (gamma*stimulated-photon/tp+gamma*beta*carrier/tn)*1e-24,
                alpha/2*(gamma*g*(carrier-nt)-1/tp)*1e-9};
        };
        double minimum=initial_current,maximum=initial_current;
        for(int64_t k=0;k<count;++k) {
            double current[3];
            for(int j=0;j<3;++j) {
                double stage=double(k)+double(j)*.5;
                int64_t left=int64_t(std::floor(stage*selector_ratio));
                if(left<0 || left+1>=count) return 2000+route;
                double a=stage*integration_ratio-double(left);
                current[j]=bias+peak*((1-a)*x[left]+a*x[left+1]);
                minimum=std::min(minimum,current[j]); maximum=std::max(maximum,current[j]);
                if(!(current[j]>=0) || !std::isfinite(current[j])) return 3000+route;
            }
            State k1=rhs(y,current[0]);
            State k2=rhs(add(y,k1,step_ns/2),current[1]);
            State k3=rhs(add(y,k2,step_ns/2),current[1]);
            State k4=rhs(add(y,k3,step_ns),current[2]);
            y.n+=step_ns*(k1.n+2*k2.n+2*k3.n+k4.n)/6;
            y.s+=step_ns*(k1.s+2*k2.s+2*k3.s+k4.s)/6;
            y.phase+=step_ns*(k1.phase+2*k2.phase+2*k3.phase+k4.phase)/6;
            if(!(y.n>0) || !(y.s>0) || !std::isfinite(y.phase+y.n+y.s)) return 4000+route;
            double amplitude=std::sqrt((y.s*1e15)*factor);
            int64_t index=2*(route*count+k);
            fields[index]=amplitude*std::cos(y.phase);
            fields[index+1]=amplitude*std::sin(y.phase);
        }
        current_bounds[2*route]=minimum;current_bounds[2*route+1]=maximum;
    }
    return 0;
}
