#include "mex.h"
#include <cmath>
#include <complex>
#include <vector>
#include <string>
#include <algorithm>

using cd = std::complex<double>;
static const cd I(0.0, 1.0);
static constexpr double PI = 3.141592653589793238462643383279502884;

static double getScalar(const mxArray* a, const char* name) {
    if (!mxIsDouble(a) || mxIsComplex(a) || mxGetNumberOfElements(a) != 1) {
        mexErrMsgIdAndTxt("SSHFT_mex:input", "%s 必须是 double 标量。", name);
    }
    return mxGetPr(a)[0];
}

static std::vector<double> getRealVector(const mxArray* a, const char* name, int expectedLen) {
    if (!mxIsDouble(a) || mxIsComplex(a)) {
        mexErrMsgIdAndTxt("SSHFT_mex:input", "%s 必须是 double 实数向量。", name);
    }
    if ((int)mxGetNumberOfElements(a) != expectedLen) {
        mexErrMsgIdAndTxt("SSHFT_mex:input", "%s 长度必须等于 %d。", name, expectedLen);
    }
    const double* pr = mxGetPr(a);
    return std::vector<double>(pr, pr + expectedLen);
}

static std::vector<cd> getComplexVector(const mxArray* a, const char* name, int expectedLen) {
    if (!mxIsDouble(a)) {
        mexErrMsgIdAndTxt("SSHFT_mex:input", "%s 必须是 double 向量。", name);
    }
    if ((int)mxGetNumberOfElements(a) != expectedLen) {
        mexErrMsgIdAndTxt("SSHFT_mex:input", "%s 长度必须等于 %d。", name, expectedLen);
    }
    const double* pr = mxGetPr(a);
    const double* pi = mxGetPi(a);
    std::vector<cd> v(expectedLen);
    for (int k = 0; k < expectedLen; ++k) {
        v[k] = cd(pr[k], pi ? pi[k] : 0.0);
    }
    return v;
}

static std::vector<int> getZDownIndices(const mxArray* a, const char* name, int Nz) {
    const int len = (int)mxGetNumberOfElements(a);
    if (len <= 0) {
        mexErrMsgIdAndTxt("SSHFT_mex:input", "%s 不能为空。", name);
    }

    std::vector<int> idx;

    // 情况1：MATLAB 逻辑掩码，长度必须等于 Nz。
    if (mxIsLogical(a)) {
        if (len != Nz) {
            mexErrMsgIdAndTxt("SSHFT_mex:input", "%s 如果是 logical，长度必须等于 Nz=%d。", name, Nz);
        }
        const mxLogical* p = mxGetLogicals(a);
        for (int k = 0; k < Nz; ++k) {
            if (p[k] != 0) idx.push_back(k); // 0-based
        }
        return idx;
    }

    // 情况2：MATLAB 下标向量。你的主函数传入的就是 1:step:Nz 这种 1-based 下标列表。
    if (mxIsDouble(a) && !mxIsComplex(a)) {
        const double* p = mxGetPr(a);

        // 兼容长度为 Nz 的 double 0/1 掩码。
        bool maybeMask = (len == Nz);
        if (maybeMask) {
            for (int k = 0; k < len; ++k) {
                if (!(p[k] == 0.0 || p[k] == 1.0)) {
                    maybeMask = false;
                    break;
                }
            }
        }
        if (maybeMask) {
            for (int k = 0; k < Nz; ++k) {
                if (p[k] != 0.0) idx.push_back(k);
            }
            return idx;
        }

        // 默认按 MATLAB 1-based 下标列表处理。
        idx.reserve(len);
        for (int k = 0; k < len; ++k) {
            double val = p[k];
            int q = (int)std::llround(val);
            if (std::abs(val - q) > 1e-9 || q < 1 || q > Nz) {
                mexErrMsgIdAndTxt("SSHFT_mex:input",
                    "%s 必须是 1 到 Nz=%d 之间的整数下标，或长度为 Nz 的 logical 掩码。", name, Nz);
            }
            idx.push_back(q - 1); // MATLAB 1-based -> C++ 0-based
        }
        return idx;
    }

    mexErrMsgIdAndTxt("SSHFT_mex:input", "%s 必须是 logical 掩码或 double 下标向量。", name);
    return idx;
}

// 等价于 MATLAB:
// pad = zeros(2*(Nz+1),1); pad(2:Nz+1)=src_part; F=fft(pad); out=F(2:Nz+1)
static std::vector<cd> fftValidFromRealPart(const std::vector<cd>& src, bool useImagPart, int Nz) {
    const int N2_fft = 2 * (Nz + 1);
    mxArray* in = mxCreateDoubleMatrix(N2_fft, 1, mxREAL);
    double* prIn = mxGetPr(in);
    for (int j = 0; j < Nz; ++j) {
        prIn[j + 1] = useImagPart ? src[j].imag() : src[j].real();
    }

    mxArray* outMx = nullptr;
    if (mexCallMATLAB(1, &outMx, 1, &in, "fft") != 0 || outMx == nullptr) {
        mxDestroyArray(in);
        mexErrMsgIdAndTxt("SSHFT_mex:fft", "调用 MATLAB fft 失败。可恶，FFT 不背这个锅。 ");
    }

    const double* pr = mxGetPr(outMx);
    const double* pi = mxGetPi(outMx);
    std::vector<cd> out(Nz);
    for (int j = 0; j < Nz; ++j) {
        out[j] = cd(pr[j + 1], pi ? pi[j + 1] : 0.0);
    }

    mxDestroyArray(in);
    mxDestroyArray(outMx);
    return out;
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs != 16) {
        mexErrMsgIdAndTxt("SSHFT_mex:nrhs",
            "需要 16 个输入: u,u_curr,eps_r,sigma,lambda,n,k0,w,z0,x,dx,Nz,Nx,dz,pz,z_down");
    }
    if (nlhs > 1) {
        mexErrMsgIdAndTxt("SSHFT_mex:nlhs", "输出只能有一个 u。 ");
    }

    const double eps_r  = getScalar(prhs[2],  "eps_r");
    const double sigma  = getScalar(prhs[3],  "sigma");
    const double lambda = getScalar(prhs[4],  "lambda");
    const double k0     = getScalar(prhs[6],  "k0");
    const double z0     = getScalar(prhs[8],  "z0");
    const double dx     = getScalar(prhs[10], "dx");
    const int Nz        = (int)std::llround(getScalar(prhs[11], "Nz"));
    const int Nx        = (int)std::llround(getScalar(prhs[12], "Nx"));
    const double dz     = getScalar(prhs[13], "dz");

    if (Nz <= 0 || Nx <= 1 || dz <= 0.0) {
        mexErrMsgIdAndTxt("SSHFT_mex:input", "Nz、Nx、dz 参数非法。 ");
    }

    std::vector<cd> u_curr = getComplexVector(prhs[1], "u_curr", Nz);
    std::vector<double> n  = getRealVector(prhs[5],  "n",  Nz);
    std::vector<double> w  = getRealVector(prhs[7],  "w",  Nz);
    std::vector<double> x  = getRealVector(prhs[9],  "x",  Nx);
    std::vector<double> pz = getRealVector(prhs[14], "pz", Nz);
    std::vector<int> z_down_idx = getZDownIndices(prhs[15], "z_down", Nz);
    const int Nz_down = (int)z_down_idx.size();
    if (Nz_down <= 0) {
        mexErrMsgIdAndTxt("SSHFT_mex:input", "z_down 中没有有效下标。 ");
    }

    // 创建输出 u，并尽量复制输入 u 的已有内容。
    plhs[0] = mxCreateDoubleMatrix(Nz_down, Nx, mxCOMPLEX);
    double* uRe = mxGetPr(plhs[0]);
    double* uIm = mxGetPi(plhs[0]);
    std::fill(uRe, uRe + Nz_down * Nx, 0.0);
    std::fill(uIm, uIm + Nz_down * Nx, 0.0);

    if (mxIsDouble(prhs[0]) && mxGetM(prhs[0]) == (mwSize)Nz_down && mxGetN(prhs[0]) == (mwSize)Nx) {
        const double* inRe = mxGetPr(prhs[0]);
        const double* inIm = mxGetPi(prhs[0]);
        const mwSize total = (mwSize)Nz_down * (mwSize)Nx;
        for (mwSize k = 0; k < total; ++k) {
            uRe[k] = inRe[k];
            uIm[k] = inIm ? inIm[k] : 0.0;
        }
    }

    mexPrintf("正在求解SSFT...\n");

    // 计算阻抗系数
    cd eps_r_earth(eps_r, 60.0 * sigma * lambda);
    std::vector<cd> beta_imp(Nx - 1);
    for (int ix = 0; ix < Nx - 1; ++ix) {
        double alpha = std::atan(z0 / x[ix + 1]); // MATLAB: x(2:end)
        double s = std::sin(alpha);
        beta_imp[ix] = I * k0 * std::sqrt(eps_r_earth - cd(s * s, 0.0));
    }

    const double dp = PI / ((Nz + 1.0) * dz);
    std::vector<cd> prop_p(Nz), medium_phase(Nz), mpw(Nz);
    std::vector<double> pz_sq(Nz);
    for (int j = 0; j < Nz; ++j) {
        pz_sq[j] = pz[j] * pz[j];
        prop_p[j] = std::exp(I * dx * (std::sqrt(cd(k0 * k0 - pz_sq[j], 0.0)) - cd(k0, 0.0)));
        medium_phase[j] = std::exp(I * k0 * dx * (n[j] - 1.0));
        mpw[j] = (2.0 / PI) * medium_phase[j] * w[j];
    }

    for (int ix = 0; ix < Nx - 1; ++ix) {
        cd bj = beta_imp[ix];

        std::vector<cd> Fre = fftValidFromRealPart(u_curr, false, Nz);
        std::vector<cd> Fie = fftValidFromRealPart(u_curr, true,  Nz);

        std::vector<cd> Us(Nz), Uc(Nz), U(Nz), pU(Nz), T1(Nz), T2(Nz);
        for (int j = 0; j < Nz; ++j) {
            Us[j] = dz * cd(-Fre[j].imag(), -Fie[j].imag());
            Uc[j] = dz * cd( Fre[j].real(),  Fie[j].real());
            U[j]  = bj * Us[j] - pz[j] * Uc[j];
            cd den = bj * bj + pz_sq[j];
            pU[j] = (prop_p[j] / den) * U[j];
            T1[j] = bj * pU[j];
            T2[j] = pz[j] * pU[j];
        }

        std::vector<cd> Fr1e = fftValidFromRealPart(T1, false, Nz);
        std::vector<cd> Fi1e = fftValidFromRealPart(T1, true,  Nz);
        std::vector<cd> Fr2e = fftValidFromRealPart(T2, false, Nz);
        std::vector<cd> Fi2e = fftValidFromRealPart(T2, true,  Nz);

        std::vector<cd> next_u(Nz);
        for (int j = 0; j < Nz; ++j) {
            cd ss = dp * cd(-Fr1e[j].imag(), -Fi1e[j].imag());
            cd sc = dp * cd( Fr2e[j].real(),  Fi2e[j].real());
            next_u[j] = mpw[j] * (ss - sc);
        }

        for (int row = 0; row < Nz_down; ++row) {
            const int j = z_down_idx[row];
            const mwSize idx = row + (mwSize)ix * Nz_down;
            uRe[idx] = u_curr[j].real();
            uIm[idx] = u_curr[j].imag();
        }

        u_curr.swap(next_u);

        if (((ix + 1) % 200 == 0) || ix == Nx - 2) {
            double pct = double(ix + 1) / double(Nx - 1);
            mexPrintf("\r%5.1f%% | %d/%d", pct * 100.0, ix + 1, Nx - 1);
            mexEvalString("drawnow;");
        }
    }

    for (int row = 0; row < Nz_down; ++row) {
        const int j = z_down_idx[row];
        const mwSize idx = row + (mwSize)(Nx - 1) * Nz_down;
        uRe[idx] = u_curr[j].real();
        uIm[idx] = u_curr[j].imag();
    }
    mexPrintf("\n");
}
