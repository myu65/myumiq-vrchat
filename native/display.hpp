#pragma once
#include <openvr_driver.h>

namespace myumiq {
class Display final : public vr::IVRDisplayComponent {
public:
    uint32_t size = 512;
    void GetWindowBounds(int32_t* x, int32_t* y, uint32_t* w, uint32_t* h) override {
        *x = 0; *y = 0; *w = size * 2; *h = size;
    }
    bool IsDisplayOnDesktop() override { return true; }
    bool IsDisplayRealDisplay() override { return false; }
    void GetRecommendedRenderTargetSize(uint32_t* w, uint32_t* h) override { *w = *h = size; }
    void GetEyeOutputViewport(vr::EVREye eye, uint32_t* x, uint32_t* y, uint32_t* w, uint32_t* h) override {
        *x = eye == vr::Eye_Left ? 0 : size; *y = 0; *w = *h = size;
    }
    void GetProjectionRaw(vr::EVREye, float* l, float* r, float* t, float* b) override {
        *l = *t = -1; *r = *b = 1;
    }
    vr::DistortionCoordinates_t ComputeDistortion(vr::EVREye, float u, float v) override {
        return {{u,v}, {u,v}, {u,v}};
    }
    bool ComputeInverseDistortion(vr::HmdVector2_t* out, vr::EVREye, uint32_t, float u, float v) override {
        out->v[0] = u; out->v[1] = v; return true;
    }
};
}
