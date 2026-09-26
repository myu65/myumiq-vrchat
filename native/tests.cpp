#include "protocol.hpp"
#include "display.hpp"
#include "driver.cpp"
#include <iostream>
#include <stdexcept>
#include <thread>

void check(bool value) { if (!value) throw std::runtime_error("contract failed"); }
int main() {
    using namespace myumiq;
    Target t;
    check(parse("v1 1 2 3 1 0 0 0 0.5 1", t));
    check(to_openvr(t.position) == std::array<double, 3>{-2, 3, -1});
    check(to_openvr({1,0,0}) == std::array<double, 3>{0,0,-1});
    check(to_openvr({0,1,0}) == std::array<double, 3>{-1,0,0});
    check(to_openvr({0,0,1}) == std::array<double, 3>{0,1,0});
    const char* bad[] = {"", "v2 0 0 0 1 0 0 0 0 0", "v1 0 0", "v1 nan 0 0 1 0 0 0 0 0",
        "v1 0 0 0 0 0 0 0 0 0", "v1 0 0 0 1 0 0 0 2 0", "v1 11 0 0 1 0 0 0 0 0",
        "v1 0 0 0 1 0 0 0 0 0 extra", "v1 0 0 0 1 0 0 0 0 -1"};
    for (auto request : bad) { check(!parse(request, t)); check(t.position[0] == 1); }
    check(!parse(std::string(513, 'x'), t));
    check(parse("v1 0 0 0 0.70710678 0 0 0.70710678 0 0", t));
    auto q = to_openvr({t.rotation[1], t.rotation[2], t.rotation[3]});
    check(std::abs(q[1] - std::sqrt(0.5)) < 1e-8);
    Display display;
    uint32_t x,y,w,h;
    display.size = 320;
    display.GetRecommendedRenderTargetSize(&w,&h); check(w == 320 && h == 320);
    display.GetEyeOutputViewport(vr::Eye_Right,&x,&y,&w,&h); check(x == 320 && y == 0 && w == 320 && h == 320);
    auto uv = display.ComputeDistortion(vr::Eye_Left,0.25f,0.75f);
    check(uv.rfRed[0] == 0.25f && uv.rfBlue[1] == 0.75f);
    Device head(0);
    char response[128]{};
    head.DebugRequest("v1 1 2 3 1 0 0 0 0 0", response, sizeof(response));
    check(std::string(response) == "ok");
    auto pose = head.GetPose();
    check(pose.shouldApplyHeadModel);
    check(pose.poseIsValid && pose.vecPosition[0] == -2 && pose.vecPosition[1] == 3 && pose.vecPosition[2] == -1);
    head.DebugRequest("v1 0 0 0 1 0 0 0 1 0", response, sizeof(response));
    check(std::string(response).find("error") == 0 && head.GetPose().vecPosition[0] == -2);
    std::this_thread::sleep_for(std::chrono::milliseconds(550));
    check(!head.GetPose().poseIsValid && head.GetPose().deviceIsConnected);
    head.Deactivate();
    head.RunFrame(); // Inactive device must not call a runtime host.
    int error = -1;
    check(HmdDriverFactory(vr::IServerTrackedDeviceProvider_Version, &error) != nullptr && error == 0);
    check(HmdDriverFactory("invalid", &error) == nullptr && error == vr::VRInitError_Init_InterfaceNotFound);
    std::cout << "Protocol rejection, transactional parsing, coordinate basis/quaternion and display contracts passed\n";
}
