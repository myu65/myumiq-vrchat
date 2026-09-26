#include <openvr.h>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <locale>
#include <sstream>
#include <thread>
#include <algorithm>
#include <charconv>
#include <limits>
#include <fstream>
#include "protocol.hpp"

int main(int argc, char** argv) {
    const bool wave = argc >= 2 && std::strcmp(argv[1], "--wave") == 0;
    const bool fixed = argc >= 3 && std::strcmp(argv[1], "--targets") == 0;
    int seconds = 10;
    if ((wave && argc == 3) || (fixed && argc == 4)) {
        const auto duration = argv[fixed ? 3 : 2];
        const auto end = duration + std::strlen(duration);
        const auto result = std::from_chars(duration, end, seconds);
        if (result.ec != std::errc{} || result.ptr != end || seconds < 5 || seconds > 600) {
            std::cerr << "Duration must be an integer from 5 to 600 seconds\n"; return 2;
        }
    }
    if (argc < 2 || argc > (fixed ? 4 : 3) || (!wave && !fixed && (argc != 2 || std::strcmp(argv[1], "--list") != 0))) {
        std::cerr << "Usage: pose_probe --list | --wave [seconds] | --targets FILE [seconds]\n"; return 2;
    }
    std::array<std::string,3> requests;
    std::array<myumiq::Target,3> targets;
    if (fixed) {
        std::ifstream file(argv[2]);
        for (int j=0;j<3;++j) if(!std::getline(file,requests[j]) || !myumiq::parse(requests[j],targets[j])) {
            std::cerr << "Target file requires exactly three valid v1 lines (head, left, right)\n"; return 2;
        }
        std::string extra;
        if (std::getline(file,extra) || targets[0].trigger!=0 || targets[0].grip!=0) {
            std::cerr << "Unexpected target data or unsupported HMD input\n"; return 2;
        }
    }
    vr::EVRInitError error;
    auto system = vr::VR_Init(&error, vr::VRApplication_Background);
    if (error != vr::VRInitError_None) { std::cerr << vr::VR_GetVRInitErrorAsEnglishDescription(error) << '\n'; return 1; }
    struct Shutdown { ~Shutdown() { vr::VR_Shutdown(); } } shutdown;
    std::array<uint32_t, 3> indices;
    indices.fill(vr::k_unTrackedDeviceIndexInvalid);
    const char* serials[] = {"myumiq-hmd", "myumiq-left", "myumiq-right"};
    for (uint32_t i = 0; i < vr::k_unMaxTrackedDeviceCount; ++i) {
        char serial[128]{};
        vr::ETrackedPropertyError property_error;
        system->GetStringTrackedDeviceProperty(i, vr::Prop_SerialNumber_String, serial, sizeof(serial), &property_error);
        if (property_error != vr::TrackedProp_Success) continue;
        for (int j = 0; j < 3; ++j) if (std::strcmp(serial, serials[j]) == 0 && system->IsTrackedDeviceConnected(i)) {
            if (indices[j] != vr::k_unTrackedDeviceIndexInvalid) { std::cerr << "Duplicate MyuMIQ serial\n"; return 1; }
            indices[j] = i;
            std::cout << serial << " index=" << i << " class=" << system->GetTrackedDeviceClass(i) << '\n';
        }
    }
    for (auto index : indices) if (index == vr::k_unTrackedDeviceIndexInvalid) { std::cerr << "Three connected MyuMIQ devices are required\n"; return 1; }
    uint32_t width = 0, height = 0;
    system->GetRecommendedRenderTargetSize(&width, &height);
    std::cout << "recommended_eye=" << width << 'x' << height << " nominal_hz="
              << system->GetFloatTrackedDeviceProperty(indices[0], vr::Prop_DisplayFrequency_Float) << '\n';
    if (!wave && !fixed) {
        float sinceVsync=0; uint64_t vsyncFrame=0;
        const bool vsyncValid=system->GetTimeSinceLastVsync(&sinceVsync,&vsyncFrame);
        std::cout << "vsync_valid=" << vsyncValid << " seconds_since_vsync=" << sinceVsync
                  << " vsync_frame=" << vsyncFrame << '\n';
        for(auto origin : {vr::TrackingUniverseRawAndUncalibrated,vr::TrackingUniverseStanding}) {
            std::array<vr::TrackedDevicePose_t,vr::k_unMaxTrackedDeviceCount> poses{};
            system->GetDeviceToAbsoluteTrackingPose(origin,0,poses.data(),static_cast<uint32_t>(poses.size()));
            for(int j=0;j<3;++j) {
                const auto& p=poses[indices[j]];
                std::cout << "origin=" << static_cast<int>(origin) << " device=" << serials[j]
                          << " valid=" << p.bPoseIsValid << " result=" << p.eTrackingResult << " xyz="
                          << p.mDeviceToAbsoluteTracking.m[0][3] << ',' << p.mDeviceToAbsoluteTracking.m[1][3]
                          << ',' << p.mDeviceToAbsoluteTracking.m[2][3] << '\n';
            }
        }
        return 0;
    }
    auto debug = vr::VRDebug();
    if (!debug) { std::cerr << "IVRDebug unavailable\n"; return 1; }
    auto deadline = std::chrono::steady_clock::now();
    const auto start = deadline;
    std::array<int,3> valid{};
    std::array<int,3> positionMatches{};
    double minHand = std::numeric_limits<double>::infinity(), maxHand = -minHand;
    double minYaw = minHand, maxYaw = maxHand;
    for (int frame = 0; frame < seconds * 60; ++frame) {
        const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        for (int j = 0; j < 3; ++j) {
            std::ostringstream request;
            request.imbue(std::locale::classic());
            const double yaw = j == 0 ? 0.15 * std::sin(t) : 0;
            request << "v1 " << (j ? 0.35 : 0) << ' ' << (j == 1 ? 0.25 : j == 2 ? -0.25 : 0)
                    << ' ' << (j ? 1.2 + (j == 2 ? 0.15 * std::sin(t * 2) : 0) : 1.6)
                    << ' ' << std::cos(yaw / 2) << " 0 0 " << std::sin(yaw / 2) << " 0 0";
            char response[128]{};
            const auto payload = fixed ? requests[j] : request.str();
            debug->DriverDebugRequest(indices[j], payload.c_str(), response, sizeof(response));
            if (std::strcmp(response, "ok") != 0) { std::cerr << serials[j] << ": " << response << '\n'; return 1; }
        }
        deadline += std::chrono::microseconds(16667);
        std::this_thread::sleep_until(deadline);
        // Raw tracking space avoids room setup/recenter offsets. This is runtime
        // feedback, not an observation of the avatar or an acknowledgement echo.
        std::array<vr::TrackedDevicePose_t, vr::k_unMaxTrackedDeviceCount> poses{};
        system->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseRawAndUncalibrated, 0, poses.data(), static_cast<uint32_t>(poses.size()));
        for (int j = 0; j < 3; ++j) if (poses[indices[j]].bPoseIsValid && poses[indices[j]].bDeviceIsConnected) ++valid[j];
        if(fixed) for(int j=0;j<3;++j) {
            auto expected=myumiq::to_openvr(targets[j].position);
            double squaredError=0;
            for(int axis=0;axis<3;++axis) {
                double delta=poses[indices[j]].mDeviceToAbsoluteTracking.m[axis][3]-expected[axis];
                squaredError+=delta*delta;
            }
            if(poses[indices[j]].bPoseIsValid && squaredError<0.000025) ++positionMatches[j];
        }
        if (poses[indices[2]].bPoseIsValid) {
            const double y = poses[indices[2]].mDeviceToAbsoluteTracking.m[1][3];
            minHand = std::min(minHand,y); maxHand = std::max(maxHand,y);
        }
        if (poses[indices[0]].bPoseIsValid) {
            const double yaw = poses[indices[0]].mDeviceToAbsoluteTracking.m[0][2];
            minYaw = std::min(minYaw,yaw); maxYaw = std::max(maxYaw,yaw);
        }
    }
    std::cout << "frames=" << seconds * 60 << " elapsed_s="
              << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()
              << " valid_samples=" << valid[0] << ',' << valid[1] << ',' << valid[2]
              << " right_hand_y_range=" << minHand << ',' << maxHand
              << " head_rotation_m02_range=" << minYaw << ',' << maxYaw << '\n';
    if(fixed) std::cout << "position_matches=" << positionMatches[0] << ',' << positionMatches[1] << ',' << positionMatches[2] << '\n';
    if (*std::min_element(valid.begin(),valid.end()) < seconds * 54
        || (wave && (maxHand-minHand < 0.2 || maxYaw-minYaw < 0.1))
        || (fixed && *std::min_element(positionMatches.begin(),positionMatches.end()) < seconds * 54)) {
        std::cerr << "Runtime feedback did not satisfy motion/validity checks\n"; return 1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(700));
    std::array<vr::TrackedDevicePose_t, vr::k_unMaxTrackedDeviceCount> stale{};
    system->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseRawAndUncalibrated, 0, stale.data(), static_cast<uint32_t>(stale.size()));
    for (auto index : indices) if (stale[index].bPoseIsValid) { std::cerr << "Stale tracking remained valid\n"; return 1; }
    std::cout << "Runtime motion and stale-tracking checks passed; avatar IK remains a separate visual check.\n";
}
