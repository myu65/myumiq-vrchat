#include <openvr.h>
#include <chrono>
#include <thread>
#include <iostream>
#include <charconv>
#include <cstring>
#include <array>

int main(int argc, char** argv) {
    int seconds = 60;
    if (argc > 2) return 2;
    if (argc == 2) {
        const auto end = argv[1] + std::strlen(argv[1]);
        const auto parsed = std::from_chars(argv[1],end,seconds);
        if (parsed.ec != std::errc{} || parsed.ptr != end || seconds < 1 || seconds > 600) {
            std::cerr << "Usage: frame_probe [seconds=60, range 1..600]\n"; return 2;
        }
    }
    vr::EVRInitError error;
    vr::VR_Init(&error,vr::VRApplication_Background);
    if(error != vr::VRInitError_None) { std::cerr << vr::VR_GetVRInitErrorAsEnglishDescription(error) << '\n'; return 1; }
    struct Shutdown { ~Shutdown(){vr::VR_Shutdown();} } shutdown;
    auto compositor = vr::VRCompositor();
    if(!compositor) { std::cerr << "No compositor\n"; return 1; }
    vr::Compositor_CumulativeStats before{}, after{};
    compositor->GetCumulativeStats(&before,sizeof(before));
    const auto pid = compositor->GetCurrentSceneFocusProcess();
    if(!pid || before.m_nPid != pid) { std::cerr << "No active scene stats\n"; return 1; }
    const auto start=std::chrono::steady_clock::now();
    std::cout << "scene_pid=" << pid << '\n';
    std::array<vr::TrackedDevicePose_t,vr::k_unMaxTrackedDeviceCount> render{},game{};
    const auto poseError=compositor->GetLastPoses(render.data(),static_cast<uint32_t>(render.size()),game.data(),static_cast<uint32_t>(game.size()));
    std::cout << "last_pose_error=" << poseError << '\n';
    if(poseError==vr::VRCompositorError_None) for(int i=0;i<3;++i) {
        std::cout << "render_device=" << i << " valid=" << render[i].bPoseIsValid
                  << " result=" << render[i].eTrackingResult << " game_valid=" << game[i].bPoseIsValid
                  << " xyz=" << render[i].mDeviceToAbsoluteTracking.m[0][3] << ','
                  << render[i].mDeviceToAbsoluteTracking.m[1][3] << ',' << render[i].mDeviceToAbsoluteTracking.m[2][3] << '\n';
    }
    for(int i=0;i<seconds;++i) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if(compositor->GetCurrentSceneFocusProcess()!=pid) { std::cerr << "Scene changed; discard measurement\n"; return 1; }
    }
    compositor->GetCumulativeStats(&after,sizeof(after));
    if(after.m_nPid!=pid || after.m_nNumFrameSubmits<=before.m_nNumFrameSubmits
        || after.m_nNumFramePresents<before.m_nNumFramePresents
        || after.m_nNumDroppedFrames<before.m_nNumDroppedFrames
        || after.m_nNumReprojectedFrames<before.m_nNumReprojectedFrames) {
        std::cerr << "No new scene frames or counters reset; discard measurement\n"; return 1;
    }
    const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    const auto submits=after.m_nNumFrameSubmits-before.m_nNumFrameSubmits;
    std::cout << "elapsed_s=" << elapsed << " submits=" << submits << " submitted_fps=" << submits/elapsed
              << " presents=" << after.m_nNumFramePresents-before.m_nNumFramePresents
              << " dropped=" << after.m_nNumDroppedFrames-before.m_nNumDroppedFrames
              << " reprojected=" << after.m_nNumReprojectedFrames-before.m_nNumReprojectedFrames << '\n'
              << "app_cpu_ms=" << (after.m_flSumApplicationCPUTimeMS-before.m_flSumApplicationCPUTimeMS)/submits
              << " app_gpu_ms=" << (after.m_flSumApplicationGPUTimeMS-before.m_flSumApplicationGPUTimeMS)/submits
              << " compositor_cpu_ms=" << (after.m_flSumCompositorCPUTimeMS-before.m_flSumCompositorCPUTimeMS)/submits
              << " compositor_gpu_ms=" << (after.m_flSumCompositorGPUTimeMS-before.m_flSumCompositorGPUTimeMS)/submits << '\n';
}
