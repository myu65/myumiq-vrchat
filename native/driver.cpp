#include "display.hpp"
#include "protocol.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>

namespace myumiq {
using Clock = std::chrono::steady_clock;
class Device final : public vr::ITrackedDeviceServerDriver {
    int role_;
    std::atomic<uint32_t> index_{vr::k_unTrackedDeviceIndexInvalid};
    std::mutex mutex_;
    Target target_;
    Clock::time_point received_ = Clock::now();
    Display display_;
    double displayHz_ = 60;
    Clock::time_point lastVsync_ = Clock::now();
    vr::VRInputComponentHandle_t trigger_ = 0, grip_ = 0, proximity_ = 0;
public:
    explicit Device(int role) : role_(role) {
        if (role) target_.position = {0.35, role == 1 ? 0.25 : -0.25, 1.2};
    }
    vr::EVRInitError Activate(uint32_t index) override {
        auto p = vr::VRProperties();
        auto c = p->TrackedDeviceToPropertyContainer(index);
        p->SetStringProperty(c, vr::Prop_ModelNumber_String, role_ ? "MyuMIQ Controller" : "MyuMIQ HMD");
        p->SetStringProperty(c, vr::Prop_TrackingSystemName_String, "myumiq");
        p->SetUint64Property(c, vr::Prop_CurrentUniverseId_Uint64, 20260914);
        if (!role_) {
            vr::EVRSettingsError error = vr::VRSettingsError_None;
            int size = vr::VRSettings()->GetInt32("driver_myumiq", "render_size", &error);
            if (error != vr::VRSettingsError_None || size < 128 || size > 2048) return vr::VRInitError_Driver_Failed;
            float hz = vr::VRSettings()->GetFloat("driver_myumiq", "refresh_hz", &error);
            if (error != vr::VRSettingsError_None || !std::isfinite(hz) || hz < 20 || hz > 120) return vr::VRInitError_Driver_Failed;
            display_.size = static_cast<uint32_t>(size);
            displayHz_ = hz;
            lastVsync_ = Clock::now();
            p->SetFloatProperty(c, vr::Prop_DisplayFrequency_Float, hz);
            p->SetBoolProperty(c, vr::Prop_ReportsTimeSinceVSync_Bool, true);
            p->SetFloatProperty(c, vr::Prop_UserIpdMeters_Float, 0.064f);
            p->SetFloatProperty(c, vr::Prop_UserHeadToEyeDepthMeters_Float, 0.f);
            p->SetFloatProperty(c, vr::Prop_SecondsFromVsyncToPhotons_Float, 1.f / hz);
            // Match the nonphysical windowed-HMD path in Valve's sample.
            p->SetBoolProperty(c, vr::Prop_IsOnDesktop_Bool, false);
            p->SetBoolProperty(c, vr::Prop_DisplayDebugMode_Bool, true);
            p->SetBoolProperty(c, vr::Prop_ContainsProximitySensor_Bool, true);
            if(vr::VRDriverInput()->CreateBooleanComponent(c,"/proximity",&proximity_) != vr::VRInputError_None)
                return vr::VRInitError_Driver_Failed;
        } else {
            p->SetInt32Property(c, vr::Prop_ControllerRoleHint_Int32,
                role_ == 1 ? vr::TrackedControllerRole_LeftHand : vr::TrackedControllerRole_RightHand);
            p->SetStringProperty(c, vr::Prop_ControllerType_String, "myumiq_controller");
            p->SetStringProperty(c, vr::Prop_InputProfilePath_String, "{myumiq}/input/controller_profile.json");
            auto input = vr::VRDriverInput();
            if (input->CreateScalarComponent(c, "/input/trigger/value", &trigger_, vr::VRScalarType_Absolute,
                    vr::VRScalarUnits_NormalizedOneSided) != vr::VRInputError_None
                || input->CreateBooleanComponent(c, "/input/grip/click", &grip_) != vr::VRInputError_None)
                return vr::VRInitError_Driver_Failed;
        }
        index_ = index;
        vr::VRDriverLog()->Log(role_ ? "Controller activated" : "HMD activated (experimental windowed display)");
        return vr::VRInitError_None;
    }
    void Deactivate() override { index_ = vr::k_unTrackedDeviceIndexInvalid; }
    void EnterStandby() override {}
    void* GetComponent(const char* name) override {
        return !role_ && name && std::strcmp(name, vr::IVRDisplayComponent_Version) == 0 ? &display_ : nullptr;
    }
    void DebugRequest(const char* request, char* response, uint32_t size) override {
        Target next;
        bool ok = request && parse(request, next);
        if (ok && !role_ && (next.trigger != 0 || next.grip != 0)) ok = false;
        if (ok) {
            std::lock_guard<std::mutex> lock(mutex_);
            target_ = next;
            received_ = Clock::now();
        }
        if (response && size) std::snprintf(response, size, "%s", ok ? "ok" : "error: invalid v1 target");
    }
    vr::DriverPose_t GetPose() override {
        std::lock_guard<std::mutex> lock(mutex_);
        vr::DriverPose_t pose{};
        pose.qWorldFromDriverRotation.w = pose.qDriverFromHeadRotation.w = 1;
        pose.shouldApplyHeadModel = role_ == 0;
        const auto p = to_openvr(target_.position);
        const auto q = to_openvr({target_.rotation[1], target_.rotation[2], target_.rotation[3]});
        for (int i = 0; i < 3; ++i) pose.vecPosition[i] = p[i];
        pose.qRotation = {target_.rotation[0], q[0], q[1], q[2]};
        pose.deviceIsConnected = true;
        pose.poseIsValid = Clock::now() - received_ <= std::chrono::milliseconds(500);
        pose.result = pose.poseIsValid ? vr::TrackingResult_Running_OK : vr::TrackingResult_Running_OutOfRange;
        return pose;
    }
    void RunFrame() {
        const auto index = index_.load();
        if (index == vr::k_unTrackedDeviceIndexInvalid) return;
        auto pose = GetPose();
        vr::VRServerDriverHost()->TrackedDevicePoseUpdated(index, pose, sizeof(pose));
        if (!role_) {
            // Virtual activity, not a claim that a human is wearing hardware.
            vr::VRDriverInput()->UpdateBooleanComponent(proximity_,pose.poseIsValid,0);
            // A nonphysical display has no hardware vsync source. Publish a
            // software clock so the compositor can predict a finite HMD pose.
            const auto now = Clock::now();
            const double elapsed = std::chrono::duration<double>(now-lastVsync_).count();
            const auto ticks = std::floor(elapsed*displayHz_);
            if(ticks >= 1) {
                lastVsync_ += std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(ticks/displayHz_));
                vr::VRServerDriverHost()->VsyncEvent(-std::chrono::duration<double>(now-lastVsync_).count());
            }
        }
        if (role_) {
            std::lock_guard<std::mutex> lock(mutex_);
            const bool fresh = Clock::now() - received_ <= std::chrono::milliseconds(500);
            vr::VRDriverInput()->UpdateScalarComponent(trigger_, fresh ? static_cast<float>(target_.trigger) : 0, 0);
            vr::VRDriverInput()->UpdateBooleanComponent(grip_, fresh && target_.grip >= 0.5, 0);
        }
    }
};

class Provider final : public vr::IServerTrackedDeviceProvider {
    std::array<std::unique_ptr<Device>, 3> devices_;
public:
    vr::EVRInitError Init(vr::IVRDriverContext* context) override {
        VR_INIT_SERVER_DRIVER_CONTEXT(context);
        const char* serials[] = {"myumiq-hmd", "myumiq-left", "myumiq-right"};
        for (int i = 0; i < 3; ++i) {
            devices_[i] = std::make_unique<Device>(i);
            if (!vr::VRServerDriverHost()->TrackedDeviceAdded(serials[i], i ? vr::TrackedDeviceClass_Controller : vr::TrackedDeviceClass_HMD, devices_[i].get())) {
                vr::VRDriverLog()->Log("Device registration failed; feasibility not established");
                return vr::VRInitError_Driver_Failed;
            }
        }
        return vr::VRInitError_None;
    }
    void Cleanup() override {
        for (auto& device : devices_) { if (device) device->Deactivate(); device.reset(); }
        VR_CLEANUP_SERVER_DRIVER_CONTEXT();
    }
    const char* const* GetInterfaceVersions() override { return vr::k_InterfaceVersions; }
    void RunFrame() override { for (auto& device : devices_) if (device) device->RunFrame(); }
    bool ShouldBlockStandbyMode() override { return true; }
    void EnterStandby() override {}
    void LeaveStandby() override {}
};
Provider provider;
}
extern "C" __declspec(dllexport) void* HmdDriverFactory(const char* name, int* error) {
    if (name && std::strcmp(name, vr::IServerTrackedDeviceProvider_Version) == 0) {
        if (error) *error = vr::VRInitError_None;
        return &myumiq::provider;
    }
    if (error) *error = vr::VRInitError_Init_InterfaceNotFound;
    return nullptr;
}
