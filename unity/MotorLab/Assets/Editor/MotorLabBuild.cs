using System;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEditor.Build.Reporting;
using UnityEngine;

public static class MotorLabBuild
{
    public static void Build()
    {
        string output = null;
        var args = Environment.GetCommandLineArgs();
        for (int i = 0; i + 1 < args.Length; i++)
            if (args[i] == "--motorlab-build-path") output = args[i + 1];
        if (output == null) throw new ArgumentException("--motorlab-build-path required");
        var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        new GameObject("Motor learning environment").AddComponent<ReachEnvironment>();
        var camera = new GameObject("Camera").AddComponent<Camera>();
        camera.transform.position = new Vector3(1.8f, 1.6f, -1.4f);
        camera.transform.LookAt(new Vector3(0.25f, 1.3f, 0.3f));
        var light = new GameObject("Light").AddComponent<Light>();
        light.type = LightType.Directional;
        light.transform.rotation = Quaternion.Euler(40, -30, 0);
        EditorSceneManager.SaveScene(scene, "Assets/MotorLab.unity");
        PlayerSettings.productName = "MyuMIQ MotorLab";
        PlayerSettings.runInBackground = true;
        PlayerSettings.defaultScreenWidth = 800;
        PlayerSettings.defaultScreenHeight = 600;
        PlayerSettings.fullScreenMode = FullScreenMode.Windowed;
        var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions {
            scenes = new[] { "Assets/MotorLab.unity" }, locationPathName = output,
            target = BuildTarget.StandaloneWindows64, options = BuildOptions.None
        });
        if (report.summary.result != BuildResult.Succeeded)
            throw new Exception("MotorLab build failed: " + report.summary.result);
    }
}
