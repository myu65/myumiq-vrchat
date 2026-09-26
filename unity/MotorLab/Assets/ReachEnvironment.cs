using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

// Independent training scene. Canonical vectors are forward/left/up in metres.
public sealed class ReachEnvironment : MonoBehaviour
{
    [Serializable] public sealed class Request
    {
        public string token;
        public string command;
        public int seed;
        public float[] action;
        public float[] goal;
    }
    [Serializable] public sealed class Response
    {
        public string protocol = "myumiq-reach-v1";
        public string error;
        public float[] observation;
        public float[] hand;
        public float[] goal;
        public float reward;
        public bool terminated;
        public bool truncated;
        public bool success;
        public int step;
    }
    [Serializable] public sealed class Ready
    {
        public string protocol = "myumiq-reach-v1";
        public string token;
        public int port;
        public int pid;
    }
    sealed class Work
    {
        public string json;
        public TaskCompletionSource<string> answer = new TaskCompletionSource<string>();
    }

    const float Dt = 1f / 30f;
    const float Speed = 0.6f;
    const float Acceleration = 1.8f;
    readonly Vector3 shoulder = new Vector3(0.05f, -0.2f, 1.35f);
    readonly ConcurrentQueue<Work> pending = new ConcurrentQueue<Work>();
    TcpListener listener;
    Thread network;
    volatile bool running;
    string token;
    Vector3 hand, goal, velocity;
    int steps;
    bool done = true;
    Transform handMarker, goalMarker;

    static Vector3 ToUnity(Vector3 canonical) => new Vector3(-canonical.y, canonical.z, canonical.x);
    static float[] Values(Vector3 value) => new[] { value.x, value.y, value.z };
    static bool Finite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);

    void Start()
    {
        Application.runInBackground = true;
        QualitySettings.vSyncCount = 0;
        Application.targetFrameRate = -1;
        handMarker = Marker("Right hand", Color.cyan, 0.07f);
        goalMarker = Marker("Reach target", Color.green, 0.05f);
        Marker("Right shoulder", Color.gray, 0.08f).position = ToUnity(shoulder);
        token = Guid.NewGuid().ToString("N");
        listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start(1);
        running = true;
        network = new Thread(Serve) { IsBackground = true };
        network.Start();
        var args = Environment.GetCommandLineArgs();
        string readyPath = null;
        for (int i = 0; i + 1 < args.Length; i++)
            if (args[i] == "--motorlab-ready") readyPath = args[i + 1];
        if (readyPath == null || File.Exists(readyPath))
        {
            Debug.LogError("A new --motorlab-ready path is required");
            Application.Quit(2);
            return;
        }
        File.WriteAllText(readyPath, JsonUtility.ToJson(new Ready {
            token = token, port = ((IPEndPoint)listener.LocalEndpoint).Port,
            pid = System.Diagnostics.Process.GetCurrentProcess().Id
        }));
    }

    Transform Marker(string name, Color color, float size)
    {
        var obj = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        obj.name = name;
        obj.transform.localScale = Vector3.one * size;
        obj.GetComponent<Renderer>().material.color = color;
        return obj.transform;
    }

    void Serve()
    {
        try
        {
            using (var client = listener.AcceptTcpClient())
            {
                client.NoDelay = true;
                using (var stream = client.GetStream())
                using (var reader = new StreamReader(stream, Encoding.UTF8))
                using (var writer = new StreamWriter(stream, new UTF8Encoding(false)) { AutoFlush = true })
                {
                    while (running)
                    {
                        string line = reader.ReadLine();
                        if (line == null || line.Length > 4096) break;
                        var work = new Work { json = line };
                        pending.Enqueue(work);
                        if (!work.answer.Task.Wait(10000)) break;
                        writer.WriteLine(work.answer.Task.Result);
                    }
                }
            }
        }
        catch (Exception ex) { if (running) Debug.LogError(ex.Message); }
        finally { running = false; }
    }

    void Update()
    {
        if (!running) { Application.Quit(); return; }
        for (int i = 0; i < 128 && pending.TryDequeue(out var work); i++)
        {
            Response response;
            try
            {
                var request = JsonUtility.FromJson<Request>(work.json);
                if (request == null || request.token != token) throw new ArgumentException("invalid session");
                response = Execute(request);
            }
            catch (Exception ex) { response = new Response { error = ex.Message }; }
            work.answer.SetResult(JsonUtility.ToJson(response));
        }
    }

    Response Execute(Request request)
    {
        float reward = 0;
        bool success = false;
        if (request.command == "reset")
        {
            var rng = new System.Random(request.seed);
            hand = new Vector3(0.2f, -0.25f, 1.15f);
            velocity = Vector3.zero;
            goal = request.goal == null || request.goal.Length == 0
                ? new Vector3(0.3f + (float)rng.NextDouble() * 0.2f,
                              -0.15f - (float)rng.NextDouble() * 0.3f,
                              1.05f + (float)rng.NextDouble() * 0.45f)
                : ReadVector(request.goal);
            if (Vector3.Distance(goal, shoulder) > 0.65f || goal.z < 0.6f)
                throw new ArgumentException("goal outside REACH workspace");
            steps = 0;
            done = false;
        }
        else if (request.command == "step")
        {
            if (done) throw new InvalidOperationException("reset required");
            Vector3 action = ReadVector(request.action);
            if (Mathf.Max(Mathf.Abs(action.x), Mathf.Abs(action.y), Mathf.Abs(action.z)) > 1)
                throw new ArgumentException("action outside -1..1");
            float before = Vector3.Distance(hand, goal);
            Vector3 oldVelocity = velocity;
            velocity = Vector3.MoveTowards(velocity, Vector3.ClampMagnitude(action, 1) * Speed,
                                           Acceleration * Dt);
            Vector3 next = hand + velocity * Dt;
            if (Vector3.Distance(next, shoulder) > 0.65f || next.z < 0.6f)
            {
                velocity = Vector3.zero;
                reward -= 0.25f;
            }
            else hand = next;
            float distance = Vector3.Distance(hand, goal);
            success = distance < 0.025f && velocity.magnitude < 0.12f;
            reward += 5 * (before - distance) - distance * Dt
                      - 0.01f * (velocity - oldVelocity).sqrMagnitude;
            if (success) reward += 2;
            steps++;
            done = success || steps >= 150;
        }
        else if (request.command == "close")
        {
            Application.Quit();
        }
        else throw new ArgumentException("unknown command");
        handMarker.position = ToUnity(hand);
        goalMarker.position = ToUnity(goal);
        Vector3 delta = goal - hand;
        Vector3 relative = hand - shoulder;
        return new Response {
            observation = new[] { delta.x, delta.y, delta.z, relative.x, relative.y, relative.z,
                                  velocity.x, velocity.y, velocity.z },
            hand = Values(hand), goal = Values(goal), reward = reward,
            terminated = success, truncated = steps >= 150, success = success, step = steps
        };
    }

    static Vector3 ReadVector(float[] values)
    {
        if (values == null || values.Length != 3 || !Array.TrueForAll(values, Finite))
            throw new ArgumentException("expected three finite values");
        return new Vector3(values[0], values[1], values[2]);
    }

    void OnDestroy() { running = false; listener?.Stop(); }
}
