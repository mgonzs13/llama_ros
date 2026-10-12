# Cross-compiling llama_ros for the Arduino VENTUNO Q (aarch64, Hexagon / OpenCL)

Builds the full `llama_ros` workspace on an x86_64 Linux host for an aarch64 target, e.g. the
Arduino VENTUNO Q (Qualcomm Dragonwing IQ8 / QCS8275), with llama.cpp's Hexagon (NPU/HTP) and
Adreno (OpenCL) backends enabled.

How it works:

- `Dockerfile` (this directory) creates an amd64 cross-build image based on Ubuntu 24.04 with
  clang targeting `aarch64-linux-gnu`, `qemu-user-static`, ROS 2 Jazzy host tools and the
  Hexagon SDK 6.6.0.0 (copied from `ghcr.io/snapdragon-toolchain/arm64-linux:v0.7`).
- `toolchain-aarch64.cmake` is passed to every package. It compiles against a sysroot mounted
  at `/sysroot` (a snapshot of the board) and treats `/workspace/install` as an additional
  search root, which is why `--merge-install` is mandatory.
- Camera/vision and other dependencies resolve from the sysroot, so the board must have the
  matching `-dev` packages installed before the snapshot is taken.

## Prerequisites

- x86_64 Linux host with Docker.
- `qemu-user-static` on the host so aarch64 binaries can run during the build:
  `sudo apt install qemu-user-static binfmt-support`
  (or `docker run --privileged --rm tonistiigi/binfmt --install arm64`).
- The board running Ubuntu 24.04 + ROS 2 Jazzy, reachable over SSH.
- `rsync` installed on the board and the host.

## 1. Prepare the board and create the sysroot

On the board:

```bash
cd ~/ros2_ws/src/llama_ros
rosdep install --from-paths . --ignore-src -y
sudo apt install build-essential python3-dev libopencv-dev
# only if you want the OpenCL backend; skip otherwise:
sudo apt install ocl-icd-opencl-dev
```

Then snapshot the target libraries to the host (run on the board):

```bash
sudo rsync -aHAX --numeric-ids \
  --exclude={"/usr/share/doc/*","/usr/share/man/*","/usr/share/locale/*","/usr/share/icons/*","/usr/share/fonts/*","/usr/lib/firmware/*","/usr/lib/modules/*","/usr/lib/debug/*","/usr/src/*"} \
  /usr /lib /opt/ros user@host:~/sysroot/
```

Re-run this after installing or removing packages on the board.

## 2. Build the cross image

On the host, from the repo root:

```bash
docker build -f cross/ventuno_q/Dockerfile -t llama_ros_cross .
```

## 3. Build llama_ros

On the host, from the repo root:

```bash
docker run --rm -it -v ~/sysroot:/sysroot -v $PWD:/workspace llama_ros_cross
```

Inside the container:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --merge-install --cmake-args \
  -DCMAKE_TOOLCHAIN_FILE=/workspace/cross/ventuno_q/toolchain-aarch64.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DGGML_OPENCL=ON \
  -DGGML_HEXAGON=ON \
  -DHEXAGON_SDK_ROOT=/opt/hexagon/6.6.0.0 \
  -DHEXAGON_TOOLS_ROOT=/opt/hexagon/6.6.0.0/tools/HEXAGON_Tools/19.0.07
```

Drop `-DGGML_OPENCL=ON` if the board has no Adreno OpenCL ICD (check with `clinfo` on the
board). Drop `-DBUILD_TESTING=OFF` to build and cross-run tests through QEMU.

`llama_cpp_vendor` exports `ggml-hexagon` and `ggml-opencl` the same way it does CUDA, so the
backends end up in `install/lib` as `libllama_ggml_hexagon.so` plus the HTP skels
(`libggml-htp-v73.so`, `v75`, `v79`, `v81`).

To rebuild a single package while iterating:

```bash
colcon build --merge-install --packages-select llama_cpp_vendor --cmake-args ...
```

## 4. Deploy to the board

From the host:

```bash
rsync -aP install/ user@ventuno:~/ros2_ws/install/
```

## 5. Run on the board

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
export LD_LIBRARY_PATH=$HOME/ros2_ws/install/lib:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH=$HOME/ros2_ws/install/lib
ros2 launch llama_bringup base.launch.py params_file:=/path/to/params.yaml
```

Expected log lines:

```
ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev 1
ggml-hex: Hexagon Arch version vXX
load_tensors:      HTP0 model buffer size  = ...
```

## Selecting the accelerator

`gpu.n_gpu_layers` (`-1` = all layers) offloads to an accelerator. With both backends built,
OpenCL is registered first, so the Adreno GPU is picked by default; select the NPU explicitly:

```bash
ros2 run llama_ros llama_node --ros-args --params-file params.yaml \
  -p gpu.devices:="['HTP0']"
```

Use the device names printed in the startup log (e.g. `HTP0` for the Hexagon NPU, the Adreno
device name for OpenCL).

## Troubleshooting

- `Could not find a package configuration file ...` or missing headers during configure:
  the dependency is not in the sysroot. Install it on the board (usually via
  `rosdep install`), re-create the sysroot, and rebuild.
- `Exec format error` while building: QEMU/binfmt is not set up on the host.
- `Make sure HEXAGON_SDK_ROOT point to the correct Hexagon SDK installation`: pass
  `-DHEXAGON_SDK_ROOT=...` and `-DHEXAGON_TOOLS_ROOT=...` as shown above.
- Hexagon backend fails to load the skel at runtime: verify `ADSP_LIBRARY_PATH` points at
  the deployed `lib` directory and that FastRPC (`libcdsprpc.so`, `cdsp` domain) is available
  on the board.
- Only Hexagon architectures v73, v75, v79 and v81 have prebuilt skels. If the runtime reports
  a different architecture, the NPU is not supported by this backend.
