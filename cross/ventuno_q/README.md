# Cross-compiling llama_ros for the Arduino VENTUNO Q (aarch64, Hexagon / OpenCL)

Builds the full `llama_ros` workspace on an x86_64 Linux host for an aarch64 target, e.g. the
Arduino VENTUNO Q (Qualcomm Dragonwing IQ8 / QCS8275), with llama.cpp's Hexagon (NPU) and
Adreno (OpenCL) backends enabled.

- `Dockerfile` — Ubuntu 24.04 cross-build image with clang for `aarch64-linux-gnu`,
  `qemu-user-static`, ROS 2 Jazzy host tools and the Hexagon SDK 6.6.0.0 (from
  `ghcr.io/snapdragon-toolchain/arm64-linux:v0.7`).
- `toolchain-aarch64.cmake` — compiles against the board snapshot at `/sysroot` and searches
  `/workspace/install` as an extra prefix, which is why `--merge-install` is mandatory.

## Prerequisites

- x86_64 Linux host with Docker.
- `qemu-user-static` on the host so aarch64 binaries can run during the build:
  `sudo apt install qemu-user-static binfmt-support` (or
  `docker run --privileged --rm tonistiigi/binfmt --install arm64`).
- Board running Ubuntu 24.04 + ROS 2 Jazzy, reachable over SSH as `arduino@ventuno`
  (`ventuno` is the board's hostname or IP).

## 1. Prepare the board and create the sysroot

Install the workspace dependencies on the board so the snapshot contains matching `-dev`
packages. Only the sources are needed there (`rosdep` reads the `package.xml` files); nothing
is built on the board:

```bash
git clone https://github.com/mgonzs13/llama_ros.git ~/ros2_ws/src/llama_ros
cd ~/ros2_ws/src/llama_ros
rosdep install --from-paths . --ignore-src -y
sudo apt install build-essential python3-dev libopencv-dev
# only for the OpenCL backend:
sudo apt install ocl-icd-opencl-dev
# let the host pull the sysroot as root over non-interactive SSH:
echo "$USER ALL=(root) NOPASSWD: /usr/bin/rsync" | sudo tee /etc/sudoers.d/rsync
sudo chmod 440 /etc/sudoers.d/rsync
# optional: the sources were only needed for rosdep
rm -rf ~/ros2_ws/src/llama_ros
```

Then snapshot the board to the host:

```bash
mkdir -p ~/sysroot/opt

sudo rsync -aHAX --numeric-ids --info=progress2 --rsync-path="sudo -n rsync" \
  --exclude={"/usr/share/doc/*","/usr/share/man/*","/usr/share/locale/*","/usr/share/icons/*","/usr/share/fonts/*","/usr/lib/firmware/*","/usr/lib/modules/*","/usr/lib/debug/*","/usr/src/*"} \
  arduino@ventuno:/usr arduino@ventuno:/lib ~/sysroot/

sudo rsync -aHAX --numeric-ids --info=progress2 --rsync-path="sudo -n rsync" \
  arduino@ventuno:/opt/ros ~/sysroot/opt/

# /etc/alternatives is not copied, so update-alternatives symlinks (libblas.so.3,
# liblapack.so.3) would dangle and break linking; copy the resolved files instead:
sudo rsync -aLR --numeric-ids --rsync-path="sudo -n rsync" \
  arduino@ventuno:/usr/lib/aarch64-linux-gnu/libblas.so.3 \
  arduino@ventuno:/usr/lib/aarch64-linux-gnu/liblapack.so.3 \
  arduino@ventuno:/usr/lib/aarch64-linux-gnu/libblas.so \
  arduino@ventuno:/usr/lib/aarch64-linux-gnu/liblapack.so \
  ~/sysroot/
```

The sysroot must mirror the board: ROS belongs at `~/sysroot/opt/ros/jazzy`. Re-run all three
rsyncs after installing or removing packages on the board. If another alternatives-managed
file dangles (`find ~/sysroot -xtype l -lname '/etc/alternatives/*'`), dereference it the same
way.

## 2. Build the cross image

From the repo root:

```bash
docker build -f cross/ventuno_q/Dockerfile -t llama_ros_cross .
```

## 3. Build llama_ros

From the repo root:

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
  -DHEXAGON_TOOLS_ROOT=/opt/hexagon/6.6.0.0/tools/HEXAGON_Tools/19.0.07 \
  -DPREBUILT_LIB_DIR=linux_aarch64
```

Notes:

- Source the container's ROS, not the sysroot copy: colcon records the underlay in
  `install/setup.bash`, and `/opt/ros/jazzy` is the path that also exists on the board. The
  target packages are still resolved from the sysroot by the toolchain.
- `PREBUILT_LIB_DIR` is required by the Hexagon SDK (`hexagon_fun.cmake`); without it the
  configure fails with `string sub-command FIND requires 3 or 4 parameters`.
- Drop `-DGGML_OPENCL=ON` if the board has no Adreno OpenCL ICD (check with `clinfo` on the
  board); drop `-DBUILD_TESTING=OFF` to build and cross-run tests through QEMU.
- The backends land in `install/lib` as `libllama_ggml_hexagon.so` plus the HTP skels
  (`libggml-htp-v73.so`, `v75`, `v79`, `v81`).

To rebuild a single package while iterating:

```bash
colcon build --merge-install --packages-select llama_cpp_vendor --cmake-args ...
```

## 4. Deploy to the board

From the host:

```bash
rsync -aP install/ arduino@ventuno:~/ros2_ws/install/
```

## 5. Run on the board

Set the environment before launching. `ADSP_LIBRARY_PATH` / `CDSP_LIBRARY_PATH` are how the
FastRPC stub finds the HTP skels:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
export LD_LIBRARY_PATH=$HOME/ros2_ws/install/lib:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH=$HOME/ros2_ws/install/lib
export CDSP_LIBRARY_PATH=$HOME/ros2_ws/install/lib
ros2 launch llama_bringup base.launch.py params_file:=/path/to/params.yaml
```

Expected log lines:

```
ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev 1
ggml-hex: Hexagon Arch version vXX
load_tensors:      HTP0 model buffer size  = ...
```

With multiple NPU sessions the registry line reports `ndev N` and each session gets its own
`HTPi model buffer size` line (see "Multiple NPU sessions").

## Selecting the accelerator

`gpu.n_gpu_layers` (`-1` = all layers) offloads to an accelerator. With both backends built,
OpenCL is registered first, so the Adreno GPU is picked by default. Select the NPU explicitly:

```bash
ros2 run llama_ros llama_node --ros-args --params-file params.yaml \
  -p gpu.devices:="['HTP0']"
```

Use the device names printed in the startup log (`HTP0` for the Hexagon NPU, the Adreno
device name for OpenCL).

## Multiple NPU sessions (larger models)

A single HTP session provides ~3 GiB of mappable memory. When the model plus its contexts
does not fit (e.g. a 4B model with `draft-mtp` speculative decoding, which uses a second
context), allocate several virtual sessions and layer-split across them.

On the board, export the session count **before launching** — the sessions are registered when
the Hexagon backend loads, so `gpu.devices` alone cannot create them:

```bash
export GGML_HEXAGON_DEVICES=4   # allocates HTP0, HTP1, HTP2, HTP3
```

Then list the sessions in the params file; with `n_gpu_layers: -1` the layers are split
evenly:

```yaml
gpu:
  n_gpu_layers: -1
  devices: [HTP0, HTP1, HTP2, HTP3]
```

With `GGML_HEXAGON_DEVICES=N` the sessions are named `HTP0` .. `HTP<N-1>`. The
`physical:virtual` notation works too (`GGML_HEXAGON_DEVICES=HTP0:0,HTP0:1,HTP0:2,HTP0:3`),
but the names listed in `gpu.devices` must match it exactly. If one session is enough, leave
`gpu.devices` unset to use the default device selection. To reduce the per-session footprint,
lower `context.n_batch` and/or set `gpu.no_kv_offload: true` (moves the KV cache to the CPU),
which may let you use fewer sessions.

## Troubleshooting

- `sudo: a terminal is required to read the password` during the sysroot pull: the `NOPASSWD`
  rule from step 1 is missing on the board.
- Missing package configuration or headers while configuring: the dependency is not in the
  sysroot. Install it on the board (`rosdep install`), refresh the sysroot, and rebuild.
- `Exec format error` while building: QEMU/binfmt is not set up on the host.
- `not found: "/sysroot/.../local_setup.bash"` when sourcing `install/setup.bash` on the
  board: the install tree was built with the sysroot ROS sourced. Rebuild per step 3 and
  redeploy; for an existing tree, source `/opt/ros/jazzy/setup.bash` and
  `~/ros2_ws/install/local_setup.bash` instead.
- `warning: liblapack.so.3, needed by ... not found` or `undefined reference to 'dgemm_'`,
  `'sgesdd_'`, ... from `libopencv_core.so`, `libarmadillo.so.12`, `libarpack.so.2`: dangling
  BLAS/LAPACK symlinks in the sysroot; run the dereferencing rsync from step 1.
- `Make sure HEXAGON_SDK_ROOT ...` or `string sub-command FIND requires 3 or 4 parameters` at
  `hexagon_fun.cmake`: Hexagon SDK flags are wrong or missing; pass the `-DHEXAGON_*` and
  `-DPREBUILT_LIB_DIR=linux_aarch64` arguments from step 3.
- `ggml-hex: ... failed to open session : ... error 0x80000406` (`AEE_EUNABLETOLOAD`): the
  FastRPC stub cannot find the HTP skel. Export `ADSP_LIBRARY_PATH` (and
  `CDSP_LIBRARY_PATH`) as in step 5 and make sure `libggml-htp-vNN.so` was deployed.
- `ggml-hex: ... buffer mapping failed : ... error 0x00000001` / `fastrpc_mmap failed`: the
  model plus its contexts exceed a session's mappable address space (~3 GiB). Reduce
  `context.n_batch`, set `gpu.no_kv_offload: true`, use a smaller model or quant, or
  layer-split across virtual sessions (see "Multiple NPU sessions").
- Only Hexagon architectures v73, v75, v79 and v81 have prebuilt skels. If the runtime reports
  a different architecture, the NPU is not supported by this backend.
