# fluid_sim_nn

A neural network-based approach to convert particle-based fluid simulations to grid-based fluid simulations.

## Prerequisites

Before you begin, ensure you have [Meson](https://mesonbuild.com/) installed on your system.

## Building and Running

### Windows

Navigate to the `sim` directory and follow these steps:

#### Debug Build

```powershell
meson setup builddir --backend=vs --vsenv
meson compile -C builddir
.\builddir\fluid_sim.exe
```

#### Release Build

```powershell
meson setup builddir_release --backend=vs --vsenv --buildtype=release
meson compile -C builddir_release
.\builddir_release\fluid_sim.exe
```

### Linux

Navigate to the `sim` directory and follow these steps:

#### Debug Build

```bash
meson setup builddir
meson compile -C builddir
./builddir/fluid_sim
```

#### Release Build

```bash
meson setup builddir_release --buildtype=release
meson compile -C builddir_release
./builddir_release/fluid_sim
```
