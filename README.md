# fluid_sim_nn

particle based fluid sim -> grid based fluid sim via neural nets

download [meson](https://mesonbuild.com/)


WINDOWS:

cd sim

for debug build:
meson setup builddir --backend=vs --vsenv
meson compile -C builddir
.\builddir\fluid_sim.exe

for release build:
meson setup builddir_release --backend=vs --vsenv --buildtype=release
meson compile -C builddir_release
.\builddir_release\fluid_sim.exe

LINUX:
idk probably similar, but no custom backend or vsenv stuff