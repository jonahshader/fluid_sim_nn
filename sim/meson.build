project('fluid_nn', 'cpp', default_options: ['default_library=static', 'cpp_std=c++20'])

sdl2_deps = dependency('sdl2')
glm_deps = dependency('glm')
omp_deps = dependency('openmp')

deps = [sdl2_deps, glm_deps, omp_deps]

if build_machine.system() == 'windows'
  deps += [dependency('sdl2main')]
endif

src = ['src/main.cpp', 'src/particle.cpp', 'src/particle_system.cpp', 'src/particle_render.cpp', 'src/tools.cpp', 'src/soil.cpp', 'src/custom_math.cpp']
# add win_subsystem: 'windows' to remove console window on windows
executable('fluid_sim', src, dependencies: deps)
