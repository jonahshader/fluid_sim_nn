#pragma once

#include <functional>

#include "glm/glm.hpp"

#include "particle.h"

class ParticleSystem
{
public:
  ParticleSystem(glm::vec2 spawn, glm::vec2 bounds, size_t num_particles, float init_mass, float init_vel, float kernel_radius);

  void update(float dt);

  const std::vector<Particle> &get_particles() const { return particles; }

  void grab(const glm::vec2 &pos, float radius, float strength);

  void spin(const glm::vec2 &pos, float radius, float strength);

private:
  glm::vec2 bounds;
  float kernel_radius;
  std::vector<Particle> particles{};
  std::vector<float> densities{};
  std::vector<glm::vec2> densities_grad{};
  std::vector<float> pressure{};
  std::vector<glm::vec2> pressure_grad{};
  std::vector<glm::vec2> acceleration{};
  std::vector<glm::vec2> old_pos{};

  void iterate_neighbors(int idx, const std::function<void(int i, int idx)> &callback);

  // assumes sorted, computations are done in parallel.
  void compute_attribute(const std::function<float(float r)> &kernel,
                         const std::function<float(int idx)> &getter, const std::function<void(float, int)> &setter);

  void compute_attribute_grad(const std::function<float(float r)> &kernel_derivative,
                              const std::function<float(int idx, int i)> &getter, const std::function<void(const glm::vec2 &, int index)> &setter);

  void compute_density(const std::function<float(float r)> &kernel);

  void compute_density_grad(const std::function<float(float r)> &kernel_derivative);
};