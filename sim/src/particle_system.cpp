#include "particle_system.h"

#include <iostream>

#include "kernels.h"

// idx is the outer loop index, i is the inner loop index
// TODO: this sucks, just use i and j

ParticleSystem::ParticleSystem(glm::vec2 spawn, glm::vec2 bounds, size_t num_particles, float init_mass, float init_vel, float kernel_radius)
    : bounds(bounds), kernel_radius(kernel_radius)
{
  std::random_device rd;
  for (size_t i = 0; i < num_particles; i++)
  {
    particles.push_back(Particle(spawn, rd, init_mass, init_vel));
    densities.push_back(0.0f);
    densities_grad.emplace_back(0.0f, 0.0f);
    pressure.push_back(0.0f);
    pressure_grad.emplace_back(0.0f, 0.0f);
    viscosity_forces.emplace_back(0.0f, 0.0f);
  }
}

void ParticleSystem::update(float dt)
{
  // build sharp and smooth kernels
  auto smooth_kernel = make_smoothstep_kernel(kernel_radius);
  auto smooth_kernel_derivative = make_smoothstep_kernel_derivative(kernel_radius);
  auto sharp_kernel = make_sharp_kernel(kernel_radius);
  auto sharp_kernel_derivative = make_sharp_kernel_derivative(kernel_radius);

  // copy over old positions
#pragma omp parallel for
  for (int i = 0; i < particles.size(); ++i)
  {
    particles[i].old_pos = particles[i].pos;
  }

  // predict positions
#pragma omp parallel for
  for (int i = 0; i < particles.size(); ++i)
  {
    particles[i].pos += (1 / 500.0f) * particles[i].vel;
  }

  // sort particles by x position
  // TODO: parallel sort?
  std::sort(particles.begin(), particles.end(), [](const Particle &a, const Particle &b)
            { return a.pos.x < b.pos.x; });

  // compute density for each particle
  compute_density(sharp_kernel);
  // compute pressure for each particle
  compute_attribute_grad(sharp_kernel_derivative, [&](int idx, int i)
                         { 
                          float shared_density = (densities[i] + densities[idx]) * 0.5f;
                          float pressure = (shared_density - 3.0f) * 400000;
                          // if (pressure < 0.0f)
                          //   pressure *= 0.25f;
                          return pressure; }, [&](const glm::vec2 &grad, int i)
                         { pressure_grad[i] = grad; });

  // compute viscosity forces
  constexpr float VISCOSITY = 8.0f;
#pragma omp parallel for
  for (int i = 0; i < particles.size(); ++i)
  {
    glm::vec2 viscosity_force = glm::vec2(0.0f, 0.0f);
    iterate_neighbors(i, [&](int j, int i)
                      { 
                        auto r = glm::length(particles[i].pos - particles[j].pos);
                        float influence = smooth_kernel(r);
                        viscosity_force += (particles[j].vel - particles[i].vel) * influence; });
    viscosity_forces[i] = VISCOSITY * viscosity_force;
  }

#pragma omp parallel for
  for (int i = 0; i < particles.size(); ++i)
  {
    particles[i].vel -= dt * pressure_grad[i] / densities[i];
    particles[i].vel += dt * viscosity_forces[i] / densities[i];
    particles[i].vel.y += 208.0f * dt;

    constexpr float WALL_ACCEL_PER_DIST = 12.0f;
    if (particles[i].pos.x < 0.0f)
    {
      particles[i].vel.x += WALL_ACCEL_PER_DIST * -particles[i].pos.x;
    }
    else if (particles[i].pos.x > bounds.x)
    {
      particles[i].vel.x += WALL_ACCEL_PER_DIST * (bounds.x - particles[i].pos.x);
    }
    if (particles[i].pos.y < 0.0f)
    {
      particles[i].vel.y += WALL_ACCEL_PER_DIST * -particles[i].pos.y;
    }
    else if (particles[i].pos.y > bounds.y)
    {
      particles[i].vel.y += WALL_ACCEL_PER_DIST * (bounds.y - particles[i].pos.y);
    }
    particles[i].pos = particles[i].old_pos + dt * particles[i].vel;
  }
}

void ParticleSystem::grab(const glm::vec2 &pos, float radius, float strength)
{
  for (auto &p : particles)
  {
    float r = glm::length(p.pos - pos);
    if (r < radius)
    {
      p.vel += strength * (pos - p.pos) / r;
    }
  }
}

void ParticleSystem::spin(const glm::vec2 &pos, float radius, float strength)
{
  for (auto &p : particles)
  {
    float r = glm::length(p.pos - pos);
    if (r < radius)
    {
      p.vel += radius * strength * glm::vec2(-(pos.y - p.pos.y), pos.x - p.pos.x) / (r * r + 0.01f);
    }
  }
}

void ParticleSystem::iterate_neighbors(int idx,
                                       const std::function<void(int i, int idx)> &callback)
{

  const auto &p = particles[idx].pos;

  for (int i = idx - 1; i >= 0; --i)
  {
    const auto &other_p = particles[i].pos;
    if (p.x - other_p.x > kernel_radius)
      break;

    float r = glm::length(other_p - p);
    if (r < kernel_radius && r >= 1e-6f)
      callback(i, idx);
  }

  for (int i = idx; i < particles.size(); ++i)
  {
    const auto &other_p = particles[i].pos;
    if (other_p.x - p.x > kernel_radius)
      break;

    float r = glm::length(other_p - p);
    if (r < kernel_radius && r >= 1e-6f)
      callback(i, idx);
  }

  // for (int i = 0; i < particles.size(); ++i)
  // {
  //   const auto &other_p = particles[i];
  //   if (i == idx)
  //     continue;
  //   float r = glm::length(other_p.pos - p.pos);
  //   if (r < kernel_radius && r >= 1e-6f)
  //     callback(i, idx, r, (p.pos - other_p.pos) / r);
  // }
}

void ParticleSystem::compute_attribute(
    const std::function<float(float r)> &kernel,
    const std::function<float(int idx, int i)> &getter, const std::function<void(float, int index)> &setter)
{
#pragma omp parallel for
  for (int idx = 0; idx < particles.size(); ++idx)
  {
    float attribute = 0.0f;
    iterate_neighbors(idx, [&](int i, int idx)
                      { 
                        float r = glm::length(particles[idx].pos - particles[i].pos);
                        attribute += particles[idx].mass * getter(idx, i) * kernel(r) / densities[i]; });
    setter(attribute, idx);
  }
}

void ParticleSystem::compute_attribute_grad(
    const std::function<float(float r)> &kernel_derivative,
    const std::function<float(int idx, int i)> &getter, const std::function<void(const glm::vec2 &, int index)> &setter)
{
#pragma omp parallel for
  for (int idx = 0; idx < particles.size(); ++idx)
  {
    glm::vec2 attribute_grad = glm::vec2(0.0f, 0.0f);
    iterate_neighbors(idx, [&](int i, int idx)
                      { 
                        auto dir = particles[idx].pos - particles[i].pos;
                        float r = glm::length(dir);
                        dir /= r;
                        attribute_grad += particles[i].mass * getter(idx, i) * kernel_derivative(r) * dir / densities[i]; });
    setter(attribute_grad, idx);
  }
}

void ParticleSystem::compute_density(const std::function<float(float r)> &kernel)
{
#pragma omp parallel for
  for (int idx = 0; idx < particles.size(); ++idx)
  {
    float density = 0.0f;
    iterate_neighbors(idx, [&](int i, int idx)
                      { 
                        float r = glm::length(particles[idx].pos - particles[i].pos);
                        density += particles[i].mass * kernel(r); });
    // have to add the particle's own mass, since it is dropped from the neighbor loop due to r < 1e-6f
    density += particles[idx].mass * kernel(0.0f);
    densities[idx] = density;
  }
}

void ParticleSystem::compute_density_grad(const std::function<float(float r)> &kernel_derivative)
{
#pragma omp parallel for
  for (int particle_index = 0; particle_index < particles.size(); ++particle_index)
  {
    glm::vec2 density_grad = glm::vec2(0.0f, 0.0f);
    iterate_neighbors(particle_index, [&](int i, int idx)
                      { 
                        auto dir = particles[idx].pos - particles[i].pos;
                        float r = glm::length(dir);
                        dir /= r;
                        density_grad += particles[i].mass * kernel_derivative(r) * dir; });
    densities_grad[particle_index] = density_grad;
  }
}