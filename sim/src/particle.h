#pragma once

#include <random>

#include "glm/glm.hpp"

#include "custom_math.h"

struct Particle
{
  glm::vec2 pos{};
  glm::vec2 old_pos{};
  glm::vec2 vel{};
  float mass{1.0f};

  Particle(glm::vec2 spawn, std::random_device &rd, float init_mass, float init_vel)
  {
    std::mt19937 gen(rd());
    mass = init_mass;
    std::uniform_real_distribution<float> dis_x(0.0f, spawn.x);
    std::uniform_real_distribution<float> dis_y(0.0f, spawn.y);
    std::uniform_real_distribution<float> dis_vel_dir(0.0f, 2.0f * M_PI_F);
    pos = glm::vec2(dis_x(gen), dis_y(gen));
    old_pos = pos;
    vel = glm::vec2(init_vel, 0.0f);
    glm::mat2 rot = glm::mat2(glm::cos(dis_vel_dir(gen)), -glm::sin(dis_vel_dir(gen)),
                              glm::sin(dis_vel_dir(gen)), glm::cos(dis_vel_dir(gen)));
    vel = rot * vel;
  }
};