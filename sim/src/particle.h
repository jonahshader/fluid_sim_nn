#pragma once

#include <cmath>
#include <random>

#include "glm/glm.hpp"

#include "custom_math.h"

struct Particle
{
  glm::vec2 pos{};
  glm::vec2 old_pos{};
  glm::vec2 vel{};
  glm::vec2 old_vel{};

  glm::vec2 pos_k[4];
  glm::vec2 vel_k[4];
  float mass{1.0f};
  int grid_id{0};

  Particle(glm::vec2 spawn, std::random_device &rd, float init_mass, glm::vec2 init_vel, float kernel_radius)
  {
    std::mt19937 gen(rd());
    mass = init_mass;
    std::uniform_real_distribution<float> dis_x(0.0f, spawn.x);
    std::uniform_real_distribution<float> dis_y(0.0f, spawn.y);
    std::uniform_real_distribution<float> dis_vel_dir(0.0f, 2.0f * M_PI_F);
    pos = glm::vec2(dis_x(gen), dis_y(gen));
    old_pos = pos;
    vel = init_vel;
    // glm::mat2 rot = glm::mat2(glm::cos(dis_vel_dir(gen)), -glm::sin(dis_vel_dir(gen)),
    //                           glm::sin(dis_vel_dir(gen)), glm::cos(dis_vel_dir(gen)));
    // vel = rot * vel;
    // old_vel = vel;
  }
};
