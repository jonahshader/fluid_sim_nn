#pragma once

#include <functional>
#include <string>
#include <vector>
#include <random>

#include "SDL.h"
#include "glm/glm.hpp"

#include "soil.h"
#include "particle_system.h"

struct MouseState
{
  bool left_button_down{false};
  bool right_button_down{false};
  bool middle_button_down{false};
  glm::vec2 mouse_world_pos{0.0f, 0.0f};
  glm::vec2 mouse_screen_pos{0.0f, 0.0f};
  int scroll{0};
};

struct Tool
{
  std::string name;
  std::function<void(const MouseState &, const SDL_Event &, float)> action;
  std::function<void(const MouseState &, const SDL_MouseButtonEvent &, SDL_Renderer *, float)> render;
};

class Tools
{
public:
  Tools(Soil &soil, ParticleSystem &particles, std::mt19937 &gen);

  void update(const SDL_Event &event, float render_scale, float dt);
  void render(SDL_Renderer *renderer, float render_scale);

  void set_tool_index(int index);

private:
  std::vector<Tool> tools{};
  MouseState mouse_state{};
  int tool_index{0};
};
