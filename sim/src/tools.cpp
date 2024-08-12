#include "tools.h"

#include <cmath>
#include <iostream>

void Tools::set_tool_index(int index)
{
  tool_index = index % tools.size();
  if (tool_index < 0)
  {
    tool_index += tools.size();
  }

  std::cout << "Selected tool: " << tools[tool_index].name << std::endl;
}

Tools::Tools(Soil &soil, ParticleSystem &particles)
{
  auto get_radius = [](const MouseState &mouse_state) -> float
  {
    return std::pow(1.25f, mouse_state.scroll);
  };
  auto render_circle = [&](const MouseState &mouse_state, const SDL_MouseButtonEvent &event, SDL_Renderer *renderer, float render_scale)
  {
    // render a circle at the radius
    glm::vec2 pos = mouse_state.mouse_screen_pos;
    float radius = get_radius(mouse_state) * render_scale;
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    // use a 16 sided polygon to approximate a circle
    glm::vec2 last_p = pos + glm::vec2{radius, 0.0f};
    for (int i = 0; i <= 16; ++i)
    {
      float angle = i / 16.0f * 2.0f * M_PI;
      glm::vec2 p = pos + glm::vec2{std::cos(angle), std::sin(angle)} * radius;
      SDL_RenderDrawLine(renderer, last_p.x, last_p.y, p.x, p.y);
      last_p = p;
    }
  };

  tools.push_back({"Grab", [&, render_circle](const MouseState &mouse_state, const SDL_Event &event, float dt)
                   {
                     if (mouse_state.left_button_down || mouse_state.right_button_down)
                     {
                       glm::vec2 pos = mouse_state.mouse_world_pos;
                       float radius = get_radius(mouse_state);
                       float force = mouse_state.left_button_down ? 1000.0f : 0.0;
                       force += mouse_state.right_button_down ? -1000.0f : 0.0f;
                       particles.grab(pos, radius, force * dt);
                     } },
                   render_circle});
  tools.push_back({"Spin", [&](const MouseState &mouse_state, const SDL_Event &event, float dt)
                   {
                     if (mouse_state.left_button_down || mouse_state.right_button_down)
                     {
                       glm::vec2 pos = mouse_state.mouse_world_pos;
                       float radius = get_radius(mouse_state);
                       float force = mouse_state.left_button_down ? 100.0f : 0.0;
                       force += mouse_state.right_button_down ? -100.0f : 0.0f;
                       particles.spin(pos, radius, force * dt);
                     } },
                   render_circle});
  tools.push_back({"Add Soil Particle", [&](const MouseState &mouse_state, const SDL_Event &event, float dt)
                   {
                    if (event.type == SDL_MOUSEBUTTONDOWN)
                    {
                      if (event.button.button == SDL_BUTTON_LEFT)
                      {
                        glm::vec2 pos = mouse_state.mouse_world_pos;
                        float radius = get_radius(mouse_state);
                        soil.add({pos, radius});
                      }
                    } }, render_circle});
  tools.push_back({"Add Soil Patch", [&](const MouseState &mouse_state, const SDL_Event &event, float dt)
                   {
                    constexpr float ADHESION_RADIUS = 0.75f;
                    if (event.type == SDL_MOUSEBUTTONDOWN)
                    {
                      if (event.button.button == SDL_BUTTON_LEFT)
                      {
                        glm::vec2 start_pos = mouse_state.mouse_world_pos;
                        float radius = get_radius(mouse_state);
                        float spacing = (2.0f * radius) + ADHESION_RADIUS * 1.5f;
                        // place in hex grid
                        for (int y = -5; y <= 5; ++y)
                        {
                          for (int x = -5; x <= 5; ++x)
                          {
                            float offset = y % 2 == 0 ? 0.0f : spacing * 0.5f;
                            glm::vec2 pos = start_pos + glm::vec2{x * spacing + offset, y * spacing / std::sqrt(3.0f/2.0f)};

                            SoilParticle p;
                            p.pos = pos;
                            p.radius = radius;
                            p.adhesion_radius = ADHESION_RADIUS;
                            soil.add(p);
                          }
                        }
                      }
                    } }, render_circle});
}

void Tools::update(const SDL_Event &event, float render_scale, float dt)
{
  auto b = event.button.button;
  // update mouse state
  if (event.type == SDL_MOUSEBUTTONDOWN)
  {
    if (b == SDL_BUTTON_LEFT)
      mouse_state.left_button_down = true;
    else if (b == SDL_BUTTON_RIGHT)
      mouse_state.right_button_down = true;
  }
  else if (event.type == SDL_MOUSEBUTTONUP)
  {
    if (b == SDL_BUTTON_LEFT)
      mouse_state.left_button_down = false;
    else if (b == SDL_BUTTON_RIGHT)
      mouse_state.right_button_down = false;
  }

  if (event.type == SDL_MOUSEWHEEL)
  {
    mouse_state.scroll += event.wheel.y;
  }

  // check for tool change (1-9 keys)
  if (event.type == SDL_KEYDOWN)
  {
    if (event.key.keysym.sym >= SDLK_1 && event.key.keysym.sym <= SDLK_9)
    {
      set_tool_index(event.key.keysym.sym - SDLK_1);
    }
  }

  // update mouse coordinates
  int mouse_x, mouse_y;
  SDL_GetMouseState(&mouse_x, &mouse_y);
  mouse_state.mouse_screen_pos = {static_cast<float>(mouse_x), static_cast<float>(mouse_y)};
  mouse_state.mouse_world_pos = mouse_state.mouse_screen_pos / render_scale;

  // call the action of the selected tool
  tools[tool_index].action(mouse_state, event, dt);
}

void Tools::render(SDL_Renderer *renderer, float render_scale)
{
  // call the render function of the selected tool
  tools[tool_index].render(mouse_state, {0, 0, 0}, renderer, render_scale);
}
