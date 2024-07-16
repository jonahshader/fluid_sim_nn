#include "SDL.h"
#include "glm/glm.hpp"

#include "particle_system.h"
#include "particle_render.h"

#include <iostream>

bool mouse_left_down = false;
bool mouse_right_down = false;
bool mouse_middle_down = false;

int main(int argc, char *argv[])
{
  std::cout << "program start" << std::endl;
  SDL_Window *window;
  SDL_Renderer *renderer;
  SDL_Surface *surface;
  SDL_Event event;

  ParticleSystem particles({64.0f * 16.0f / 9.0f, 48.0f}, {64.0f * 16.0f / 9.0f, 64.0f}, 8000, 1.0f, 0.0f, 2.5f);

  if (SDL_Init(SDL_INIT_VIDEO) < 0)
  {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Couldn't initialize SDL: %s", SDL_GetError());
    return 3;
  }

  window = SDL_CreateWindow("Fluid Sim Vis", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1280, 720, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
  if (window == NULL)
  {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Couldn't create window: %s", SDL_GetError());
    return 3;
  }

  renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
  if (renderer == NULL)
  {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Couldn't create renderer: %s", SDL_GetError());
    return 3;
  }

  while (1)
  {
    SDL_PollEvent(&event);
    if (event.type == SDL_QUIT)
    {
      break;
    }
    else if (event.type == SDL_MOUSEBUTTONDOWN)
    {
      if (event.button.button == SDL_BUTTON_LEFT)
      {
        mouse_left_down = true;
      }
      else if (event.button.button == SDL_BUTTON_RIGHT)
      {
        mouse_right_down = true;
      }
      else if (event.button.button == SDL_BUTTON_MIDDLE)
      {
        mouse_middle_down = true;
      }
    }
    else if (event.type == SDL_MOUSEBUTTONUP)
    {
      if (event.button.button == SDL_BUTTON_LEFT)
      {
        mouse_left_down = false;
      }
      else if (event.button.button == SDL_BUTTON_RIGHT)
      {
        mouse_right_down = false;
      }
      else if (event.button.button == SDL_BUTTON_MIDDLE)
      {
        mouse_middle_down = false;
      }
    }
    SDL_SetRenderDrawColor(renderer, 0x00, 0x00, 0x00, 0x00);
    SDL_RenderClear(renderer);

    // calculate render_scale so that the bounds fit the screen
    int screen_width, screen_height;
    SDL_GetWindowSize(window, &screen_width, &screen_height);
    auto bounds = particles.get_bounds();
    float render_scale = std::min(screen_width / bounds.x, screen_height / bounds.y);

    // calculate mouse world position
    int mouse_x, mouse_y;
    SDL_GetMouseState(&mouse_x, &mouse_y);
    glm::vec2 mouse_world_pos = {mouse_x / render_scale, mouse_y / render_scale};

    constexpr float TOOL_RADIUS = 8.0f;
    constexpr float DT = 2.5f / 165.0f;

    // grab
    if (mouse_left_down)
    {
      particles.grab(mouse_world_pos, TOOL_RADIUS, 800.0f * DT);
    }
    else if (mouse_right_down)
    {
      particles.grab(mouse_world_pos, TOOL_RADIUS, -1000.0f * DT);
    }

    // spin
    if (mouse_middle_down)
    {
      particles.spin(mouse_world_pos, TOOL_RADIUS / 1.5f, 100.0f * DT);
    }

    // particles.update(0.3f / 165.0f);
    particles.update_rk4(DT);
    render(particles, renderer, render_scale);

    SDL_RenderPresent(renderer);
  }

  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);

  SDL_Quit();

  return 0;
}
