#include "SDL.h"
#include "glm/glm.hpp"

#include "particle_system.h"
#include "particle_render.h"
#include "soil.h"
#include "tools.h"

#include <iostream>

int main(int argc, char *argv[])
{
  std::cout << "program start" << std::endl;
  SDL_Window *window;
  SDL_Renderer *renderer;
  SDL_Surface *surface;
  SDL_Event event;
  constexpr float KERNEL_RADIUS = 2.0f;
  glm::vec2 bounds = {64.0f * 16.0f / 9.0f, 64.0f};
  ParticleSystem particles({64.0f * 16.0f / 9.0f, 48.0f}, bounds, 7000, 1.0f, 0.0f, KERNEL_RADIUS);
  Soil soil(bounds, KERNEL_RADIUS);
  Tools tools(soil, particles);

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

    SDL_SetRenderDrawColor(renderer, 0x00, 0x00, 0x00, 0x00);
    SDL_RenderClear(renderer);

    // calculate render_scale so that the bounds fit the screen
    int screen_width, screen_height;
    SDL_GetWindowSize(window, &screen_width, &screen_height);
    auto bounds = particles.get_bounds();
    float render_scale = std::min(screen_width / bounds.x, screen_height / bounds.y);

    constexpr float DT = 1.0f / 165.0f;
    tools.update(event, render_scale, DT);
    // particles.update(0.3f / 165.0f);
    particles.update_rk4(DT);
    render_soil(soil, renderer, render_scale);
    render_velocity(particles, renderer, render_scale);
    tools.render(renderer, render_scale);

    SDL_RenderPresent(renderer);
  }

  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);

  SDL_Quit();

  return 0;
}
