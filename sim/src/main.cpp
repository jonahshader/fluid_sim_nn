#include "SDL.h"
#include "glm/glm.hpp"

#include "particle_system.h"
#include "particle_render.h"
#include "soil.h"
#include "tools.h"

#include <iostream>

const glm::vec2 bounds = {128.0f, 128.0f};
const glm::vec2 spawn = {64.0f, 64.0f};

constexpr float KERNEL_RADIUS = 2.0f;
constexpr int PARTICLES = 7000;
constexpr float PARTICLE_MASS = 1.0f;

void init_sim(ParticleSystem &particles, Soil &soil)
{
  particles = ParticleSystem(spawn, bounds, PARTICLES, PARTICLE_MASS, 0.0f, KERNEL_RADIUS);
  soil = Soil(bounds, KERNEL_RADIUS);
}

int main(int argc, char *argv[])
{
  std::cout << "program start" << std::endl;
  SDL_Window *window;
  SDL_Renderer *renderer;
  SDL_Surface *surface;
  SDL_Event event;
  ParticleSystem particles(spawn, bounds, PARTICLES, PARTICLE_MASS, 0.0f, KERNEL_RADIUS);
  Soil soil(bounds, KERNEL_RADIUS);
  Tools tools(soil, particles);

  bool paused = false;

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
    if (event.type == SDL_KEYDOWN)
    {
      if (event.key.keysym.sym == SDLK_SPACE)
      {
        paused = !paused;
      }
      else if (event.key.keysym.sym == SDLK_r)
      {
        init_sim(particles, soil);
      }
    }

    SDL_SetRenderDrawColor(renderer, 0x00, 0x00, 0x00, 0x00);
    SDL_RenderClear(renderer);

    // calculate render_scale so that the bounds fit the screen
    int screen_width, screen_height;
    SDL_GetWindowSize(window, &screen_width, &screen_height);
    auto bounds = particles.get_bounds();
    float render_scale = std::min(screen_width / bounds.x, screen_height / bounds.y);

    constexpr float DT = 0.25f / 165.0f;
    tools.update(event, render_scale, DT);
    if (!paused)
    {
      // particles.update(0.3f / 165.0f);
      particles.update_rk4(soil, DT);
    }

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
