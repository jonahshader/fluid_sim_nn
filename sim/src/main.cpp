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

  ParticleSystem particles({512.0f, 720.0f}, {1280.0f, 720.0f}, 3000, 1.0f, 0.0f, 24.0f);

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

    // grab
    if (mouse_left_down)
    {
      int x, y;
      SDL_GetMouseState(&x, &y);
      particles.grab({(float)x, (float)y}, 128.0f, 10.0f);
    }
    else if (mouse_right_down)
    {
      int x, y;
      SDL_GetMouseState(&x, &y);
      particles.grab({(float)x, (float)y}, 128.0f, -10.0f);
    }

    // spin
    if (mouse_middle_down)
    {
      int x, y;
      SDL_GetMouseState(&x, &y);
      particles.spin({(float)x, (float)y}, 256.0f, 3.0f);
    }

    particles.update(1 / 165.0f);
    render(particles, renderer);

    SDL_RenderPresent(renderer);
  }

  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);

  SDL_Quit();

  return 0;
}
