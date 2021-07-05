#include <imgui.h>
#include <imgui-SFML.h>
#include <implot.h>

#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Text.hpp>
#include <SFML/System/Clock.hpp>
#include <SFML/Window/Event.hpp>

#include "../global_typedef.h"
#include "../util/util.h"
#include <list>


#define WINDOW_WIDTH  (980)
#define WINDOW_HEIGHT (680)

#define GRID_LINE_COLOR sf::Color::Color(0, 0, 0, 170)
#define WINDOW_BG_COLOR sf::Color::Color(247, 247, 247)

struct WorldConfig
{
  float grid_spacing = 25.0F;     // sfml units
  float tick_len = 4.0F;          // sfml units
  float boundary_padding = 35.0F; // sfml units
  float scaling = 0.65F;          // sfml units
};

struct GlobalData
{
  sf::RenderWindow      window;
  std::array<float, 2>  world2sfml_factor; 
  std::array<float, 2>  world_origin;
  std::array<float, 2>  world_bounds;
  std::vector<std::array<sf::Vertex, 2>> grid_ticks;
  std::vector<sf::Text> grid_tick_text;
  sf::Font font;

  std::list<sf::CircleShape> tasks;
  sf::CircleShape robot_position;

  GlobalData():window(sf::RenderWindow(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "Robot Tasker")){}
};

static WorldConfig world_config;
static GlobalData global_data;

static tuple<float, float> world2sfml(const float x, const float y)
{
  float x_sfml = world_config.scaling*x*global_data.world2sfml_factor[0]; 
  float y_sfml = world_config.scaling*y*global_data.world2sfml_factor[1];

  // transform with respect to required origin
  x_sfml = x_sfml + global_data.world_origin[0];
  y_sfml = global_data.world_origin[1] - y_sfml;

  return std::make_tuple(x_sfml, y_sfml);
}


std::string to_string_with_precision(const float value, const int precision = 1)
{
  std::ostringstream out;
  out.precision(precision);
  out << std::fixed << value;
  return out.str();
}

static
tuple<std::vector<std::array<sf::Vertex, 2>>, std::vector<sf::Text>>
generate_grid_data(const float world_max_x_sf, 
                   const float world_max_y_sf)
{
  vector<array<sf::Vertex, 2>> grid_ticks;
  vector<sf::Text>      tick_text;
 
  // generate x-grid lines
  int n_ticks_x = (int)std::ceil(world_max_x_sf/world_config.grid_spacing);
  grid_ticks.reserve(n_ticks_x);
  for (float x_coord = global_data.world_origin[0]; x_coord < world_max_x_sf; x_coord += world_config.grid_spacing)
  {
    const float& x1 = x_coord;
    const float& y1 = global_data.world_origin[1];
    const float& x2 = x_coord;
    const float y2  = y1 + world_config.tick_len;

    grid_ticks.push_back(array<sf::Vertex, 2>{sf::Vertex(sf::Vector2f{x1, y1}), 
                                              sf::Vertex(sf::Vector2f{x2, y2})});
    grid_ticks.back()[0].color = GRID_LINE_COLOR;
    grid_ticks.back()[1].color = GRID_LINE_COLOR;
  }

  // generate y-grid lines
  int n_ticks_y = (int)std::ceil(world_max_y_sf/world_config.grid_spacing);
  grid_ticks.reserve(n_ticks_y);
  for (float y_coord = world_config.boundary_padding; y_coord < world_max_y_sf; y_coord += world_config.grid_spacing)
  {
    const float& x1 = global_data.world_origin[0];
    const float& y1 = y_coord;
    const float  x2 = x1 - world_config.tick_len;
    const float& y2 = y_coord;  

    grid_ticks.push_back(array<sf::Vertex, 2>{sf::Vertex(sf::Vector2f{x1, y1}), 
                                              sf::Vertex(sf::Vector2f{x2, y2})});
    grid_ticks.back()[0].color = GRID_LINE_COLOR;
    grid_ticks.back()[1].color = GRID_LINE_COLOR;
  }

  // draw world bounds
  grid_ticks.push_back(array<sf::Vertex, 2>{sf::Vertex(sf::Vector2f{global_data.world_origin[0], 
                                                                       global_data.world_origin[1]}), 
                                            sf::Vertex(sf::Vector2f{global_data.world_origin[0] + world_max_x_sf, 
                                                                    global_data.world_origin[1]})});
  grid_ticks.back()[0].color = GRID_LINE_COLOR;
  grid_ticks.back()[1].color = GRID_LINE_COLOR;
  
  // axis max tick text
  tick_text.push_back(sf::Text());
  sf::Text* this_text = &tick_text.back();
  this_text->setPosition(global_data.world_origin[0] + world_max_x_sf, global_data.world_origin[1] + world_config.tick_len + 10.0F);
  this_text->setString("X-" + to_string_with_precision(global_data.world_bounds[0], 0) + " (m)");
  this_text->setFillColor(sf::Color::Black);
  this_text->setStyle(sf::Text::Bold);
  this_text->setCharacterSize(12);
  this_text->setFont(global_data.font);
  
  grid_ticks.push_back(array<sf::Vertex, 2>{sf::Vertex(sf::Vector2f{global_data.world_origin[0], 
                                                                    global_data.world_origin[1]}), 
                                            sf::Vertex(sf::Vector2f{global_data.world_origin[0], 
                                                                    world_config.boundary_padding})});                                                               
  grid_ticks.back()[0].color = GRID_LINE_COLOR;
  grid_ticks.back()[1].color = GRID_LINE_COLOR;
  
  // axis max tick text
  tick_text.push_back(sf::Text());
  this_text = &tick_text.back();
  this_text->setPosition(global_data.world_origin[0] - world_config.tick_len - 10.0F, world_config.boundary_padding-10.0F);
  this_text->setString("Y-" + to_string_with_precision(global_data.world_bounds[1], 0) + " (m)");
  this_text->setFillColor(sf::Color::Black);
  this_text->setStyle(sf::Text::Bold);
  this_text->setCharacterSize(12);
  this_text->setFont(global_data.font);

  grid_ticks.shrink_to_fit();

  return std::make_tuple(grid_ticks, tick_text);
}


void initialize_world(const std::string& config)
{
  auto global_config = env_util::read_global_config(config);
  const float world_max_x = global_config.at("world/size/x"); 
  const float world_max_y = global_config.at("world/size/y"); 

  global_data.window.setFramerateLimit(60);
  ImGui::SFML::Init(global_data.window);
  ImPlot::CreateContext();

  global_data.world2sfml_factor[0] = WINDOW_WIDTH/world_max_x;
  global_data.world2sfml_factor[1] = WINDOW_HEIGHT/world_max_y;
  global_data.world_bounds[0]      = world_max_x;
  global_data.world_bounds[1]      = world_max_y;

  global_data.world_origin[0] = world_config.boundary_padding;
  global_data.world_origin[1] = (world_config.scaling*WINDOW_HEIGHT)+world_config.boundary_padding;

  tie(global_data.grid_ticks, global_data.grid_tick_text) = generate_grid_data(world_config.scaling*WINDOW_WIDTH + global_data.world_origin[0], 
                                                                               world_config.scaling*WINDOW_HEIGHT + world_config.boundary_padding);
  // create marker for robot position
  auto [x_ini_pos, y_ini_pos] = world2sfml(50.0F, 50.0F);
  global_data.robot_position = sf::CircleShape{5, 3}; // triangle
  global_data.robot_position.setFillColor(sf::Color::Magenta);
  global_data.robot_position.setPosition(sf::Vector2f{x_ini_pos, y_ini_pos});

  auto this_file_folder = util::get_file_dir_path(__FILE__);
  if (!global_data.font.loadFromFile(this_file_folder + "/font.ttf"))
  {
    std::cout << "Font load failed\n";
  }
}


void run_app()
{
  auto& main_window = global_data.window;
  sf::Clock delta_clock;

  while(main_window.isOpen())
  {
    main_window.clear(WINDOW_BG_COLOR);
    sf::Event event;
    while(main_window.pollEvent(event))
    {
      ImGui::SFML::ProcessEvent(event);
      if (event.type == sf::Event::Closed)
      {
        main_window.close();
      }
    }
    ImGui::SFML::Update(main_window, delta_clock.restart());
    
    ImGui::Begin("RobotPosition");
    static float robot_pose_x = 50.0F, robot_pose_y = 50.0F;
    ImGui::SliderFloat("Pose-X", &robot_pose_x, 0.0F, 200.0F);
    ImGui::SliderFloat("Pose-Y", &robot_pose_y, 0.0F, 300.0F);
    ImGui::End();

    ImGui::Begin("World");
    ImGui::SliderFloat("Scaling", &world_config.scaling, 0.25F, 0.85F);
    ImGui::End();

    global_data.world_origin[0] = world_config.boundary_padding;
    global_data.world_origin[1] = (world_config.scaling*WINDOW_HEIGHT)+world_config.boundary_padding;
    float world_max_x = world_config.scaling*WINDOW_WIDTH  + global_data.world_origin[0];
    float world_max_y = world_config.scaling*WINDOW_HEIGHT + world_config.boundary_padding;

    // draw robot position
    auto [x_sfml, y_sfml] = world2sfml(robot_pose_x, robot_pose_y);
    global_data.robot_position.setPosition(x_sfml, y_sfml);
    main_window.draw(global_data.robot_position);

    // draw grid lines
    tie(global_data.grid_ticks, global_data.grid_tick_text) = generate_grid_data(world_max_x, world_max_y);
    std::for_each(global_data.grid_ticks.begin(), global_data.grid_ticks.end(), 
                  [&main_window](auto& vertices) {main_window.draw(vertices.data(), 2, sf::Lines);} );
    std::for_each(global_data.grid_tick_text.begin(), global_data.grid_tick_text.end(), 
                  [&main_window](auto& text) {main_window.draw(text); });

    ImGui::SFML::Render(main_window);
    main_window.display();
  }
  ImPlot::DestroyContext();
}


int main()
{
  auto config_full_path = util::get_file_dir_path(__FILE__) + "/" + "../../global_params.yaml";
  initialize_world(config_full_path);
  run_app();

  return EXIT_SUCCESS;
}