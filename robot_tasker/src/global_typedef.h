#ifndef _GLOBAL_TYPEDEF_H_
#define _GLOBAL_TYPEDEF_H_

#include <vector>
#include <array>
#include <tuple>
#include <string>
#include <string_view>
#include <charconv>
#include <unordered_map>
#include <any>

#include <fstream>
#include <chrono>

#include <algorithm>
#include <numeric>

#pragma warning(push, 0)        
// includes with unfixable warnings
#include <Eigen/Core>
#include <Eigen/Dense>
#include "yaml-cpp/yaml.h"
#pragma warning(pop)

using std::vector, std::string, std::size_t, std::array, std::fstream, std::tuple, std::tie;
using std::for_each, std::copy, std::transform;
using Eigen::all, Eigen::seq, Eigen::last, Eigen::NoChange;

namespace eig    = Eigen;
namespace chrono = std::chrono;
namespace yml    = YAML;
using namespace std::string_literals;
using namespace std::chrono_literals;

#define ITER(X) std::begin(X), std::end(X)
#define TIME_NOW chrono::system_clock::now()
#define TIME_DIFF(end_time, start_time) (std::chrono::duration<float>(end_time - start_time)).count()
#define str2float(str, result) std::from_chars(str.data(), str.data()+str.length(), result)
#define INF 100000.0F
#define UNUSED(X) (void)(X)
#define NOT(X) (!(X))
#define PI (3.14159265F)
#define TWO_PI (2.0F*PI)
#define PI_2 (PI/2.0F)
#define deg2rad(x) ((x)*0.017453292519943F)
#define rad2deg(x) ((x)*57.2957795130823208F)

namespace global_typedef {

using global_config_t = std::unordered_map<std::string, float>;

} // namespace {global_typedef}

#endif
