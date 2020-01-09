/**
  @author: Rafał Skinderowicz (rafal.skinderowicz@us.edu.pl)
*/

#include <fstream>

#include "utils.h"
#include "json.hpp"


static nlohmann::json &get_best_known_solutions() {
    static nlohmann::json solutions;
    return solutions;
}


void load_best_known_solutions(const std::string &path) {
    std::ifstream in(path);
    auto &db = get_best_known_solutions();

    if (in.is_open()) {
        in >> db;
        in.close();
    }
}


double get_best_known_value(const std::string &instance_name,
                            double default_value = 0.0) {
    const auto &db = get_best_known_solutions();
    auto it = db.find(instance_name);
    return it != db.end() ? it.value().get<double>() : default_value;
}