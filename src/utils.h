#pragma once
/**
  @author: Rafał Skinderowicz (rafal.skinderowicz@us.edu.pl)
*/

#include <string>

/**
 * Loads a map of the best-known or optimal solution values from the
 * JSON file at the given path.
 */
void load_best_known_solutions(const std::string &path);


/**
 * Returns the best-known or optimal solution value for the TSP
 * instance with the given name or default_value if the value is not
 * known.
 */
double get_best_known_value(const std::string &instance_name,
                            double default_value);