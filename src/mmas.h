#pragma once

#include "docopt.h"

/**
 * Runs the MMAS based on the command line arguments passed in args
 */
void run_mmas_experiment(std::map<std::string, docopt::value> &args);