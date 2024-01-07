#pragma once

#include <time.h>
#include <string>
#include <vector>

float operator-(timespec ts1, timespec ts2);
bool containsCaseInsensitive(const std::string& text, const std::string& searchTerm);
void z_scale(std::vector<float>& arr);

