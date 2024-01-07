#include "Utilities.h"
#include <algorithm>
#include <iostream>

float operator-(timespec ts1, timespec ts2)
{
    time_t secDifPart = ts1.tv_sec - ts2.tv_sec;
    long nsecDifPart = ts1.tv_nsec - ts2.tv_nsec;
    float nsDif = static_cast<float>((secDifPart * 1000000000) + nsecDifPart);
    return nsDif;
}

bool caseInsensitiveCompare(char c1, char c2)
{
    return std::tolower(c1) == std::tolower(c2);
}

bool containsCaseInsensitive(const std::string& text, const std::string& searchTerm)
{
    auto it = std::search(
        text.begin(), text.end(),
        searchTerm.begin(), searchTerm.end(),
        caseInsensitiveCompare
    );

    return it != text.end();
}

void z_scale(std::vector<float>& arr)
{
    // Calculate mean
    float sum = 0.0;
    for (float value : arr)
    {
        sum += value;
    }
    float mean = sum / arr.size();

    // Calculate standard deviation
    float sum_squared_diff = 0.0;
    for (float value : arr)
    {
        float diff = value - mean;
        sum_squared_diff += diff * diff;
    }

    float std_dev = sqrt(sum_squared_diff / arr.size()) + 0.00001f;

    // Z-scale the vector
    for (size_t i = 0; i < arr.size(); ++i)
    {
        float z_score = (arr[i] - mean) / std_dev;
        arr[i] = z_score;
    }
}
