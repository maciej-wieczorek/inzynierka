#pragma once
#include <iostream>
#include <memory>

#include "ConnectionContent.h"
#include "Graph.h"

template <int NumGroups>
struct Group
{
    unsigned int minSize;
    unsigned int maxSize;
    unsigned int count{};
    size_t sum;
    timespec latestTimestamp{};
    float delaysSum[NumGroups]{};
    unsigned int delaysCount[NumGroups]{};
};

class Grouper
{
public:
    std::unique_ptr<Graph> group(ConnectionContent& connection);
    static constexpr unsigned int numGroups = 30;

private:
    void initGroups();
    int getGroupIndex(unsigned int size);
    Group<numGroups> groups[numGroups];
};

Grouper& getGrouper();

