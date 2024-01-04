#pragma once

#include <time.h>
#include <vector>

struct ConnectionEntry
{
    ConnectionEntry(timespec ts, int s);
    timespec timestamp;
    int size;
};

using ConnectionContent = std::vector<ConnectionEntry>; 

