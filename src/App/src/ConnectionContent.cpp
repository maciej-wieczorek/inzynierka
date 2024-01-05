#include "ConnectionContent.h"


size_t ConnectionContent::getCountPackets() const
{
    return entries.size();
}

size_t ConnectionContent::getTotalSizePackets() const
{
    return m_totalSizePackets;
}

timespec ConnectionContent::getFirstTimestamp() const
{
    return entries[0].timestamp;
}

timespec ConnectionContent::getLastTimestamp() const
{
    return entries[entries.size() - 1].timestamp;
}

void ConnectionContent::addEntry(timespec ts, int s, std::unique_ptr<char[]>&& packetData)
{
    entries.emplace_back(ts, s, std::forward<std::unique_ptr<char[]>>(packetData));
    m_totalSizePackets += s;
}
