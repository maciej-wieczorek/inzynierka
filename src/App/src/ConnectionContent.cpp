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

void ConnectionContent::addEntry(timespec ts, int s, int ps, std::unique_ptr<uint8_t[]>&& packetData)
{
    entries.emplace_back(ts, s, ps, std::forward<std::unique_ptr<uint8_t[]>>(packetData));
    m_totalSizePackets += s;
}
