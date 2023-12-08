#include "Defines.h"

#include "Connection.h"

Connection::Connection(pcpp::IPv4Address clientIP, uint16_t clientPort,
    pcpp::IPv4Address serverIP, uint16_t serverPort)

    : m_clinetIP{ clientIP }, m_clientPort{ clientPort },
    m_serverIP{ serverIP }, m_serverPort{ serverPort }
{
}

void Connection::reset()
{
    m_content.clear();
    m_totalSizePackets = 0;
}

void Connection::addPacket(timespec timestamp, int len)
{
    m_content.emplace_back(timestamp, len);
    m_totalSizePackets += len;
}

size_t Connection::getCountPackets() const
{
    return m_content.size();
}

size_t Connection::getTotalSizePackets() const
{
    return m_totalSizePackets;
}

timespec Connection::getFirstTimestamp() const
{
    return m_content[0].timestamp;
}

timespec Connection::getLastTimestamp() const
{
    return m_content[m_content.size() - 1].timestamp;
}

void Connection::save(std::ostream& stream, const char* dataSource)
{
    std::unique_ptr<Graph> graph = getGrouper().group(m_content);
    // graphs can be empty when all packets were of same group size
    if (graph->edgeList.size() > 0)
    {
        stream << m_clinetIP << ':' << m_clientPort << ',' << m_serverIP << ':' << m_serverPort << ',' << dataSource << ",\"" << *graph << "\"\n";
    }
}
