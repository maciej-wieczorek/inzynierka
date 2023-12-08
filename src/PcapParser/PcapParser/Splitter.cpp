#include "Defines.h"

#include "Splitter.h"
#include "Utilities.h"

Splitter::Splitter(const char* dataSource)
    : m_dataSource{ dataSource }
{
    if (std::filesystem::exists(graphsFilename))
    {
        m_graphsFile = std::ofstream{ graphsFilename, std::ios::app }; // open for appending
    }
    else
    {
        m_graphsFile = std::ofstream{ graphsFilename };
        m_graphsFile << csvHeader << '\n';
    }
}

Splitter::~Splitter()
{
    // Dump remaining connections
    for (auto& find : m_connections)
    {
        if (find.second.getCountPackets() >= Splitter::minSizeConnection)
        {
            find.second.save(m_graphsFile, m_dataSource.c_str());
        }
    }
}

void Splitter::add_packet(pcpp::IPv4Address clientIP, uint16_t clientPort, pcpp::IPv4Address serverIP, uint16_t serverPort, timespec timestamp, int len)
{
    std::stringstream key;
    key << clientIP << ':' << clientPort << '-' << serverIP << ':' << serverPort;
    const auto& find = m_connections.find(key.str());
    if (find == m_connections.end())
    {
        Connection connection{clientIP, clientPort, serverIP, serverPort};
        connection.addPacket(timestamp, len);

        m_connections.insert(std::pair<std::string, Connection>{ key.str(), std::move(connection) });
    }
    else
    {
        if (shouldReset(find->second))
        {
            find->second.save(m_graphsFile, m_dataSource.c_str());
            find->second.reset();
        }

        find->second.addPacket(timestamp, len);
    }
}

bool Splitter::shouldReset(const Connection& conn)
{
    if (conn.getCountPackets() < Splitter::minSizeConnection)
    {
        return false;
    }

    if (conn.getCountPackets() >= Splitter::maxSizeConnection)
    {
        return true;
    }
    else if (conn.getTotalSizePackets() >= Splitter::maxTotalByteSizeConnection)
    {
        return true;
    }
    else if (conn.getLastTimestamp() - conn.getFirstTimestamp() >= Splitter::maxTimeLenConnection * 1000)
    {
        return true;
    }

    return false;
}

