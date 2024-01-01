#include "Defines.h"

#include "Splitter.h"
#include "Utilities.h"

#include "pcapplusplus/Packet.h"
#include "pcapplusplus/IPv4Layer.h"
#include "pcapplusplus/TcpLayer.h"
#include "pcapplusplus/UdpLayer.h"

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

void Splitter::consumePacket(const pcpp::Packet& packet)
{
    // verify the packet is IPv4
    if (packet.isPacketOfType(pcpp::IPv4))
    {
        uint16_t srcPort{ 0 }, dstPort{ 0 };

        // extract ports
        if (packet.isPacketOfType(pcpp::TCP))
        {
            pcpp::TcpLayer* TcpLayer = packet.getLayerOfType<pcpp::TcpLayer>();
            srcPort = TcpLayer->getSrcPort();
            dstPort = TcpLayer->getDstPort();
        }
        else if (packet.isPacketOfType(pcpp::UDP))
        {
            pcpp::UdpLayer* UdpLayer = packet.getLayerOfType<pcpp::UdpLayer>();
            srcPort = UdpLayer->getSrcPort();
            dstPort = UdpLayer->getDstPort();
        }

        // extract IPs
        if (dstPort != 0 && srcPort != 0)
        {
            pcpp::IPv4Address srcIP = packet.getLayerOfType<pcpp::IPv4Layer>()->getSrcIPv4Address();
            pcpp::IPv4Address dstIP = packet.getLayerOfType<pcpp::IPv4Layer>()->getDstIPv4Address();

            pcpp::RawPacket* rawPacket = packet.getRawPacket();
            int packetLen = rawPacket->getRawDataLen();

            // add packet to connections
            if (srcPort < dstPort)
            {
                addPacket(dstIP, dstPort, srcIP, srcPort, rawPacket->getPacketTimeStamp(), packetLen);
            }
            else
            {
                addPacket(srcIP, srcPort, dstIP, dstPort, rawPacket->getPacketTimeStamp(), packetLen);
            }
        }
    }
}

void Splitter::addPacket(pcpp::IPv4Address clientIP, uint16_t clientPort, pcpp::IPv4Address serverIP, uint16_t serverPort, timespec timestamp, int len)
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
    else if (conn.getLastTimestamp() - conn.getFirstTimestamp() >= Splitter::maxTimeLenConnection * 1000000000)
    {
        return true;
    }

    return false;
}

