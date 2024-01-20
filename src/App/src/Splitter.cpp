#include "App.h"
#include "Splitter.h"
#include "Utilities.h"

#include <Packet.h>
#include <EthLayer.h>
#include <IPv4Layer.h>
#include <TcpLayer.h>
#include <UdpLayer.h>

Splitter::~Splitter()
{
    // Handle remaining connections
    for (auto& it : m_connections)
    {
		Connection& connection = it.second;
		if (getGrouper().canGroup(connection.getContent()))
		{
            m_app->processConnection(connection);
		}
		connection.reset();
    }
}

void Splitter::consumePacket(pcpp::Packet&& packet)
{
    // verify the packet is IPv4
    if (packet.isPacketOfType(pcpp::IPv4))
    {
        uint16_t srcPort{ 0 }, dstPort{ 0 };

        // extract ports and mask them
        if (packet.isPacketOfType(pcpp::TCP))
        {
            pcpp::TcpLayer* TcpLayer = packet.getLayerOfType<pcpp::TcpLayer>();
            srcPort = TcpLayer->getSrcPort();
            dstPort = TcpLayer->getDstPort();
            TcpLayer->getTcpHeader()->portSrc = 0;
            TcpLayer->getTcpHeader()->portDst = 0;
        }
        else if (packet.isPacketOfType(pcpp::UDP))
        {
            pcpp::UdpLayer* UdpLayer = packet.getLayerOfType<pcpp::UdpLayer>();
            srcPort = UdpLayer->getSrcPort();
            dstPort = UdpLayer->getDstPort();
            UdpLayer->getUdpHeader()->portSrc = 0;
            UdpLayer->getUdpHeader()->portDst = 0;
        }

        // extract IPs and mask them
        if (dstPort != 0 && srcPort != 0)
        {
            pcpp::IPv4Layer* IPv4Layer = packet.getLayerOfType<pcpp::IPv4Layer>();
            pcpp::IPv4Address srcIP = IPv4Layer->getSrcIPv4Address();
            pcpp::IPv4Address dstIP = IPv4Layer->getDstIPv4Address();
            IPv4Layer->setDstIPv4Address(pcpp::IPv4Address{});
            IPv4Layer->setSrcIPv4Address(pcpp::IPv4Address{});

            // mask link layer
            auto ethLayer = packet.getLayerOfType<pcpp::EthLayer>();
            if (ethLayer)
            {
                size_t linkLayerHeaderSize = ethLayer->getHeaderLen();
                memset(ethLayer->getEthHeader(), 0, linkLayerHeaderSize);
            }

            // add packet to connections
            if (srcPort < dstPort)
            {
                addPacket(dstIP, dstPort, srcIP, srcPort, std::forward<pcpp::Packet>(packet));
            }
            else
            {
                addPacket(srcIP, srcPort, dstIP, dstPort, std::forward<pcpp::Packet>(packet));
            }
        }
    }
}

void Splitter::addPacket(pcpp::IPv4Address clientIP, uint16_t clientPort, pcpp::IPv4Address serverIP, uint16_t serverPort, pcpp::Packet&& packet)
{
    std::stringstream key;
    key << clientIP << ':' << clientPort << '-' << serverIP << ':' << serverPort;
    auto it = m_connections.find(key.str());
    if (it == m_connections.end())
    {
        Connection connection{getCurrentDataSource(), clientIP, clientPort, serverIP, serverPort};
        connection.addPacket(std::forward<pcpp::Packet>(packet));

        m_connections.emplace(key.str(), std::move(connection));
    }
    else
    {
        Connection& connection = it->second;
        if (getGrouper().shouldGroup(connection.getContent()))
        {
            m_app->processConnection(connection);

            connection.reset();
        }

        connection.addPacket(std::forward<pcpp::Packet>(packet));
    }
}

void Splitter::setCurrentDataSource(const char* dataSource)
{
    m_currentDataSource = dataSource;
}

const char* Splitter::getCurrentDataSource() const
{
    return m_currentDataSource.c_str();
}

