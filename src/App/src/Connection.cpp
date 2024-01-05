#include "Connection.h"
#include "Packet.h"

Connection::Connection(pcpp::IPv4Address clientIP, uint16_t clientPort,
    pcpp::IPv4Address serverIP, uint16_t serverPort)

    : m_clinetIP{ clientIP }, m_clientPort{ clientPort },
    m_serverIP{ serverIP }, m_serverPort{ serverPort }
{
}

void Connection::reset()
{
    m_content = ConnectionContent{};
}

void Connection::addPacket(pcpp::Packet&& packet)
{
	pcpp::RawPacket* rawPacket = packet.getRawPacket();
	int len = rawPacket->getRawDataLen();
    timespec timestamp = rawPacket->getPacketTimeStamp();

    auto packetData = std::make_unique<char[]>(len);
    memcpy(packetData.get(), rawPacket->getRawData(), len);

    m_content.addEntry(timestamp, len, std::move(packetData));
}

GraphTensorData Connection::getAsGraph() const
{
    std::unique_ptr<Graph> graph = getGrouper().group(m_content);

    return graph->getAsTensors();
}

ConnectionContent& Connection::getContent()
{
    return m_content;
}

const ConnectionContent& Connection::getContent() const
{
    return m_content;
}

