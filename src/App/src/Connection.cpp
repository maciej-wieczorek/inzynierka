#include "Connection.h"
#include "Packet.h"

Connection::Connection(const char* dataSource, pcpp::IPv4Address clientIP, uint16_t clientPort,
    pcpp::IPv4Address serverIP, uint16_t serverPort)

    : m_dataSource{ dataSource },
    m_clientIP{ clientIP }, m_clientPort{ clientPort },
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

ConnectionInfo Connection::getConnectionInfo() const
{
    return ConnectionInfo
    {
		m_clientIP.toString(),
		m_clientPort,
		m_serverIP.toString(),
		m_serverPort,
		m_dataSource,
		m_content.getCountPackets(),
		m_content.getTotalSizePackets(),
		m_content.getFirstTimestamp(),
		m_content.getLastTimestamp()
    };
}

std::ostream& operator<<(std::ostream& os, const timespec& obj)
{
    os << obj.tv_sec << '.' << obj.tv_nsec;
    return os;
}

std::ostream& operator<<(std::ostream& os, const ConnectionInfo& obj)
{
    os << obj.clientIP << ',' << obj.clienPort << ',' << obj.serverIP << ','
       << obj.serverPort << ',' << obj.dataSource << ',' << obj.countPackets << ','
       << obj.totalSizePackets << ',' << obj.firstTimestamp << ',' << obj.lastTimestamp;
    return os;
}
