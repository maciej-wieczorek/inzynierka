#pragma once

#include "Graph.h"
#include "PcapFileDevice.h"
#include "Grouper.h"

#include <vector>

struct ConnectionInfo
{
    std::string clientIP;
    uint16_t clienPort;
    std::string serverIP;
    uint16_t serverPort;
    std::string dataSource;
    size_t countPackets;
    size_t totalSizePackets;
    timespec firstTimestamp;
    timespec lastTimestamp;

};

std::ostream& operator<<(std::ostream& os, const timespec& obj);
std::ostream& operator<<(std::ostream& os, const ConnectionInfo& obj);

class Connection
{
public:
    Connection(const char* dataSource, pcpp::IPv4Address clientIP, uint16_t clientPort, pcpp::IPv4Address serverIP, uint16_t serverPort);
    void reset();
    void addPacket(pcpp::Packet&& packet);


    GraphTensorData getAsGraph() const;
    ConnectionContent& getContent();
    const ConnectionContent& getContent() const;
    ConnectionInfo getConnectionInfo() const;

    std::string m_dataSource;
    pcpp::IPv4Address m_clientIP;
    uint16_t m_clientPort;
    pcpp::IPv4Address m_serverIP;
    uint16_t m_serverPort;
    ConnectionContent m_content;
};


