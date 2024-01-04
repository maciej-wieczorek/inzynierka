#pragma once

#include <vector>

#include "PcapFileDevice.h"

#include "Graph.h"
#include "Grouper.h"

class Connection
{
public:
    Connection(pcpp::IPv4Address clientIP, uint16_t clientPort, pcpp::IPv4Address serverIP, uint16_t serverPort);
    void reset();
    void addPacket(timespec timestamp, int len);

    size_t getCountPackets() const;
    size_t getTotalSizePackets() const;
    timespec getFirstTimestamp() const;
    timespec getLastTimestamp() const;

    void save(std::ostream& stream, const char* dataSource);

private:
    pcpp::IPv4Address m_clinetIP;
    uint16_t m_clientPort;
    pcpp::IPv4Address m_serverIP;
    uint16_t m_serverPort;
    ConnectionContent m_content;
    size_t m_totalSizePackets{};
};


