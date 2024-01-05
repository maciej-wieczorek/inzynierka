#pragma once

#include "Graph.h"
#include "PcapFileDevice.h"
#include "Grouper.h"

#include <vector>

class Connection
{
public:
    Connection(pcpp::IPv4Address clientIP, uint16_t clientPort, pcpp::IPv4Address serverIP, uint16_t serverPort);
    void reset();
    void addPacket(pcpp::Packet&& packet);


    GraphTensorData getAsGraph() const;
    ConnectionContent& getContent();
    const ConnectionContent& getContent() const;

    pcpp::IPv4Address m_clinetIP;
    uint16_t m_clientPort;
    pcpp::IPv4Address m_serverIP;
    uint16_t m_serverPort;
    ConnectionContent m_content;
};


