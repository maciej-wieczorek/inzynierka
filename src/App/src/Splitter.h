#pragma once

#include "Connection.h"

#include <Packet.h>

#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

class App;

class Splitter
{
public:

    Splitter(App* app) : m_app{ app } {};
    ~Splitter();
    void consumePacket(pcpp::Packet&& packet);
    void addPacket(pcpp::IPv4Address clientIP, uint16_t clientPort,
        pcpp::IPv4Address serverIP, uint16_t serverPort, pcpp::Packet&& packet);

private:
    std::map<std::string, Connection> m_connections;
    App* m_app;
};

