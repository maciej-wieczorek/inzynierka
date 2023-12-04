#pragma once

#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

#include "Connection.h"

class Splitter
{
public:
    static constexpr size_t minSizeConnection = 100;
    static constexpr size_t maxSizeConnection = 1000;
    static constexpr size_t maxTotalByteSizeConnection = 1024 * 1024; // 1MB
    static constexpr size_t maxTimeLenConnection = 60; // 60s

    static constexpr const char* graphsFilename = "graphs.csv";
    static constexpr const char* csvHeader = "client,server,graph";

    Splitter();
    ~Splitter();
    void add_packet(pcpp::IPv4Address clientIP, uint16_t clientPort,
        pcpp::IPv4Address serverIP, uint16_t serverPort, timespec timestamp, int len);
    static bool shouldReset(const Connection& conn);

private:
    std::map<std::string, Connection> m_connections;
    std::ofstream m_graphsFile;
};

