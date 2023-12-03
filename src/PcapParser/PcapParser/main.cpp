#define _SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <filesystem>

#include "pcapplusplus/Packet.h"
#include "pcapplusplus/PcapFileDevice.h"
#include "pcapplusplus/IPv4Layer.h"
#include "pcapplusplus/TcpLayer.h"
#include "pcapplusplus/UdpLayer.h"

#pragma comment(lib, "ws2_32.lib")

struct ConnectionEntry
{
    ConnectionEntry(timespec ts, int s) : timestamp{ ts }, size{ s } {}
    timespec timestamp;
    int size;
};

using ConnectionContent = std::vector<ConnectionEntry>;

struct Edge
{
    unsigned int indexStart;
    unsigned int indexEnd;
    float weight;
};

struct Graph
{
    std::vector<Edge> edgeList;

    friend std::ostream& operator<<(std::ostream& os, const Graph& graph)
    {
        size_t i = 0;
        for (const auto& edge : graph.edgeList)
        {
            os << edge.indexStart << ',' << edge.indexEnd << ',' << edge.weight;

            if (i != graph.edgeList.size() - 1)
            {
                os << ' ';
            }
            ++i;
        }

        return os;
    }
};

float operator-(timespec ts1, timespec ts2)
{
    time_t secDifPart = ts1.tv_sec - ts2.tv_sec;
    long nsecDifPart = ts1.tv_nsec - ts2.tv_nsec;
    float msDif = (secDifPart * 1000) + (nsecDifPart / 1000000);
    return msDif;
}

const unsigned int Grouper_numGroups = 30;

struct Group
{
    unsigned int minSize;
    unsigned int maxSize;
    unsigned int count{};
    size_t sum;
    timespec latestTimestamp{};
    float delaysSum[Grouper_numGroups]{};
    unsigned int delaysCount[Grouper_numGroups]{};
};

class Grouper
{
public:
    std::unique_ptr<Graph> group(ConnectionContent& connection)
    {
        initGroups();
        for (const auto& entry : connection)
        {
            int groupIndex = getGroupIndex(entry.size);
            groups[groupIndex].count += 1;
            groups[groupIndex].sum += entry.size;
            groups[groupIndex].latestTimestamp = entry.timestamp;

            // update delays to other groups
            for (unsigned int i = 0; i < numGroups; ++i)
            {
                if (i != groupIndex && groups[i].count > 0)
                {
                    float delay = entry.timestamp - groups[i].latestTimestamp;
                    groups[i].delaysSum[groupIndex] += delay;
                    groups[i].delaysCount[groupIndex] += 1;
                }
            }
        }

        std::unique_ptr<Graph> graph = std::unique_ptr<Graph>(new Graph);
        for (unsigned int i = 0; i < numGroups; ++i)
        {
            for (unsigned int j = 0; j < numGroups; ++j)
            {
                if (groups[i].delaysCount[j] > 0)
                {
                    float avgDelay = groups[i].delaysSum[j] / groups[i].delaysCount[j];
                    graph->edgeList.push_back(Edge{ i, j, avgDelay });
                }
            }
        }

        return std::move(graph);
    }
    
    static constexpr unsigned int numGroups = 30;

private:
    void initGroups()
    {
        unsigned int sizeStart = 40;
        unsigned int sizeEnd = 1520;
        unsigned int n = numGroups / 2;
        unsigned int inc = (sizeEnd - sizeStart) / n;
        unsigned int i = 0;

        // lower half
        for (; i < n; ++i)
        {
            groups[i] = Group{ sizeStart, sizeStart + inc };
            sizeStart += inc + 1;
        }

        sizeStart = sizeEnd + 1;
        sizeEnd = 65535;
        inc = (sizeEnd - sizeStart) / n;

        // upper half
        for (; i < numGroups; ++i)
        {
            groups[i] = Group{ sizeStart, sizeStart + inc };
            sizeStart += inc + 1;
        }
    }

    int getGroupIndex(unsigned int size)
    {
        int left = 0;
        int right = numGroups - 1;

        // bin search for group
        while (left <= right)
        {
            int mid = left + (right - left) / 2;

            if (size >= groups[mid].minSize && size <= groups[mid].maxSize)
            {
                return mid;
            }
            else if (size < groups[mid].minSize)
            {
                right = mid - 1;
            }
            else
            {
                left = mid + 1;
            }
        }

        std::cerr << "Could not find group index";
        throw std::exception{};
    }

    Group groups[numGroups];
};

Grouper& getGrouper()
{
    static Grouper grouper;
    return grouper;
}

class Connection
{
public:
    Connection(pcpp::IPv4Address clientIP, uint16_t clientPort, pcpp::IPv4Address serverIP, uint16_t serverPort)
        : m_clinetIP{ clientIP }, m_clientPort{ clientPort }, m_serverIP{ serverIP }, m_serverPort{ serverPort }
    {
    }

    void reset()
    {
        m_content.clear();
        m_totalSizePackets = 0;
    }

    void addPacket(timespec timestamp, int len)
    {
        m_content.emplace_back(timestamp, len);
        m_totalSizePackets += len;
    }

    size_t getCountPackets() const
    {
        return m_content.size();
    }

    size_t getTotalSizePackets() const
    {
        return m_totalSizePackets;
    }

    timespec getFirstTimestamp() const
    {
        return m_content[0].timestamp;
    }

    timespec getLastTimestamp() const
    {
        return m_content[m_content.size() - 1].timestamp;
    }

    void save(std::ostream& stream)
    {
        std::unique_ptr<Graph> graph = getGrouper().group(m_content);
        // graphs can be empty when all packets were of same group size
        if (graph->edgeList.size() > 0)
        {
            stream << m_clinetIP << ':' << m_clientPort << ',' << m_serverIP << ':' << m_serverPort << ",\"" << *graph << "\"\n";
        }
    }

private:
    pcpp::IPv4Address m_clinetIP;
    uint16_t m_clientPort;
    pcpp::IPv4Address m_serverIP;
    uint16_t m_serverPort;
    ConnectionContent m_content;
    size_t m_totalSizePackets{};
};

bool shouldReset(const Connection& conn);

class Splitter
{
public:
    static constexpr size_t minSizeConnection = 100;
    static constexpr size_t maxSizeConnection = 1000;
    static constexpr size_t maxTotalByteSizeConnection = 1024 * 1024; // 1MB
    static constexpr size_t maxTimeLenConnection = 60; // 60s

    static constexpr const char* graphsFilename = "graphs.csv";
    static constexpr const char* csvHeader = "client,server,graph";

    Splitter()
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

    ~Splitter()
    {
        // Dump remaining connections
        for (auto& find : m_connections)
        {
            if (find.second.getCountPackets() >= Splitter::minSizeConnection)
            {
                find.second.save(m_graphsFile);
            }
        }
    }

    void add_packet(pcpp::IPv4Address clientIP, uint16_t clientPort, pcpp::IPv4Address serverIP, uint16_t serverPort, timespec timestamp, int len)
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
                find->second.save(m_graphsFile);
                find->second.reset();
            }

            find->second.addPacket(timestamp, len);
        }
    }

private:
    std::map<std::string, Connection> m_connections;
    std::ofstream m_graphsFile;
};

Splitter splitter{};

bool shouldReset(const Connection& conn)
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

void readCaptureFile(const char* filepath)
{
    std::cout << "Reading: " << filepath << '\n';

   // open a pcap file for reading
    pcpp::PcapFileReaderDevice reader(filepath);
    if (!reader.open())
    {
        std::cerr << "Error opening the capture file" << std::endl;
        throw std::exception{};
    }

    // read packet from the file
    pcpp::RawPacket rawPacket;
    while (reader.getNextPacket(rawPacket))
    {
        // parse the raw packet into a parsed packet
        pcpp::Packet parsedPacket(&rawPacket);

        // verify the packet is IPv4
        if (parsedPacket.isPacketOfType(pcpp::IPv4))
        {
            uint16_t srcPort{0}, dstPort{0};

            // extract ports
            if (parsedPacket.isPacketOfType(pcpp::TCP))
            {
                pcpp::TcpLayer* TcpLayer = parsedPacket.getLayerOfType<pcpp::TcpLayer>();
                srcPort = TcpLayer->getSrcPort();
                dstPort = TcpLayer->getDstPort();
            }
            else if (parsedPacket.isPacketOfType(pcpp::UDP))
            {
                pcpp::UdpLayer* UdpLayer = parsedPacket.getLayerOfType<pcpp::UdpLayer>();
                srcPort = UdpLayer->getSrcPort();
                dstPort = UdpLayer->getDstPort();
            }

            // extract IPs
            if (dstPort != 0 && srcPort != 0)
            {
                pcpp::IPv4Address srcIP = parsedPacket.getLayerOfType<pcpp::IPv4Layer>()->getSrcIPv4Address();
                pcpp::IPv4Address dstIP = parsedPacket.getLayerOfType<pcpp::IPv4Layer>()->getDstIPv4Address();

                // print source and dest IPs
                if (srcPort < dstPort)
                {
                    splitter.add_packet(dstIP, dstPort, srcIP, srcPort, rawPacket.getPacketTimeStamp(), rawPacket.getRawDataLen());
                }
                else
                {
                    splitter.add_packet(srcIP, srcPort, dstIP, dstPort, rawPacket.getPacketTimeStamp(), rawPacket.getRawDataLen());
                }
            }
        }
    }

    // close the file
    reader.close();
}

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <capture_file or folder_with_captures>";
    }

    if (std::filesystem::is_regular_file(argv[1]))
    {
        readCaptureFile(argv[1]);
    }
    else if (std::filesystem::is_directory(argv[1]))
    {
        for (const auto& entry : std::filesystem::directory_iterator(argv[1]))
        {
            if (entry.is_regular_file())
            {
                readCaptureFile(entry.path().string().c_str());
            }
        }
    }

    return 0;
}

