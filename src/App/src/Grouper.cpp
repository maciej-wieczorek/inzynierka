#include "Grouper.h"
#include "Utilities.h"

std::unique_ptr<Graph> SizeDelayGrouper::group(const ConnectionContent& connection)
{
    initGroups();
    for (const auto& entry : connection.entries)
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

    auto graph = std::make_unique<SizeDelayGraph>();
    std::vector<float> delays;
    for (unsigned int i = 0; i < numGroups; ++i)
    {
        for (unsigned int j = 0; j < numGroups; ++j)
        {
            if (groups[i].delaysCount[j] > 0)
            {
                float avgDelay = groups[i].delaysSum[j] / groups[i].delaysCount[j];
				delays.push_back(avgDelay);
                graph->edgeList.push_back(SizeDelayGraph::Edge{ i, j, avgDelay });
            }
        }
    }

    z_scale(delays);
    for (size_t i = 0; i < graph->edgeList.size(); ++i)
    {
        graph->edgeList[i].weight = delays[i];
    }

    return std::move(graph);
}

bool SizeDelayGrouper::canGroup(const ConnectionContent& connection)
{
    return connection.getCountPackets() >= minSizeConnection && connection.getTotalSizePackets() >= minTotalByteSizeConnection;
}

bool SizeDelayGrouper::shouldGroup(const ConnectionContent& connection)
{
    if (canGroup(connection))
    {
		if (connection.getCountPackets() >= maxSizeConnection)
		{
			return true;
		}
		else if (connection.getTotalSizePackets() >= maxTotalByteSizeConnection)
		{
			return true;
		}
		else if (connection.getLastTimestamp() - connection.getFirstTimestamp() >= maxTimeLenConnection * 1000000000)
		{
			return true;
		}
    }
    else
    {
        return false;
    }
}

void SizeDelayGrouper::initGroups()
{
    unsigned int sizeStart = 40;
    unsigned int sizeEnd = 1520;
    unsigned int n = numGroups / 2;
    unsigned int inc = (sizeEnd - sizeStart) / n;
    unsigned int i = 0;

    // lower half
    for (; i < n; ++i)
    {
        groups[i] = Group<numGroups>{ sizeStart, sizeStart + inc };
        sizeStart += inc + 1;
    }

    sizeStart = sizeEnd + 1;
    sizeEnd = 65535;
    inc = (sizeEnd - sizeStart) / n;

    // upper half
    for (; i < numGroups; ++i)
    {
        groups[i] = Group<numGroups>{ sizeStart, sizeStart + inc };
        sizeStart += inc + 1;
    }
}

int SizeDelayGrouper::getGroupIndex(unsigned int size)
{
    int left = 0;
    int right = numGroups - 1;

    // handle edge cases
    if (size < groups[left].minSize)
    {
        return left;
    }
    else if (size > groups[right].maxSize)
    {
        return right;
    }

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

Grouper& getGrouper(std::string graphType)
{
    static std::unique_ptr<Grouper> grouper = nullptr;

    if (graphType == "packet_list")
    {
        grouper = std::make_unique<PacketListGrouper>();
    }
    else if (graphType == "size_delay")
    {
        grouper = std::make_unique<SizeDelayGrouper>();
    }

    return *grouper;
}

std::unique_ptr<Graph> PacketListGrouper::group(const ConnectionContent& connection)
{
    auto graph = std::make_unique<PacketListGraph>();

    for (const auto& entry : connection.entries)
    {
        graph->nodes.emplace_back(std::make_unique<uint8_t[]>(entry.size));
        memcpy(graph->nodes[graph->nodes.size() - 1].get(), entry.rawPacketData.get(), entry.payloadSize);

        graph->sizes.emplace_back(entry.payloadSize);
    }

    return std::move(graph);
}

bool PacketListGrouper::canGroup(const ConnectionContent& connection)
{
    size_t countNonZero = 0;
    for (const auto& entry : connection.entries)
    {
        if (entry.payloadSize > 0)
        {
            ++countNonZero;
        }
    }
    return countNonZero >= minSizeConnection && connection.getTotalSizePackets() >= minTotalByteSizeConnection;
}

bool PacketListGrouper::shouldGroup(const ConnectionContent& connection)
{
    return canGroup(connection);
}
