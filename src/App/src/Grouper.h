#pragma once
#include "Graph.h"
#include "ConnectionContent.h"

#include <iostream>
#include <memory>

class Grouper
{
public:	
	virtual ~Grouper() = default;
    virtual std::unique_ptr<Graph> group(const ConnectionContent& connection) = 0;
	virtual bool canGroup(const ConnectionContent& connection) = 0;
	virtual bool shouldGroup(const ConnectionContent& connection) = 0;

    static constexpr unsigned int numGroups = 30;
};


class SizeDelayGrouper : public Grouper
{
public:
	virtual ~SizeDelayGrouper() = default;
    virtual std::unique_ptr<Graph> group(const ConnectionContent& connection) override;
	virtual bool canGroup(const ConnectionContent& connection) override;
	virtual bool shouldGroup(const ConnectionContent& connection) override;

    static constexpr size_t minSizeConnection = 50;
    static constexpr size_t minTotalByteSizeConnection = 1024 * 10; // 10 kB
    static constexpr size_t maxSizeConnection = 50;
    static constexpr size_t maxTotalByteSizeConnection = 1024 * 20; // 20 kB
    static constexpr size_t maxTimeLenConnection = 60; // 60s

private:
	template <int NumGroups>
	struct Group
	{
		unsigned int minSize;
		unsigned int maxSize;
		unsigned int count{};
		size_t sum;
		timespec latestTimestamp{};
		float delaysSum[NumGroups]{};
		unsigned int delaysCount[NumGroups]{};
	};

    void initGroups();
    int getGroupIndex(unsigned int size);
    Group<numGroups> groups[numGroups];
};

class PacketListGrouper : public Grouper
{
public:
	virtual ~PacketListGrouper() = default;
    virtual std::unique_ptr<Graph> group(const ConnectionContent& connection) override;
	virtual bool canGroup(const ConnectionContent& connection) override;
	virtual bool shouldGroup(const ConnectionContent& connection) override;

    static constexpr size_t minSizeConnection = 10;
	static constexpr size_t minTotalByteSizeConnection = 0;
};

Grouper& getGrouper(std::string graphType = "");

