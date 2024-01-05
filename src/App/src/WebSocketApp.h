#include <string>
#include <memory>
#include <thread>

class WebSocketApp
{
public:
	void run();
	void broadcast(const std::string& data);

private:
	std::thread m_thread;
};