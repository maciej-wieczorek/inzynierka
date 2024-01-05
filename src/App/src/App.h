#include "Classifier.h"

#include "Connection.h"
#include "WebSocketApp.h"

#include <string>

class App
{
public:
	App(std::string action, std::string source, std::string path);
	void run();
	void processConnection(const Connection& connection);

private:
	std::string m_action;
	std::string m_source;
	std::string m_path;
	WebSocketApp m_webSocketApp{};
	Classifier m_classifier;
};
