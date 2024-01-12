#include "App.h"
#include "args.h"

#include <iostream>

int main(int argc, char* argv[])
{
    args::ArgumentParser parser("App", "App for running predictions on network traffic or creating datasets for training a model.");

    args::Group commands(parser, "Actions", args::Group::Validators::AtLeastOne);
    args::Command predict(commands, "predict", "Makes predictions and sends them through websocket");
    args::Command dataset(commands, "dataset", "Creates/Appends dataset");
    
    args::Group arguments(parser, "arguments", args::Group::Validators::DontCare, args::Options::Global);
    args::HelpFlag help(arguments, "help", "Display this help menu", { 'h', "help" });
    args::ValueFlag<std::string> datasetDir(arguments, "dir", "Set the output directory for dataset", { 'd' }, "dataset");
    args::ValueFlag<std::string> modelPath(arguments, "model", "Path to serialized model", { 'm' }, "model.pt");
    args::ValueFlag<std::string> graphRepr(arguments, "repr", "Graph representation: packet_list or size_delay", { 'r' }, "packet_list");
    args::Positional<std::string> source(arguments, "source", "Source of packets: <interface_ip_or_name>|<capture_file>|<folder_with_captures>");

    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch (const args::Help&)
    {
        std::cout << parser;
        return 0;
    }
    catch (const args::ParseError& e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
    catch (...)
    {
        std::cerr << parser;
        return 1;
    }

    getGrouper(graphRepr.Get()); // setup grouper

    std::string action = predict ? "predict" : "dataset";
    std::string path = predict ? modelPath.Get() : datasetDir.Get();

	App app{ action, source.Get(), path};
	app.run();

    return 0;
}

