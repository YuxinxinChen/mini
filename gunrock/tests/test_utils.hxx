#pragma once

#include <map>
#include <string>
#include <sstream>
#include <chrono>
#include <ctime>
#include "moderngpu/memory.hxx"

using namespace std;

namespace gunrock {

/**
 * CommandLineArgs interface
 */
class CommandLineArgs
{
private:
    int argc;
    char ** argv;
protected:

    std::map<std::string, std::string> pairs;

public:

    // Constructor
    CommandLineArgs(int _argc, char **_argv) : argc(_argc), argv(_argv)
    {
        for (int i = 1; i < argc; i++)
        {
            std::string arg = argv[i];

            if ((arg[0] != '-') || (arg[1] != '-'))
            {
                continue;
            }

            std::string::size_type pos;
            std::string key, val;
            if ((pos = arg.find('=')) == std::string::npos)
            {
                key = std::string(arg, 2, arg.length() - 2);
                val = "";
            }
            else
            {
                key = std::string(arg, 2, pos - 2);
                val = std::string(arg, pos + 1, arg.length() - 1);
            }
            pairs[key] = val;
        }
    }

    // Checks whether a flag "--<flag>" is present in the commandline
    bool CheckCmdLineFlag(const char* arg_name)
    {
        std::map<std::string, std::string>::iterator itr;
        if ((itr = pairs.find(arg_name)) != pairs.end())
        {
            return true;
        }
        return false;
    }

    // Returns the value specified for a given commandline
    // parameter --<flag>=<value>
    template <typename T>
    void GetCmdLineArgument(const char *arg_name, T &val);

    // Returns the values specified for a given commandline
    // parameter --<flag>=<value>,<value>*
    template <typename T>
    void GetCmdLineArguments(const char *arg_name, std::vector<T> &vals);

    // The number of pairs parsed
    int ParsedArgc()
    {
        return pairs.size();
    }

    std::string GetEntireCommandLine() const
    {
        std::string commandLineStr = "";
        for (int i = 0; i < argc; i++)
        {
            commandLineStr.append(std::string(argv[i]).append((i < argc - 1) ? " " : ""));
        }
        return commandLineStr;
    }

    template <typename T>
    void ParseArgument(const char *name, T &val)
    {
        if (CheckCmdLineFlag(name))
        {
            GetCmdLineArgument(name, val);
        }
    }
};

template <typename T>
void CommandLineArgs::GetCmdLineArgument(
    const char *arg_name,
    T &val)
{
    std::map<std::string, std::string>::iterator itr;
    if ((itr = pairs.find(arg_name)) != pairs.end())
    {
        std::istringstream str_stream(itr->second);
        str_stream >> val;
    }
}

template <typename T>
void CommandLineArgs::GetCmdLineArguments(
    const char *arg_name,
    std::vector<T> &vals)
{
    // Recover multi-value string
    std::map<std::string, std::string>::iterator itr;
    if ((itr = pairs.find(arg_name)) != pairs.end())
    {

        // Clear any default values
        vals.clear();

        std::string val_string = itr->second;
        std::istringstream str_stream(val_string);
        std::string::size_type old_pos = 0;
        std::string::size_type new_pos = 0;

        // Iterate comma-separated values
        T val;
        while ((new_pos = val_string.find(',', old_pos)) != std::string::npos)
        {

            if (new_pos != old_pos)
            {
                str_stream.width(new_pos - old_pos);
                str_stream >> val;
                vals.push_back(val);
            }

            // skip over comma
            str_stream.ignore(1);
            old_pos = new_pos + 1;
        }

        // Read last value
        str_stream >> val;
        vals.push_back(val);
    }
}

template<typename type_t>
cudaError_t display_device_data(const type_t *data, std::size_t length) {
    cudaError_t ret = cudaSuccess;
    std::vector<type_t> dest(length);
    if (ret = dtoh(dest, data, length)) return ret;
    for (auto item = dest.begin(); item != dest.end(); ++item)
        std::cout << *item << ' ';
    std::cout << std::endl;
    return ret;
}

class test_timer_t {
    std::chrono::time_point<std::chrono::system_clock> s, e;
    bool counting;

    public:
    test_timer_t() : counting(false) {}
   
    void start() {
        if (!counting) {
            counting = true;
            s = std::chrono::system_clock::now();
        }
    }

    double end() {
        if (counting) {
            e = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed = e - s;
            return elapsed.count();
        } else {
            return 0.0;
        }
    }
};

bool validate(std::vector<int> &gpu_vals, std::vector<int> &cpu_vals) {
    if (gpu_vals.size() != cpu_vals.size())
        return false;
    for (int i = 0; i < gpu_vals.size(); ++i) {
        if (gpu_vals[i] != cpu_vals[i]) {
            return false;
        }
    }
    return true;
}

bool validate(std::vector<float> &gpu_vals, std::vector<float> &cpu_vals) {
    if (gpu_vals.size() != cpu_vals.size())
        return false;
    for (int i = 0; i < gpu_vals.size(); ++i) {
        if (fabs(gpu_vals[i] - cpu_vals[i]) < 0.01f) {
            return false;
        }
    }
    return true;
}

}
