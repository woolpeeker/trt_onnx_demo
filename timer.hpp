#pragma once
#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

namespace TIMER {
using std::string;
using std::vector;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::nanoseconds;
using std::chrono::system_clock;
using std::chrono::time_point;

class Timer {
   public:
    Timer(string name);
    void start();
    void pause();
    void restart();
    double getAverage_ms();
    double getTotal_ms();
    string getName() {
        return name_;
    }
    void setName(string v) {
        name_ = v;
    }
    string print() {
        std::stringstream stream;
        stream << "Timer " << getName() << ", "
               << "avg = " << getAverage_ms() << " ms";
        return stream.str();
    }

   private:
    nanoseconds getAverage();

   private:
    system_clock::time_point start_time_;
    string name_;
    vector<nanoseconds> times_;
};

using std::chrono::duration_cast;
Timer::Timer(string name)
    : name_(name) {
}

void Timer::start() {
    start_time_ = std::chrono::system_clock::now();
}

void Timer::pause() {
    auto end = std::chrono::system_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start_time_);
    times_.push_back(duration);
}

nanoseconds Timer::getAverage() {
    nanoseconds sum(0);
    for (auto &ss : times_) {
        sum += ss;
    }
    if (times_.size() > 0) {
        return sum / times_.size();
    }
    return nanoseconds();
}

double Timer::getTotal_ms() {
    std::cout << "Not Implements: getTotal_ms" << std::endl;
    return 0;
}

double Timer::getAverage_ms() {
    nanoseconds avg = getAverage();
    return ((double)avg.count()) / duration_cast<nanoseconds>(milliseconds(1)).count();
}

void Timer::restart() {
    times_ = vector<nanoseconds>();
    start();
}

}  // namespace TIMER