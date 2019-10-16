/*
 * FMM/Timer.hpp
 *
 *  Created on: Nov 17, 2017
 *      Author: wyan
 *
 *      Reference: https://gist.github.com/jtilly/a423be999929d70406489a4103e67453
 */

#ifndef TIMER_HPP_
#define TIMER_HPP_

#include <chrono>
#include <cstdio>
#include <iostream>
#include <sstream>

class Timer {
  private:
    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::high_resolution_clock::time_point stopTime;
    std::stringstream logfile;
    bool work = true;

  public:
    Timer() = default;

    Timer(bool work_) : Timer() { work = work_; }

    Timer(const Timer &other) {
        startTime = other.startTime;
        stopTime = other.stopTime;
        logfile << other.logfile.str();
        work = other.work;
    };

    const Timer &operator=(const Timer &other) {
        startTime = other.startTime;
        stopTime = other.stopTime;
        logfile << other.logfile.str();
        work = other.work;
        return *this;
    }

    ~Timer() = default;

    void start() {
        if (work)
            this->startTime = std::chrono::high_resolution_clock::now();
    }

    void stop(const std::string &s) {
        if (work) {
            this->stopTime = std::chrono::high_resolution_clock::now();
            logfile << s.c_str() << " Time elapsed = "
                    << std::chrono::duration_cast<std::chrono::microseconds>(stopTime - startTime).count() / 1e6
                    << std::endl;
        }
    }

    double getTime() {
        return std::chrono::duration_cast<std::chrono::microseconds>(stopTime - startTime).count() / 1e6;
    }

    void dump() {
        if (work)
            std::cout << logfile.rdbuf();
    }
};

#endif
