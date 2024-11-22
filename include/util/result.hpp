#ifndef RESULT_H_
#define RESULT_H_

#include<string>
#include<optional>

struct Status {
    Status(bool succeed, std::string errorMessage = ""):
        succeed(succeed), errorMessage(errorMessage) {}

    bool succeed;
    std::string errorMessage;
};

template<typename T>
struct Result {
    Result(bool succeed, std::string errorMessage = ""):
        status(succeed, errorMessage) {}
    Result(Status status):
        status(status) {}
    Result(T value):
        value(value), status(true) {}

    std::optional<T> value;
    Status status;
};

#endif