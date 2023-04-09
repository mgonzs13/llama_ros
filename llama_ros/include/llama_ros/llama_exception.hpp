

#ifndef LLAMA_EXCEPTION_HPP
#define LLAMA_EXCEPTION_HPP

#include <iostream>

namespace llama_ros {

class LlamaException : public std::exception {
public:
  LlamaException(std::string msg) : message(msg) {}
  const char *what() const throw() { return message.c_str(); }

private:
  std::string message;
};
} // namespace llama_ros

#endif
