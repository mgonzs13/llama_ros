#include <iostream>

class LlamaException : public std::exception {
public:
  LlamaException(std::string msg) : message(msg) {}
  const char *what() const throw() { return message.c_str(); }

private:
  std::string message;
};