

#include "llama_ros/schema_converter.hpp"

using namespace llama_ros;
using json = nlohmann::json;

SchemaConverter::SchemaConverter(const std::vector<std::string> &prop_order) {

  for (int i = 0; i < (int)prop_order.size(); ++i) {
    if (prop_order[i].size() > 0) {
      this->prop_order[prop_order[i]] = i;
    }
  }

  this->rules["space"] = SPACE_RULE;
}

std::string SchemaConverter::visit(const json &schema,
                                   const std::string &name) {

  std::string schema_type = schema.value("type", "");
  std::string rule_name = name.empty() ? "root" : name;

  if (schema.contains("oneOf") || schema.contains("anyOf")) {
    std::vector<std::string> alt_rules;

    for (const auto &alt_schema :
         schema.contains("oneOf")
             ? schema.value("oneOf", std::vector<std::string>({}))
             : schema.value("anyOf", std::vector<std::string>({}))) {
      alt_rules.push_back(visit(alt_schema));
    }

    std::string rule =
        std::accumulate(alt_rules.begin(), alt_rules.end(), std::string(" | "));

    return add_rule(rule_name, rule);

  } else if (schema.contains("const")) {
    return add_rule(rule_name, format_literal(schema["const"]));

  } else if (schema.contains("enum")) {
    std::vector<std::string> enum_literals;

    for (const auto &v : schema["enum"]) {
      enum_literals.push_back(format_literal(v));
    }

    std::string rule = std::accumulate(enum_literals.begin(),
                                       enum_literals.end(), std::string(" | "));

    return add_rule(rule_name, rule);

  } else if (schema_type == "object" && schema.contains("properties")) {
    std::vector<std::pair<std::string, json>> prop_pairs;

    for (auto it = schema["properties"].begin();
         it != schema["properties"].end(); it++) {
      prop_pairs.push_back({it.key(), it.value()});
    }

    std::sort(prop_pairs.begin(), prop_pairs.end(),
              [this](const auto &a, const auto &b) {
                return this->prop_order[a.first] < this->prop_order[b.first] ||
                       (this->prop_order[a.first] ==
                            this->prop_order[b.first] &&
                        a.first < b.first);
              });

    std::string rule = "\"{\" space";

    for (int i = 0; i < (int)prop_pairs.size(); ++i) {
      std::string prop_rule_name =
          visit(prop_pairs[i].second, name.empty()
                                          ? prop_pairs[i].first
                                          : name + "-" + prop_pairs[i].first);
      if (i > 0) {
        rule += "\",\" space";
      }

      rule += " " + format_literal(prop_pairs[i].first) +
              " space \":\" space " + prop_rule_name;
    }

    rule += " \"}\" space";

    return add_rule(rule_name, rule);

  } else if (schema_type == "array" && schema.contains("items")) {

    std::string item_rule_name =
        visit(schema["items"], name.empty() ? "item" : name + "-item");
    std::string list_item_operator = "(\",\" space " + item_rule_name + ")";
    std::string successive_items = "";
    std::string first_item;

    int min_items = schema.value("minItems", 0);
    int max_items = schema.value("maxItems", -1);

    if (min_items > 0) {
      first_item = "(" + item_rule_name + ")";

      for (int i = 0; i < min_items - 1; i++) {
        successive_items += "*";
      }

      min_items--;

    } else {
      first_item = "(" + item_rule_name + ")?";
    }

    if (max_items > min_items) {
      for (int i = 0; i < max_items - min_items - 1; i++) {
        successive_items += list_item_operator + "?";
      }

    } else {
      successive_items += list_item_operator + "*";
    }

    std::string rule =
        "\"[\" space " + first_item + " " + successive_items + " \"]\" space";

    return add_rule(rule_name, rule);

  } else {
    auto it = PRIMITIVE_RULES.find(schema_type);
    assert(it != PRIMITIVE_RULES.end());
    return add_rule((rule_name == "root") ? "root" : schema_type, it->second);
  }
}

std::string SchemaConverter::format_grammar() {
  std::string result;
  for (const auto &rule : this->rules) {
    result += rule.first + " ::= " + rule.second + "\n";
  }
  return result;
}

std::string SchemaConverter::format_literal(const json &literal) {
  std::string escaped =
      json::parse("\"\\\"" + literal.get<std::string>() + "\\\"\"").dump();
  return escaped;
}

std::string SchemaConverter::add_rule(const std::string &name,
                                      const std::string &rule) {

  std::string esc_name = name;
  std::string key;

  std::replace_if(
      esc_name.begin(), esc_name.end(),
      [](char c) { return !std::isalnum(c) && c != '-'; }, '-');

  if (this->rules.find(esc_name) == this->rules.end() ||
      this->rules[esc_name] == rule) {
    key = esc_name;

  } else {
    int i = 0;

    while (this->rules.find(esc_name + std::to_string(i)) !=
           this->rules.end()) {
      i++;
    }
    key = esc_name + std::to_string(i);
  }

  this->rules[key] = rule;
  return key;
}