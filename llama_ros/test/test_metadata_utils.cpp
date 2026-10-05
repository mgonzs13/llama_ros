// MIT License
//
// Copyright (c) 2026 Miguel Ángel González Santamarta
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "llama_ros/metadata_utils.hpp"

TEST(MetadataUtilsTest, ParsesStringArray) {
  const auto values = llama_ros::parse_metadata_array(
      R"(["gemma3", "gemma", "google", "text-generation"])");
  EXPECT_EQ(values, (std::vector<std::string>{"gemma3", "gemma", "google",
                                              "text-generation"}));
}

TEST(MetadataUtilsTest, ParsesNumericArray) {
  const auto values = llama_ros::parse_metadata_array("[1, 2, 3]");
  EXPECT_EQ(values, (std::vector<std::string>{"1", "2", "3"}));
}

TEST(MetadataUtilsTest, ParsesEscapesAndCommas) {
  const auto values =
      llama_ros::parse_metadata_array(R"(["a\"b", "c\\d", "e,f"])");
  EXPECT_EQ(values, (std::vector<std::string>{"a\"b", "c\\d", "e,f"}));
}

TEST(MetadataUtilsTest, ParsesQuotedEmptyAndWhitespaceElements) {
  EXPECT_EQ(llama_ros::parse_metadata_array(R"([""])"),
            (std::vector<std::string>{""}));
  EXPECT_EQ(llama_ros::parse_metadata_array(R"([" a "])"),
            (std::vector<std::string>{" a "}));
  EXPECT_EQ(llama_ros::parse_metadata_array("[a, , b]"),
            (std::vector<std::string>{"a", "b"}));
}

TEST(MetadataUtilsTest, ParsesBracketsInsideQuotes) {
  EXPECT_EQ(llama_ros::parse_metadata_array(R"(["a]b", "c[d"])"),
            (std::vector<std::string>{"a]b", "c[d"}));
}

TEST(MetadataUtilsTest, ParsesEmptyAndMalformedArrays) {
  EXPECT_TRUE(llama_ros::parse_metadata_array("[]").empty());
  EXPECT_TRUE(llama_ros::parse_metadata_array("[ ]").empty());
  EXPECT_TRUE(llama_ros::parse_metadata_array("not-an-array").empty());
  EXPECT_TRUE(llama_ros::parse_metadata_array("").empty());
}

TEST(MetadataUtilsTest, RejectsMalformedArrays) {
  EXPECT_TRUE(llama_ros::parse_metadata_array(R"(["a, b])").empty());
  EXPECT_TRUE(llama_ros::parse_metadata_array(R"(["a])").empty());
  EXPECT_TRUE(llama_ros::parse_metadata_array("junk [x] junk").empty());
  EXPECT_TRUE(llama_ros::parse_metadata_array("[[1,2],[3,4]]").empty());
  EXPECT_TRUE(llama_ros::parse_metadata_array("[a\\]").empty());
}

TEST(MetadataUtilsTest, ParsesIntegersSafely) {
  EXPECT_EQ(llama_ros::parse_metadata_int("20"), 20);
  EXPECT_EQ(llama_ros::parse_metadata_int("-3"), -3);
  EXPECT_EQ(llama_ros::parse_metadata_int("[512, 512]"), 0);
  EXPECT_EQ(llama_ros::parse_metadata_int(""), 0);
  EXPECT_EQ(llama_ros::parse_metadata_int("abc", 7), 7);
}

TEST(MetadataUtilsTest, RejectsOutOfRangeIntegers) {
  EXPECT_EQ(llama_ros::parse_metadata_int("2147483648"), 0);
  EXPECT_EQ(llama_ros::parse_metadata_int("-2147483649", 7), 7);
}

TEST(MetadataUtilsTest, ParsesFloatsSafely) {
  EXPECT_FLOAT_EQ(llama_ros::parse_metadata_float("0.95"), 0.95f);
  EXPECT_FLOAT_EQ(llama_ros::parse_metadata_float("1e-5"), 1e-5f);
  EXPECT_FLOAT_EQ(llama_ros::parse_metadata_float("[1.5, 2.5]"), 0.0f);
  EXPECT_FLOAT_EQ(llama_ros::parse_metadata_float("", 1.0f), 1.0f);
}

TEST(MetadataUtilsTest, SplitsSemicolonList) {
  EXPECT_EQ(llama_ros::split_semicolon("top_k;top_p;temp"),
            (std::vector<std::string>{"top_k", "top_p", "temp"}));
  EXPECT_TRUE(llama_ros::split_semicolon("").empty());
  EXPECT_EQ(llama_ros::split_semicolon("top_k;"),
            (std::vector<std::string>{"top_k"}));
  EXPECT_EQ(llama_ros::split_semicolon(" top_k ; top_p "),
            (std::vector<std::string>{"top_k", "top_p"}));
  EXPECT_TRUE(llama_ros::split_semicolon(";;").empty());
}
