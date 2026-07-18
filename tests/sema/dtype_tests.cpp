#include "blackforge/sema/dtype.hpp"

#include <gtest/gtest.h>

using namespace blackforge;

namespace {

ast::DottedName makeName(const std::string& text) {
    ast::DottedName name;
    std::string segment;
    for (char c : text) {
        if (c == '.') {
            name.segments.push_back(segment);
            segment.clear();
        } else {
            segment += c;
        }
    }
    name.segments.push_back(segment);
    return name;
}

}  // namespace

TEST(DTypeTest, RiconosceTuttiIFormatiSupportati) {
    EXPECT_EQ(sema::parseDType(makeName("fp8.e4m3")), sema::DType::FP8_E4M3);
    EXPECT_EQ(sema::parseDType(makeName("fp8.e5m2")), sema::DType::FP8_E5M2);
    EXPECT_EQ(sema::parseDType(makeName("fp16")), sema::DType::FP16);
    EXPECT_EQ(sema::parseDType(makeName("bf16")), sema::DType::BF16);
    EXPECT_EQ(sema::parseDType(makeName("tf32")), sema::DType::TF32);
    EXPECT_EQ(sema::parseDType(makeName("fp32")), sema::DType::FP32);
}

TEST(DTypeTest, RifiutaNomeSconosciuto) {
    EXPECT_FALSE(sema::parseDType(makeName("int8")).has_value());
    EXPECT_FALSE(sema::parseDType(makeName("fp8.e3m4")).has_value());
}

TEST(DTypeTest, DimensioneInByteCorretta) {
    EXPECT_EQ(sema::dtypeSizeInBytes(sema::DType::FP8_E4M3), 1);
    EXPECT_EQ(sema::dtypeSizeInBytes(sema::DType::FP16), 2);
    EXPECT_EQ(sema::dtypeSizeInBytes(sema::DType::BF16), 2);
    EXPECT_EQ(sema::dtypeSizeInBytes(sema::DType::TF32), 4);
    EXPECT_EQ(sema::dtypeSizeInBytes(sema::DType::FP32), 4);
}

TEST(DTypeTest, Tf32NonEValidoComeFormatoDiStorage) {
    EXPECT_FALSE(sema::isValidForStorage(sema::DType::TF32));
    EXPECT_TRUE(sema::isValidForStorage(sema::DType::BF16));
    EXPECT_TRUE(sema::isValidForStorage(sema::DType::FP32));
}

TEST(DTypeTest, NomeTestualeRoundTrip) {
    EXPECT_EQ(sema::dtypeName(sema::DType::BF16), "bf16");
    EXPECT_EQ(sema::dtypeName(sema::DType::FP8_E4M3), "fp8.e4m3");
}
