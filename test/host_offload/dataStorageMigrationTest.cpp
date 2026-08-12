#include "fastllm.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Expect(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

uint64_t HashBytes(const uint8_t *data, size_t bytes) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= data[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

std::vector<float> MakeValues(size_t count) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = static_cast<float>((i * 37 + 11) % 257) / 257.0f;
    }
    return values;
}

void TestCudaHostCudaRoundTripIsExact() {
    const std::vector<float> values = MakeValues(4096);
    fastllm::Data tensor(
        fastllm::DataType::FLOAT32, {1, static_cast<int>(values.size())}, values);
    tensor.name = "offload.roundtrip";
    tensor.isModelWeight = true;
    const uint64_t expectedBytes = tensor.expansionBytes;
    const uint64_t expectedHash = HashBytes(
        reinterpret_cast<const uint8_t *>(values.data()),
        values.size() * sizeof(float));

    tensor.ToDevice(fastllm::DataDevice::CUDA);
    Expect(tensor.cudaData != nullptr && tensor.cpuData == nullptr,
           "test tensor must start CUDA-only");

    fastllm::DataOffloadRecord record;
    std::string error;
    Expect(tensor.MoveCudaStorageToHost(record, &error),
           "GPU to RAM migration failed: " + error);
    Expect(tensor.cudaData == nullptr && tensor.cpuData != nullptr,
           "RAM tier must own the only stable copy after D2H");
    Expect(record.bytes == expectedBytes,
           "offload record must preserve allocation bytes");
    Expect(record.checksum == expectedHash,
           "offload record checksum must cover exact payload bytes");
    Expect(HashBytes(tensor.cpuData, record.bytes) == expectedHash,
           "RAM payload bytes must match the original tensor");

    Expect(tensor.RestoreCudaStorageFromHost(record, &error),
           "RAM to GPU migration failed: " + error);
    Expect(tensor.cudaData != nullptr && tensor.cpuData == nullptr,
           "READY tier must release manager-owned RAM bytes after H2D");

    tensor.ToDevice(fastllm::DataDevice::CPU);
    Expect(tensor.cpuData != nullptr && tensor.cudaData == nullptr,
           "verification copy must return to CPU");
    Expect(HashBytes(tensor.cpuData, expectedBytes) == expectedHash,
           "GPU round trip must preserve every byte");
}

void TestReleaseWithoutCopyPreservesTensorMetadata() {
    const std::vector<float> values = MakeValues(1024);
    fastllm::Data tensor(
        fastllm::DataType::FLOAT32, {4, 256}, values);
    tensor.name = "offload.source_rebuild";
    tensor.weightType = fastllm::WeightType::LINEAR;
    tensor.isModelWeight = true;
    tensor.ToDevice(fastllm::DataDevice::CUDA);

    const auto dims = tensor.dims;
    const auto strides = tensor.strides;
    const uint64_t expansionBytes = tensor.expansionBytes;
    const auto dataType = tensor.dataType;
    const auto weightType = tensor.weightType;
    const std::string name = tensor.name;

    std::string error;
    Expect(tensor.ReleaseCudaStorageWithoutHostCopy(&error),
           "source-rebuild eviction failed: " + error);
    Expect(tensor.cudaData == nullptr && tensor.cpuData == nullptr,
           "source-rebuildable tensor must have no resident payload");
    Expect(tensor.dims == dims && tensor.strides == strides,
           "payload eviction must preserve tensor geometry");
    Expect(tensor.expansionBytes == expansionBytes,
           "payload eviction must preserve allocation geometry");
    Expect(tensor.dataType == dataType && tensor.weightType == weightType,
           "payload eviction must preserve tensor types");
    Expect(tensor.name == name && tensor.isModelWeight,
           "payload eviction must preserve model identity metadata");
}

void TestBorrowedCudaStorageIsRejected() {
    const std::vector<float> values = MakeValues(128);
    fastllm::Data owner(
        fastllm::DataType::FLOAT32, {1, static_cast<int>(values.size())}, values);
    owner.ToDevice(fastllm::DataDevice::CUDA);

    fastllm::Data borrowed(
        fastllm::DataType::FLOAT32, {1, static_cast<int>(values.size())});
    borrowed.dataDevice = fastllm::DataDevice::CUDA;
    borrowed.cudaData = owner.cudaData;
    borrowed.cudaDataBorrowed = true;
    borrowed.expansionSize = owner.expansionSize;
    borrowed.expansionBytes = owner.expansionBytes;

    fastllm::DataOffloadRecord record;
    std::string error;
    Expect(!borrowed.MoveCudaStorageToHost(record, &error),
           "borrowed CUDA storage must not be migrated as an owner");
    Expect(borrowed.cudaData == owner.cudaData && borrowed.cpuData == nullptr,
           "rejected borrowed storage must remain untouched");
    Expect(!error.empty(), "borrowed-storage rejection must explain the error");

    borrowed.cudaData = nullptr;
    borrowed.cudaDataBorrowed = false;
}

} // namespace

int main() {
#ifdef USE_CUDA
    try {
        TestCudaHostCudaRoundTripIsExact();
        TestReleaseWithoutCopyPreservesTensorMetadata();
        TestBorrowedCudaStorageIsRejected();
        std::cout << "data storage migration: PASS\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "data storage migration: FAIL: " << error.what() << "\n";
        return 1;
    }
#else
    std::cout << "data storage migration: SKIP (CUDA disabled)\n";
    return 0;
#endif
}
