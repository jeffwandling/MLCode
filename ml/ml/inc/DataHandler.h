#ifndef _DATA_HANDLER_H__
#define _DATA_HANDLER_H__

#include <fstream>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include <unordered_set>

#include "Data.h"

class DataHandler
{
    std::vector<Data*>* m_data_array;
    std::vector<Data*>* m_training_data;
    std::vector<Data*>* m_test_data;
    std::vector<Data*>* m_validation_data;
    uint32_t m_num_classes;
    uint32_t m_feature_vector_size;
    std::map<uint8_t, int> m_class_map;

    const uint32_t TRAIN_SET_PERCENT = 75u;
    const uint32_t TEST_SET_PERCENT = 20u;
    const uint32_t VALIDATION_PERCENT = 5u;
public:
    DataHandler();
    virtual ~DataHandler();

    bool read_feature_vector(std::string path);
    bool read_feature_labels(std::string path);

    void split_data();
    void count_classes();

    uint32_t convert_to_little_endian(unsigned char* bytes) const;
    std::vector<Data*>* get_training_data();
    std::vector<Data*>* get_test_data();
    std::vector<Data*>* get_validation_data();
};

#endif
