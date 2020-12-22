#include <iostream>
#include "config.h"
#include "DataHandler.h"

DataHandler::DataHandler() {
  m_data_array = new std::vector<Data *>;
  m_test_data = new std::vector<Data *>;
  m_training_data = new std::vector<Data *>;
  m_validation_data = new std::vector<Data *>;
  m_num_classes = 0u;
  m_feature_vector_size = 0u;
}

DataHandler::~DataHandler() {
  SAFE_DELETE(m_data_array);
  SAFE_DELETE(m_test_data);
  SAFE_DELETE(m_training_data);
  SAFE_DELETE(m_validation_data);
}

bool DataHandler::read_feature_vector(std::string path)
{
  uint32_t header[4] = {0};
  unsigned char bytes[4] = {0};

  FILE *fp = fopen(path.c_str(), "rb");
  if (fp == NULL)
  {
    printf("ERROR: Could not find the file: %s\n", path.c_str());
    return false;
  }

  for (uint32_t n = 0u; n < 4u; n++)
  {
    if (fread(bytes, sizeof(bytes), 1u, fp))
    {
      header[n] = convert_to_little_endian(bytes);
    }
  }

  uint32_t image_size = header[2] * header[3];
#ifdef _DEBUG
  printf("INFO: Done getting file header\n");
  printf("INFO: Image_size = %d\n", image_size);
#endif
  for (uint32_t i = 0u; i < header[1]; i++) {
    Data *pData = new Data;
    // we can read faster by reading image_size bytes at a time
    uint8_t *pElements = new uint8_t[image_size];
    if (size_t r = fread(pElements, sizeof(uint8_t), image_size, fp)) {
      // now that we read that set, append it.
      for (uint32_t j = 0; j < image_size; j++) {
        pData->append_to_feature_vector(pElements[j]);
      }
    } else {
      printf("ERROR: Unsuccessful trying to read data.\n");
      SAFE_DELETE(pData);
      SAFE_DELETEA(pElements);
      return false;
    }
    m_data_array->push_back(pData);
    SAFE_DELETEA(pElements);
  }
#ifdef _DEBUG
  printf("INFO: Size of data_array %d\n", (uint32_t) m_data_array->size());
  printf("INFO: Successfully read/stored feature vectors.\n");
#endif
  return true;
}

bool DataHandler::read_feature_labels(std::string path)
{
  uint32_t header[4] = {0};
  unsigned char bytes[4] = {0};

  FILE *fp = fopen(path.c_str(), "rb");
  if (fp == NULL)
  {
    printf("ERROR: Could not find the file %s\n", path.c_str());
    return false;
  }

  for (uint32_t n = 0; n < 2; n++) {
    if (fread(bytes, sizeof(bytes), 1, fp)) {
      header[n] = convert_to_little_endian(bytes);
    }
  }
#ifdef _DEBUG
  printf("INFO: Done getting label file header\n");
#endif
  uint8_t *pElements = new uint8_t[header[1]];
  if (fread(pElements, sizeof(uint8_t), header[1], fp)) {
    for (uint32_t i = 0; i < header[1]; i++) {
      m_data_array->at(i)->set_label(pElements[i]);
    }
  } else {
    printf("ERROR: error reading from file.\n");
    SAFE_DELETEA(pElements);
    return false;
  }
  SAFE_DELETEA(pElements);
#ifdef _DEBUG
  printf("INFO: Successfully read/stored label.\n");
#endif
  return true;
}

void DataHandler::split_data()
{
  std::unordered_set<int> used_indexes;
  uint32_t data_size = (uint32_t)m_data_array->size();

  uint32_t train_size = (data_size * TRAIN_SET_PERCENT) / 100u;
  uint32_t test_size = (data_size * TEST_SET_PERCENT) / 100u;
  uint32_t valid_size = (data_size * VALIDATION_PERCENT) / 100u;

  srand((unsigned int)time(NULL));

  while (train_size--) 
  {
    uint32_t rand_index = rand() % data_size;
    if (used_indexes.find(rand_index) == used_indexes.end()) {
      m_training_data->push_back(m_data_array->at(rand_index));
      used_indexes.insert(rand_index);
    }
  }

  while (test_size--) {
    uint32_t rand_index = rand() % data_size;
    if (used_indexes.find(rand_index) == used_indexes.end()) {
      m_test_data->push_back(m_data_array->at(rand_index));
      used_indexes.insert(rand_index);
    }
  }

  while (valid_size--) {
    uint32_t rand_index = rand() % data_size;
    if (used_indexes.find(rand_index) == used_indexes.end()) {
      m_validation_data->push_back(m_data_array->at(rand_index));
      used_indexes.insert(rand_index);
    }
  }

#ifdef _DEBUG
  printf("INFO: Training data size: %d\n", (uint32_t)m_training_data->size());
  printf("INFO: Test data size: %d\n", (uint32_t)m_test_data->size());
  printf("INFO: Validation data size: %d\n", (uint32_t)m_validation_data->size());
#endif
}

void DataHandler::count_classes()
{
  uint32_t count = 0;
  for (uint32_t i = 0; i < m_data_array->size(); i++) {
    if (m_class_map.find(m_data_array->at(i)->get_label()) == m_class_map.end()) {
      m_class_map[m_data_array->at(i)->get_label()] = count;
      m_data_array->at(i)->set_enum_label(count);
      count++;
    }
  }
  m_num_classes = count;
#ifdef _DEBUG
  printf("INFO: Successfully extracted %d unique classes\n", m_num_classes);
#endif
}

inline uint32_t
DataHandler::convert_to_little_endian(unsigned char *bytes) const
{
  uint32_t result = (uint32_t)((bytes[0] << 24) | (bytes[1] << 16) |
                               (bytes[2] << 8) | (bytes[3]));
  return result;
}

std::vector<Data*>*
DataHandler::get_training_data()
{
    return m_training_data;
}

std::vector<Data*>*
DataHandler::get_test_data()
{
    return m_test_data;
}

std::vector<Data*>*
DataHandler::get_validation_data() 
{
    return m_validation_data;
}

