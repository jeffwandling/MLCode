#include <Data.h>
#include "config.h"

Data::Data() {
  m_distance = 0.0;
  m_label = 0u;
  m_enum_label = 0u;
  m_feature_vector = new std::vector<uint8_t>;
}

Data::~Data() {
    SAFE_DELETE(m_feature_vector);
}

double
Data::get_distance() const {
    return m_distance;
}

void
Data::set_distance(double d) {
    m_distance = d;
}

bool
Data::set_feature_vector(std::vector<uint8_t> *pvec) {
    if (pvec == NULL) {
         return false;
    }
    m_feature_vector = pvec;
    return true;
}

bool
Data::append_to_feature_vector(uint8_t n) {
    if (m_feature_vector == NULL) {
       return false;
    }
    m_feature_vector->push_back(n);
    return true;
}

void
Data::set_label(uint8_t n) {
    m_label = n;
}

void
Data::set_enum_label(int n) {
    m_enum_label = n;
}

uint32_t
Data::get_feature_vector_size() const {
    if (m_feature_vector == NULL) {
         return 0u;
    }
    return (uint32_t)m_feature_vector->size();
}

uint8_t
Data::get_label() const {
    return m_label;
}

uint32_t
Data::get_enum_label() const {
    return m_enum_label;
}

std::vector<uint8_t>*
Data::get_feature_vector() {
    return m_feature_vector;
}

