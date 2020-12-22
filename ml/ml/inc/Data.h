#ifndef _DATA_H__
#define _DATA_H__

#include <vector>
#include <stdio.h>
#include <stdint.h>

class Data
{
    std::vector<uint8_t>* m_feature_vector;
    uint8_t m_label;
    uint32_t m_enum_label;
    double m_distance;
public:
    Data();
    virtual ~Data();

    bool set_feature_vector(std::vector<uint8_t>* pvec);
    bool append_to_feature_vector(uint8_t n);

    void set_label(uint8_t n);
    void set_enum_label(int n);
    void set_distance(double d);
    double get_distance() const;
    uint32_t get_feature_vector_size() const;
    uint8_t get_label() const;
    uint32_t get_enum_label() const;
    std::vector<uint8_t>* get_feature_vector();
};
#endif
