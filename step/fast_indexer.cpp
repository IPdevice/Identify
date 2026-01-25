#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <set>
#include <sstream>
#include <algorithm>

namespace py = pybind11;
using namespace std;

string clean_field_name(const string& field_name) {
    string cleaned = field_name;
    if (!cleaned.empty() && cleaned[0] == '_') {
        cleaned = cleaned.substr(1);
    }
    for (char& c : cleaned) {
        if (c == ' ') {
            c = '_';
        }
    }
    return cleaned;
}


unordered_map<string, string> flatten_dict_cpp(
    const py::dict& d,
    const string& parent_key = "",
    const string& sep = ".",
    int max_depth = 3,
    int current_depth = 0) {
    
    unordered_map<string, string> result;
    
    if (current_depth >= max_depth) {
        return result;
    }
    
    for (auto item : d) {
        string key = py::cast<string>(item.first);
        string cleaned_key = clean_field_name(key);
        string new_key = parent_key.empty() ? cleaned_key : parent_key + sep + cleaned_key;
        
        py::object value = py::cast<py::object>(item.second);
        
      
        if (py::isinstance<py::dict>(value)) {
            py::dict nested_dict = py::cast<py::dict>(value);
            auto nested_items = flatten_dict_cpp(nested_dict, new_key, sep, max_depth, current_depth + 1);
            result.insert(nested_items.begin(), nested_items.end());
        }
       
        else if (py::isinstance<py::list>(value)) {
            py::list lst = py::cast<py::list>(value);
            vector<string> str_items;
            size_t limit = min((size_t)100, (size_t)py::len(lst));
            
            for (size_t i = 0; i < limit; i++) {
                py::object item_obj = lst[i];
                if (py::isinstance<py::str>(item_obj)) {
                    str_items.push_back(py::cast<string>(item_obj));
                } else {
                    str_items.push_back(py::cast<string>(py::str(item_obj)));
                }
            }
            
            string list_str;
            for (size_t i = 0; i < str_items.size(); i++) {
                if (i > 0) list_str += ",";
                list_str += str_items[i];
            }
            
            if (py::len(lst) > 100) {
                list_str += "...";
            }
            
            result[new_key] = list_str;
        }
       
        else {
            string str_value;
            if (value.is_none()) {
                str_value = "";
            } else if (py::isinstance<py::str>(value)) {
                str_value = py::cast<string>(value);
            } else {
                str_value = py::cast<string>(py::str(value));
            }
            result[new_key] = str_value;
        }
    }
    
    return result;
}


py::dict clean_json_record_cpp(const py::dict& record) {
    py::dict cleaned;
    
    vector<string> basic_fields = {
        "ip", "ip_str", "country_code", "country_name", "city",
        "latitude", "longitude", "isp", "org", "asn", "last_update",
        "hostnames", "domains", "tags", "data"
    };
    
    for (const auto& field : basic_fields) {
        if (record.contains(field.c_str())) {
            py::object value = py::cast<py::object>(record[field.c_str()]);
            
            if (!value.is_none()) {
               
                if (py::isinstance<py::str>(value)) {
                    string str_value = py::cast<string>(value);
                    if (str_value.length() > 10000) {
                        str_value = str_value.substr(0, 10000) + "...";
                    }
                    cleaned[field.c_str()] = str_value;
                }
               
                else if (py::isinstance<py::list>(value)) {
                    py::list lst = py::cast<py::list>(value);
                    py::list cleaned_list;
                    
                    for (auto item : lst) {
                        py::object item_obj = py::cast<py::object>(item);
                        
                        if (py::isinstance<py::str>(item_obj)) {
                            string str_item = py::cast<string>(item_obj);
                            if (str_item.length() > 1000) {
                                str_item = str_item.substr(0, 1000) + "...";
                            }
                            cleaned_list.append(str_item);
                        }
                        else if (py::isinstance<py::int_>(item_obj) ||
                                 py::isinstance<py::float_>(item_obj) ||
                                 py::isinstance<py::bool_>(item_obj)) {
                            cleaned_list.append(item_obj);
                        }
                    }
                    
                    if (py::len(cleaned_list) > 0) {
                        cleaned[field.c_str()] = cleaned_list;
                    }
                }
               
                else if (py::isinstance<py::int_>(value) ||
                         py::isinstance<py::float_>(value) ||
                         py::isinstance<py::bool_>(value)) {
                    cleaned[field.c_str()] = value;
                }
                
                else if (py::isinstance<py::dict>(value)) {
                    cleaned[field.c_str()] = value;
                }
            }
        }
    }
    
    return cleaned;
}

PYBIND11_MODULE(fast_indexer, m) {
    m.doc() = "Fast indexer module with C++ optimizations";
    
    m.def("flatten_dict", &flatten_dict_cpp,
          "Flatten a nested dictionary",
          py::arg("d"),
          py::arg("parent_key") = "",
          py::arg("sep") = ".",
          py::arg("max_depth") = 3,
          py::arg("current_depth") = 0);
    
    m.def("clean_json_record", &clean_json_record_cpp,
          "Clean a JSON record",
          py::arg("record"));
}