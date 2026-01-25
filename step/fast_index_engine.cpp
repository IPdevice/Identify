#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <iostream>
#include <cmath>

namespace py = pybind11;
using namespace std;


using PostingList = unordered_map<int, int>;

using InvertedIndex = unordered_map<string, PostingList>;


using DocumentStore = unordered_map<int, py::object>;


vector<string> tokenize(const string& text) {
    vector<string> tokens;
    string current;
    
    for (char c : text) {
        if (isalnum(c) || c == '-' || c == '.' || c == '_' || c == '/') {
            current += tolower(c);
        } else {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
        }
    }
    if (!current.empty()) {
        tokens.push_back(current);
    }
    
    return tokens;
}


vector<string> tokenize_exact(const string& text) {
    vector<string> tokens;
    string current;
    
    for (char c : text) {
        if (isalnum(c) || c == '-' || c == '.' || c == '_' || c == ':' || c == '/') {
            current += tolower(c);
        } else {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
        }
    }
    if (!current.empty()) {
        tokens.push_back(current);
    }
    
    return tokens;
}


class FastIndexEngine {
private:
    InvertedIndex index_;  // content
    InvertedIndex index_exact_;  // content_exact
    DocumentStore documents_;  // 
    unordered_set<int> document_ids_;  // 
    map<string, vector<string>> field_values_;  
    int doc_count_;
    
public:
    FastIndexEngine() : doc_count_(0) {}
    
    
    bool add_document(int doc_id, const py::dict& doc_data, const vector<string>& selected_fields) {
        if (document_ids_.find(doc_id) != document_ids_.end()) {
            return false;  
        }
        
        document_ids_.insert(doc_id);
       
        if (doc_data.contains("full_document")) {
            documents_[doc_id] = doc_data["full_document"];
        } else {
            documents_[doc_id] = py::none();
        }
        
      
        string content;
        string content_exact;
        
        vector<string> field_list;
        for (auto item : doc_data) {
            string key = py::cast<string>(item.first);
            
           
            if (key == "doc_id" || key == "full_document") {
                continue;
            }
            
          
            if (find(selected_fields.begin(), selected_fields.end(), key) != selected_fields.end() ||
                key == "content" || key == "content_exact") {
                string value = py::cast<string>(py::str(item.second));
                
                if (key == "content") {
                    content = value;
                } else if (key == "content_exact") {
                    content_exact = value;
                } else {
                    if (!content.empty()) content += " ";
                    content += value;
                    if (!content_exact.empty()) content_exact += " ";
                    content_exact += value;
                    
                   
                    if (field_values_[key].size() <= (size_t)doc_id) {
                        field_values_[key].resize(doc_id + 1);
                    }
                    field_values_[key][doc_id] = value;
                }
            }
        }
        
       
        if (!content.empty()) {
            vector<string> tokens = tokenize(content);
            for (const string& token : tokens) {
                index_[token][doc_id]++;
            }
        }
        
       
        if (!content_exact.empty()) {
            vector<string> tokens_exact = tokenize_exact(content_exact);
            for (const string& token : tokens_exact) {
                index_exact_[token][doc_id]++;
            }
        }
        
        doc_count_++;
        return true;
    }
    
  
    vector<py::dict> search_or(const string& query_str, int limit, int k_filter = -1) {
        vector<string> query_terms = tokenize(query_str);
        if (query_terms.empty()) {
            return {};
        }
        
      
        unordered_map<int, double> doc_scores; 
        
        for (const string& term : query_terms) {
            if (index_.find(term) != index_.end()) {
                const PostingList& postings = index_.at(term);
                double idf = log((double)doc_count_ / (postings.size() + 1));
                
                for (const auto& [doc_id, tf] : postings) {
                    if (k_filter >= 0 && doc_id >= k_filter) {
                        continue;
                    }
                   
                    double score = tf * idf;
                    doc_scores[doc_id] += score;
                }
            }
        }
        
      
        vector<pair<int, double>> sorted_docs;
        for (const auto& [doc_id, score] : doc_scores) {
            sorted_docs.push_back({doc_id, score});
        }
        sort(sorted_docs.begin(), sorted_docs.end(),
             [](const pair<int, double>& a, const pair<int, double>& b) {
                 return a.second > b.second;
             });
        
        vector<py::dict> results;
        for (int i = 0; i < min(limit, (int)sorted_docs.size()); i++) {
            int doc_id = sorted_docs[i].first;
            py::dict result;
            result["doc_id"] = to_string(doc_id);
            result["score"] = sorted_docs[i].second;
            result["matches"] = py::dict();
            result["document"] = documents_[doc_id];
            results.push_back(result);
        }
        
        return results;
    }
    
  
    vector<py::dict> search_and(const string& query_str, int limit, int k_filter = -1) {
        vector<string> query_terms = tokenize(query_str);
        if (query_terms.empty()) {
            return {};
        }
        
    
        set<int> candidate_docs;
        if (index_.find(query_terms[0]) != index_.end()) {
            for (const auto& [doc_id, tf] : index_.at(query_terms[0])) {
                if (k_filter < 0 || doc_id < k_filter) {
                    candidate_docs.insert(doc_id);
                }
            }
        }
        
      
        for (size_t i = 1; i < query_terms.size(); i++) {
            set<int> term_docs;
            if (index_.find(query_terms[i]) != index_.end()) {
                for (const auto& [doc_id, tf] : index_.at(query_terms[i])) {
                    if (k_filter < 0 || doc_id < k_filter) {
                        term_docs.insert(doc_id);
                    }
                }
            }
            
            set<int> intersection;
            set_intersection(candidate_docs.begin(), candidate_docs.end(),
                           term_docs.begin(), term_docs.end(),
                           inserter(intersection, intersection.begin()));
            candidate_docs = intersection;
        }
        
       
        vector<pair<int, double>> scored_docs;
        for (int doc_id : candidate_docs) {
            double score = 0.0;
            for (const string& term : query_terms) {
                if (index_.find(term) != index_.end() && 
                    index_.at(term).find(doc_id) != index_.at(term).end()) {
                    int tf = index_.at(term).at(doc_id);
                    double idf = log((double)doc_count_ / (index_.at(term).size() + 1));
                    score += tf * idf;
                }
            }
            scored_docs.push_back({doc_id, score});
        }
        
        sort(scored_docs.begin(), scored_docs.end(),
             [](const pair<int, double>& a, const pair<int, double>& b) {
                 return a.second > b.second;
             });
        
        vector<py::dict> results;
        for (int i = 0; i < min(limit, (int)scored_docs.size()); i++) {
            int doc_id = scored_docs[i].first;
            py::dict result;
            result["doc_id"] = to_string(doc_id);
            result["score"] = scored_docs[i].second;
            result["matches"] = py::dict();
            result["document"] = documents_[doc_id];
            results.push_back(result);
        }
        
        return results;
    }
    
   
    vector<py::dict> search_exact_terms(const vector<string>& terms, int limit, int k_filter = -1) {
        vector<py::dict> results;
        
        try {
            if (terms.empty()) {
                return results;  
            }
            
            if (doc_count_ == 0) {
                return results;  
            }
            
            vector<string> all_tokens;
            for (const string& term : terms) {
                if (term.empty()) {
                    continue;
                }
                
                string normalized = term;
                transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);
                
                vector<string> term_tokens = tokenize_exact(normalized);
                for (const string& token : term_tokens) {
                    if (!token.empty()) {
                        all_tokens.push_back(token);
                    }
                }
            }
            
            if (all_tokens.empty()) {
                return results;  
            }
            
           
            set<int> candidate_docs;
            
           
            if (index_exact_.find(all_tokens[0]) != index_exact_.end()) {
                for (const auto& [doc_id, tf] : index_exact_.at(all_tokens[0])) {
                    if (k_filter < 0 || doc_id < k_filter) {
                        candidate_docs.insert(doc_id);
                    }
                }
            }
            
           
            if (candidate_docs.empty()) {
                return results;
            }
            
           
            for (size_t i = 1; i < all_tokens.size(); i++) {
                set<int> term_docs;
                if (index_exact_.find(all_tokens[i]) != index_exact_.end()) {
                    for (const auto& [doc_id, tf] : index_exact_.at(all_tokens[i])) {
                        if (k_filter < 0 || doc_id < k_filter) {
                            term_docs.insert(doc_id);
                        }
                    }
                }
                
               
                set<int> intersection;
                set_intersection(candidate_docs.begin(), candidate_docs.end(),
                               term_docs.begin(), term_docs.end(),
                               inserter(intersection, intersection.begin()));
                candidate_docs = intersection;
                
               
                if (candidate_docs.empty()) {
                    return results;
                }
            }
            
           
            for (int doc_id : candidate_docs) {
                if ((int)results.size() >= limit) {
                    break;
                }
                
                try {
                    py::dict result;
                    result["doc_id"] = to_string(doc_id);
                    result["score"] = 1.0;  
                    result["matches"] = py::dict();
                    

                    if (documents_.find(doc_id) != documents_.end()) {
                        result["document"] = documents_[doc_id];
                    } else {
                        result["document"] = py::none();
                    }
                    
                    results.push_back(result);
                } catch (...) {
                
                    continue;
                }
            }
        } catch (...) {
           
            return results;
        }
        
        return results;
    }
    
  
    int get_doc_count() const {
        return doc_count_;
    }
    
   
    void clear() {
        index_.clear();
        index_exact_.clear();
        documents_.clear();
        document_ids_.clear();
        field_values_.clear();
        doc_count_ = 0;
    }
};

PYBIND11_MODULE(fast_index_engine, m) {
    m.doc() = "Fast inverted index engine with C++ implementation";
    
    py::class_<FastIndexEngine>(m, "FastIndexEngine")
        .def(py::init<>())
        .def("add_document", &FastIndexEngine::add_document,
             "Add a document to the index",
             py::arg("doc_id"), py::arg("doc_data"), py::arg("selected_fields"))
        .def("search_or", &FastIndexEngine::search_or,
             "Search with OR mode",
             py::arg("query_str"), py::arg("limit") = 10, py::arg("k_filter") = -1)
        .def("search_and", &FastIndexEngine::search_and,
             "Search with AND mode",
             py::arg("query_str"), py::arg("limit") = 10, py::arg("k_filter") = -1)
        .def("search_exact_terms", &FastIndexEngine::search_exact_terms,
             "Search with exact term matching",
             py::arg("terms"), py::arg("limit") = 1000, py::arg("k_filter") = -1)
        .def("get_doc_count", &FastIndexEngine::get_doc_count,
             "Get total document count")
        .def("clear", &FastIndexEngine::clear,
             "Clear the index");
}
