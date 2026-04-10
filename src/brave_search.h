// brave_search.h — Brave Web Search API client
//
// Requires BRAVE_API_KEY environment variable.
// Returns search snippets suitable for feeding to gwen for answer extraction.

#pragma once

#include <string>
#include <vector>

namespace brave {

struct SearchResult {
    std::string title;
    std::string url;
    std::string description;
};

// Search Brave and return top results. Returns empty on failure.
std::vector<SearchResult> search(const std::string & query, int count = 5);

// Format results as context string for LLM consumption.
std::string format_results(const std::vector<SearchResult> & results);

} // namespace brave
