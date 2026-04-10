// brave_search.cpp — Brave Web Search API client using cpp-httplib

#include "brave_search.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>

namespace brave {

using json = nlohmann::json;

std::vector<SearchResult> search(const std::string & query, int count) {
    const char * api_key = getenv("BRAVE_API_KEY");
    if (!api_key || !api_key[0]) {
        fprintf(stderr, "brave::search: BRAVE_API_KEY not set\n");
        return {};
    }

    httplib::Client cli("https://api.search.brave.com");
    cli.set_connection_timeout(5);
    cli.set_read_timeout(10);

    std::string path = "/res/v1/web/search?q="
                     + httplib::encode_query_component(query, false)
                     + "&count=" + std::to_string(count);

    httplib::Headers headers = {
        {"Accept", "application/json"},
        {"X-Subscription-Token", api_key}
    };

    auto res = cli.Get(path, headers);
    if (!res) {
        fprintf(stderr, "brave::search: connection failed: %s\n",
                httplib::to_string(res.error()).c_str());
        return {};
    }
    if (res->status != 200) {
        fprintf(stderr, "brave::search: HTTP %d\n", res->status);
        return {};
    }

    std::vector<SearchResult> results;
    try {
        auto j = json::parse(res->body);
        if (j.contains("web") && j["web"].contains("results")) {
            for (auto & r : j["web"]["results"]) {
                SearchResult sr;
                if (r.contains("title"))       sr.title       = r["title"].get<std::string>();
                if (r.contains("url"))         sr.url         = r["url"].get<std::string>();
                if (r.contains("description")) sr.description = r["description"].get<std::string>();
                results.push_back(std::move(sr));
            }
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "brave::search: JSON parse error: %s\n", e.what());
    }
    return results;
}

std::string format_results(const std::vector<SearchResult> & results) {
    if (results.empty()) return "No search results found.";

    std::string out;
    for (size_t i = 0; i < results.size(); i++) {
        out += "[" + std::to_string(i + 1) + "] " + results[i].title + "\n";
        auto & desc = results[i].description;
        if (desc.size() > 500) {
            out += desc.substr(0, 500) + "...\n\n";
        } else {
            out += desc + "\n\n";
        }
    }
    return out;
}

} // namespace brave
