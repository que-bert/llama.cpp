#pragma once

#include <string>
#include <utility>
#include <vector>

struct common_params;

// "just works" serving presets: settings picked from the GGUF metadata, explicit flags always win

struct common_serving_preset_result {
    std::string name = "none";
    std::string kind = "none"; // identity | arch-fallback | none
    std::vector<std::pair<std::string, std::string>> applied;
    std::vector<std::pair<std::string, std::string>> skipped; // explicitly set by the user, preset value ignored
};

// reads the GGUF metadata of params.model.path (no tensors) and applies the matching preset to params
// returns kind "none" when the file cannot be read or no preset matches
common_serving_preset_result common_serving_preset_select(common_params & params);

// "preset: <name> (<kind>) applied: k=v ..."
std::string common_serving_preset_summary(const common_serving_preset_result & res);
// multi-line description for --print-preset
std::string common_serving_preset_describe(const common_serving_preset_result & res);
