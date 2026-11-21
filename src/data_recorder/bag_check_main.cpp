// Checks a bag written by //src/data_recorder:record for missing messages.
//
//   bazel run //src/data_recorder:bag_check -- BAG_DIR
//
// The recorder stores each message's publisher sequence number in the MCAP message's sequence field
// (rosbag2's reader doesn't return it, so this reads the MCAP files directly). Per topic, the gaps in the
// sorted sequence numbers are the messages that never made it into the bag, whether lost in DDS or
// dropped by the recorder. A topic with several publishers has interleaved sequences (repeated numbers)
// and can't be checked this way; nor can one without sequence numbers (other recorders write 0).
// Exit code 2 if any checked topic has a gap.
#include <mcap/reader.hpp>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace
{
struct TopicStats
{
  std::string type;
  uint64_t messages = 0;
  bool unsequenced = false;  // a message without a sequence number
  std::vector<uint32_t> sequences;
};
}  // namespace

int main(int argc, char ** argv)
{
  if (argc != 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")
  {
    std::printf("usage: bag_check BAG_DIR\n");
    return argc == 2 ? 0 : 1;
  }
  std::filesystem::path bag = argv[1];
  if (const char * cwd = std::getenv("BUILD_WORKING_DIRECTORY"); cwd && bag.is_relative()) bag = std::filesystem::path(cwd) / bag;

  std::vector<std::filesystem::path> files;
  if (std::filesystem::is_directory(bag))
  {
    for (const auto & entry : std::filesystem::directory_iterator(bag))
      if (entry.path().extension() == ".mcap") files.push_back(entry.path());
  }
  else
  {
    files.push_back(bag);
  }
  if (files.empty())
  {
    std::fprintf(stderr, "No .mcap files in %s\n", bag.c_str());
    return 1;
  }

  std::map<std::string, TopicStats> topics;
  for (const auto & file : files)
  {
    mcap::McapReader reader;
    if (const auto status = reader.open(file.string()); !status.ok())
    {
      std::fprintf(stderr, "%s: %s\n", file.c_str(), status.message.c_str());
      return 1;
    }
    const auto on_problem = [&](const mcap::Status & status)
    { std::fprintf(stderr, "%s: %s\n", file.c_str(), status.message.c_str()); };
    for (const auto & view : reader.readMessages(on_problem))
    {
      auto & stats = topics[view.channel->topic];
      if (stats.type.empty() && view.schema) stats.type = view.schema->name;
      ++stats.messages;
      if (view.message.sequence == 0) stats.unsequenced = true;
      else stats.sequences.push_back(view.message.sequence);
    }
    reader.close();
  }

  bool any_gap = false;
  std::printf("%-56s %8s %8s %8s  %s\n", "topic", "msgs", "missing", "gaps", "");
  for (auto & [name, stats] : topics)
  {
    auto & s = stats.sequences;
    std::sort(s.begin(), s.end());
    const bool repeated = std::adjacent_find(s.begin(), s.end()) != s.end();
    if (stats.unsequenced || repeated || s.size() < 2)
    {
      std::printf(
        "%-56s %8" PRIu64 " %8s %8s  %s\n", name.c_str(), stats.messages, "-", "-",
        stats.unsequenced ? "no sequence numbers" : repeated ? "several publishers: not checked" : "");
      continue;
    }
    uint64_t missing = 0, gaps = 0;
    for (size_t i = 1; i < s.size(); ++i)
    {
      const uint64_t step = static_cast<uint32_t>(s[i] - s[i - 1]);  // wraps at 2^32 like the stored numbers
      if (step > 1)
      {
        ++gaps;
        missing += step - 1;
      }
    }
    any_gap |= missing > 0;
    std::printf("%-56s %8" PRIu64 " %8" PRIu64 " %8" PRIu64 "\n", name.c_str(), stats.messages, missing, gaps);
  }
  std::printf("%s\n", any_gap ? "MISSING MESSAGES" : "No missing messages");
  return any_gap ? 2 : 0;
}
